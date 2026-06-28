# ============================================================
# Pool Shot Detector - one-shot project setup script (PowerShell)
# Creates folder structure, writes all source files, sets up a
# Python 3.11 virtual environment, and installs dependencies.
# ============================================================

$ProjectName = "pool_shot_detector"

Write-Host "Creating project folder: $ProjectName" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$ProjectName\src" | Out-Null

Write-Host "Writing requirements.txt" -ForegroundColor Cyan
@'
ultralytics>=8.2.0
opencv-python>=4.9.0
numpy>=1.26.0
gradio>=4.0.0
'@ | Set-Content -Path "$ProjectName\requirements.txt" -Encoding UTF8

Write-Host "Writing main.py" -ForegroundColor Cyan
@'
"""
CLI entrypoint — thin wrapper around src/pipeline.py.

Usage:
  python main.py --model best.pt --video clip.mp4 --cue-class "white ball" --out output_annotated.mp4
"""
import argparse
from src.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained YOLO weights (.pt)")
    ap.add_argument("--video", required=True, help="Path to input video clip")
    ap.add_argument("--cue-class", default="white ball",
                    help="Cue-ball class name exactly as your model labels it")
    ap.add_argument("--out", default="output_annotated.mp4")
    ap.add_argument("--distance-px", type=float, default=28.0)
    args = ap.parse_args()

    out_path, summary = run_pipeline(
        model_path=args.model,
        video_path=args.video,
        cue_class_name=args.cue_class,
        distance_px=args.distance_px,
        out_path=args.out,
    )
    print(summary)
    print(f"Annotated video saved to {out_path}")


if __name__ == "__main__":
    main()
'@ | Set-Content -Path "$ProjectName\main.py" -Encoding UTF8

Write-Host "Writing src\__init__.py" -ForegroundColor Cyan
@'

'@ | Set-Content -Path "$ProjectName\src\__init__.py" -Encoding UTF8

Write-Host "Writing src\perception.py" -ForegroundColor Cyan
@'
"""
Ball detection + tracking wrapper around Ultralytics YOLO.

Runs YOLO's built-in tracker (ByteTrack, shipped with ultralytics — no
separate DeepSORT integration needed) frame by frame and returns a clean,
pipeline-friendly representation of tracked balls.
"""
from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter

from ultralytics import YOLO


@dataclass
class TrackedBall:
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]              # (cx, cy)
    confidence: float
    frame_idx: int


class BallPerception:
    def __init__(self, model_path: str, tracker_cfg: str = "bytetrack.yaml", conf: float = 0.35):
        """
        model_path: path to a trained YOLO .pt checkpoint for pool balls.
        tracker_cfg: "bytetrack.yaml" or "botsort.yaml" (both ship with ultralytics).
        conf: detection confidence threshold.
        """
        self.model = YOLO(model_path)
        self.tracker_cfg = tracker_cfg
        self.conf = conf

    def track_video(self, video_path: str):
        """
        Generator that yields (frame_idx, frame_bgr, List[TrackedBall]) for every frame
        in the video, in order.
        """
        results_gen = self.model.track(
            source=video_path,
            tracker=self.tracker_cfg,
            conf=self.conf,
            stream=True,
            persist=True,
            verbose=False,
        )
        for frame_idx, result in enumerate(results_gen):
            frame = result.orig_img
            balls: List[TrackedBall] = []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names
                for box, tid, conf, cls in zip(boxes, ids, confs, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    balls.append(
                        TrackedBall(
                            track_id=int(tid),
                            class_name=names[int(cls)],
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            center=(float(cx), float(cy)),
                            confidence=float(conf),
                            frame_idx=frame_idx,
                        )
                    )
            yield frame_idx, frame, balls


def find_cue_track_id(all_balls_by_frame: List[List[TrackedBall]], cue_class_name: str):
    """
    Pick the track id most frequently classified as the cue-ball class
    across the whole clip. Returns None if that class never appears —
    which usually means your dataset's class names don't match what
    you passed in (check data.yaml).
    """
    counts: Counter = Counter()
    for balls in all_balls_by_frame:
        for b in balls:
            if b.class_name.lower() == cue_class_name.lower():
                counts[b.track_id] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]
'@ | Set-Content -Path "$ProjectName\src\perception.py" -Encoding UTF8

Write-Host "Writing src\collision.py" -ForegroundColor Cyan
@'
"""
Heuristic collision detection from tracked ball trajectories.

Why not just "distance below threshold"?
Two balls can pass close to each other on screen (due to perspective or
near-misses) without touching. To cut false positives, a collision is only
flagged when proximity AND a physical reaction are both present:

  1. Build a per-track history of (frame_idx, center) positions.
  2. Compute per-frame velocity vectors for each track.
  3. Flag a candidate collision between two tracks when:
       a) their centers come within `distance_px` of each other, AND
       b) within the next `velocity_window` frames, at least one of the
          two balls shows a sudden change in speed and/or direction —
          i.e. an actual collision response, not just a near-miss.

Tune `distance_px` to your video's resolution/ball size — start by
checking a ball's bbox width in pixels and set distance_px to roughly
1x-1.5x that.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class CollisionEvent:
    frame_idx: int
    track_id_a: int
    track_id_b: int
    distance_px: float


class CollisionDetector:
    def __init__(
        self,
        distance_px: float = 28.0,
        velocity_window: int = 3,
        direction_change_deg: float = 25.0,
        speed_change_ratio: float = 0.3,
    ):
        self.distance_px = distance_px
        self.velocity_window = velocity_window
        self.direction_change_deg = direction_change_deg
        self.speed_change_ratio = speed_change_ratio
        self.history: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

    def update(self, frame_idx: int, balls) -> None:
        """Call once per frame with that frame's TrackedBall list, before detect()."""
        for b in balls:
            self.history.setdefault(b.track_id, []).append((frame_idx, b.center))

    def _velocity(self, track_id: int, frame_idx: int) -> Optional[Tuple[float, float]]:
        pts = self.history.get(track_id, [])
        idx = next((i for i, (f, _) in enumerate(pts) if f == frame_idx), None)
        if idx is None or idx == 0:
            return None
        (f0, p0), (f1, p1) = pts[idx - 1], pts[idx]
        if f1 == f0:
            return None
        return ((p1[0] - p0[0]) / (f1 - f0), (p1[1] - p0[1]) / (f1 - f0))

    @staticmethod
    def _speed_and_angle(v: Tuple[float, float]) -> Tuple[float, float]:
        speed = (v[0] ** 2 + v[1] ** 2) ** 0.5
        angle = np.degrees(np.arctan2(v[1], v[0]))
        return speed, angle

    def _reacted(self, track_id: int, frame_idx: int) -> bool:
        """Did this ball noticeably change speed/direction shortly after frame_idx?"""
        v_before = self._velocity(track_id, frame_idx)
        if v_before is None:
            return False
        speed_before, angle_before = self._speed_and_angle(v_before)
        for offset in range(1, self.velocity_window + 1):
            v_after = self._velocity(track_id, frame_idx + offset)
            if v_after is None:
                continue
            speed_after, angle_after = self._speed_and_angle(v_after)
            angle_diff = abs((angle_after - angle_before + 180) % 360 - 180)
            speed_diff_ratio = abs(speed_after - speed_before) / (speed_before + 1e-6)
            if angle_diff > self.direction_change_deg or speed_diff_ratio > self.speed_change_ratio:
                return True
        return False

    def detect(self, frame_idx: int, balls) -> List[CollisionEvent]:
        """Call once per frame, after update(), with that same frame's ball list."""
        events: List[CollisionEvent] = []
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                a, b = balls[i], balls[j]
                dist = ((a.center[0] - b.center[0]) ** 2 + (a.center[1] - b.center[1]) ** 2) ** 0.5
                if dist <= self.distance_px:
                    if self._reacted(a.track_id, frame_idx) or self._reacted(b.track_id, frame_idx):
                        events.append(CollisionEvent(frame_idx, a.track_id, b.track_id, dist))
        return events
'@ | Set-Content -Path "$ProjectName\src\collision.py" -Encoding UTF8

Write-Host "Writing src\pipeline.py" -ForegroundColor Cyan
@'
"""
Shared pipeline logic, used by both main.py (CLI) and app.py (Gradio UI).
Keeping this in one place means you only fix bugs once.
"""
import os
import tempfile
import cv2

from .perception import BallPerception, find_cue_track_id
from .collision import CollisionDetector
from .rules import RuleEngine


def draw_frame(frame, balls, had_collision_this_frame, verdict_text=None):
    for b in balls:
        x1, y1, x2, y2 = map(int, b.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{b.class_name}#{b.track_id}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )
    if had_collision_this_frame:
        cv2.putText(frame, "COLLISION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if verdict_text:
        cv2.putText(
            frame, verdict_text, (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )
    return frame


def run_pipeline(model_path: str, video_path: str, cue_class_name: str = "white ball",
                  distance_px: float = 28.0, out_path: str = None):
    """
    Runs the full video -> detection+tracking -> collision -> rule engine pipeline.

    Returns:
        (output_video_path, verdict_summary_text)
    """
    if out_path is None:
        out_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")

    perception = BallPerception(model_path)
    detector = CollisionDetector(distance_px=distance_px)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_frames, all_balls_by_frame, all_events = [], [], []

    for frame_idx, frame, balls in perception.track_video(video_path):
        detector.update(frame_idx, balls)
        events = detector.detect(frame_idx, balls)
        all_frames.append(frame)
        all_balls_by_frame.append(balls)
        all_events.append(events)

    cue_id = find_cue_track_id(all_balls_by_frame, cue_class_name)

    if cue_id is None:
        summary = (f"WARNING: no ball matched class '{cue_class_name}'. "
                   f"Check the class name matches your model's labels exactly.")
        verdict_text = "ERROR: cue ball not found"
    else:
        engine = RuleEngine(cue_id)
        flat_events = [e for events in all_events for e in events]
        verdict = engine.evaluate(flat_events)
        verdict_text = "LEGAL HIT" if verdict.legal else "FOUL: no contact"
        summary = f"{verdict_text} — {verdict.reason}"
        if verdict.legal:
            summary += f" (first contact at frame {verdict.first_contact_frame})"

    for frame, balls, events in zip(all_frames, all_balls_by_frame, all_events):
        annotated = draw_frame(frame, balls, bool(events), verdict_text)
        writer.write(annotated)

    writer.release()
    return out_path, summary
'@ | Set-Content -Path "$ProjectName\src\pipeline.py" -Encoding UTF8

Write-Host "Writing src\rules.py" -ForegroundColor Cyan
@'
"""
Minimal 8-ball rule engine — MVP scope only.

Deliberately scoped to "legal contact" vs "foul (no contact)", matching
the project's stated MVP (pocket detection, ball-type fouls, and full
scoring logic are future extensions, not part of this pass). Calling the
output "point" anywhere downstream is misleading: hitting another ball is
necessary-but-not-sufficient for scoring in real 8-ball — keep the
verdict labeled as legality, not points, unless pocket detection is added.
"""
from dataclasses import dataclass
from typing import List, Optional

from .collision import CollisionEvent


@dataclass
class ShotVerdict:
    legal: bool
    reason: str
    first_contact_frame: Optional[int] = None
    first_contact_with: Optional[int] = None


class RuleEngine:
    def __init__(self, cue_track_id: int):
        self.cue_track_id = cue_track_id

    def evaluate(self, events: List[CollisionEvent]) -> ShotVerdict:
        cue_events = [e for e in events if self.cue_track_id in (e.track_id_a, e.track_id_b)]
        if not cue_events:
            return ShotVerdict(legal=False, reason="Cue ball made no contact with another ball.")
        first = min(cue_events, key=lambda e: e.frame_idx)
        other_id = first.track_id_b if first.track_id_a == self.cue_track_id else first.track_id_a
        return ShotVerdict(
            legal=True,
            reason="Cue ball made contact with another ball.",
            first_contact_frame=first.frame_idx,
            first_contact_with=other_id,
        )
'@ | Set-Content -Path "$ProjectName\src\rules.py" -Encoding UTF8

# ------------------------------------------------------------------
# Virtual environment setup (Python 3.11)
# ------------------------------------------------------------------
Write-Host ""
Write-Host "Setting up Python 3.11 virtual environment..." -ForegroundColor Cyan
Push-Location $ProjectName

$usedPy311 = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.11 --version > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        py -3.11 -m venv venv
        $usedPy311 = $true
    }
}
if (-not $usedPy311) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Host "Python 3.11 not found via 'py -3.11'. Falling back to default 'python'." -ForegroundColor Yellow
        Write-Host "Check your version with: python --version  (3.10-3.12 all work fine)" -ForegroundColor Yellow
        python -m venv venv
    } else {
        Write-Host "ERROR: No Python installation found on PATH." -ForegroundColor Red
        Write-Host "Install Python 3.11 from https://www.python.org/downloads/ (check 'Add to PATH' during install), then re-run this script." -ForegroundColor Red
        Pop-Location
        exit 1
    }
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

Write-Host "Installing dependencies from requirements.txt (this can take a few minutes)..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

Pop-Location

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Project created at: .\$ProjectName"
Write-Host ""
Write-Host "Still needed before you can run the pipeline:" -ForegroundColor Yellow
Write-Host "  1. Copy your trained 'best.pt' into .\$ProjectName\"
Write-Host "  2. Copy your test video (e.g. test_clip.mp4) into .\$ProjectName\"
Write-Host ""
Write-Host "Next time you open a new terminal, re-activate the venv with:"
Write-Host "  cd $ProjectName"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run the pipeline with:"
Write-Host "  python main.py --model best.pt --video test_clip.mp4 --cue-class `"white ball`" --out output_annotated.mp4"