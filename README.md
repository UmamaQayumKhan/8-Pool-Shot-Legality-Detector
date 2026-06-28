# Pool Shot Legality Detector

A computer vision pipeline that watches a video of an 8-ball pool shot and
determines whether the cue ball made **legal contact** with another ball,
or whether the shot was a **foul** (no contact at all).

## video demo


https://github.com/user-attachments/assets/b1738897-f850-41a4-b8d0-01fa01c89aa9


https://github.com/user-attachments/assets/63fddf98-cf7d-48dd-9aec-6e5a5bc58ba0



https://github.com/user-attachments/assets/4d91e092-781b-428a-ae28-c476dd5dd0f3




## Pipeline
Video --> Ball Detection (YOLOv8s) --> Tracking (ByteTrack)

--> Collision Detection --> Rule Engine --> Legal Hit / Foul
A pipeline of separate, swappable stages was used instead of one end-to-end
model, so each stage can be debugged, tuned, and explained independently -
this matters more than raw accuracy when the project also needs to be
demoed and defended.

## Scope - what this does and does not check

This checks **one thing**: did the cue ball touch another ball at all?

- Cue ball touches another ball -> **Legal Hit**
- Cue ball touches nothing -> **Foul**

It does **not** check pocketing, scoring, scratches, wrong-ball-first
fouls, or rail-contact rules. Those are real 8-ball rules but are out of
scope for this version - see Future Work below.

## Model

- **Base model:** YOLOv8s (Ultralytics), fine-tuned
- **Dataset:** [Billiard Pool](https://universe.roboflow.com/billiard-ball-data-set/billiard-pool)
  (Roboflow), 746 images, 12 classes (9 numbered balls, white ball, pool
  table, rack)
- **Training:** 100 epochs, augmentation enabled (rotation, scale,
  perspective, HSV jitter, flips) for robustness to camera angle and
  lighting variation
- **Validation results:** overall mAP50 0.948; cue ball (white ball)
  specifically: precision 0.93, recall 0.985, mAP50 0.971

## Project structure
pool_shot_detector/

src/

perception.py   - YOLO detection + ByteTrack tracking, cue-ball identification

collision.py     - proximity-based collision detection (resolution-independent)

rules.py         - legal hit / foul rule engine

pipeline.py       - wires the above together, used by both main.py and app.py

main.py             - command-line entry point

app.py              - Gradio web UI

best.pt             - trained model weights

requirements.txt
## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### Command line

```powershell
python main.py --model best.pt --video clip.mp4 --cue-class "white ball" --out result.mp4
```

### Web UI

```powershell
python app.py
```

Opens a local web page - upload a clip, get an annotated video back plus a
plain-language verdict.

## How collision detection works

Two balls are flagged as touching when their centers come within
`distance_factor` (default 1.3) times the average ball diameter measured
in that same frame. Measuring the threshold relative to ball size (rather
than a fixed pixel count) means it automatically scales correctly whether
the input video is a small clip or a 4K recording.

## How cue-ball identification works

The cue ball's tracked identity can fragment across a clip - fast motion
right after a strike causes brief detection gaps, and the tracker assigns
a new ID once it reappears. Rather than picking one "winner" ID for the
whole video, the system collects a **set** of plausibly-cue-ball track IDs
(tracks consistently classified as the cue ball class, or whose pixels are
near-pure white as a fallback) and treats a collision as valid if it
involves *any* ID in that set.

## Known limitations

- **Contact-only scoring.** A "legal hit" here means contact happened, not
  that a ball was legally pocketed - see Scope above.
- **Camera motion sensitivity.** Significant camera movement (panning,
  dolly shots) can disrupt tracking continuity more than a static camera.
- **Domain gap.** The model is trained on real photographed pool tables;
  accuracy on visually different footage (e.g. rendered/game graphics, very
  different lighting or table colors) may vary and hasn't been exhaustively
  validated.
- **Assumes exactly one active cue ball.** Footage with multiple white
  balls on the table at once (e.g. carom/practice drills) is out of scope -
  the system has no way to disambiguate which one is "the" cue ball.

## Future work

- Pocket detection and real scoring (potted ball = point)
- Additional foul rules: scratches, wrong-ball-first, no-rail-contact
- Broader training data across more table/lighting styles for generalization
- Real-time camera input (not required by the current use case, which is
  evaluating pre-recorded clips)

## Tech stack

Python, YOLOv8 (Ultralytics), OpenCV, ByteTrack, Gradio, imageio/ffmpeg
