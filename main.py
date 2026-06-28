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
    ap.add_argument("--distance-factor", type=float, default=1.3,
                     help="Collision threshold as a multiple of ball diameter (resolution-independent)")
    args = ap.parse_args()

    out_path, summary = run_pipeline(
        model_path=args.model,
        video_path=args.video,
        cue_class_name=args.cue_class,
        distance_factor=args.distance_factor,
        out_path=args.out,
    )
    print(summary)
    print(f"Annotated video saved to {out_path}")


if __name__ == "__main__":
    main()
