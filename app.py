"""
Gradio web UI for the pool shot legality detector.

Wraps the same run_pipeline() used by main.py - upload a clip, get an
annotated video back plus a plain-language verdict. This is the actual
deliverable: either share the local link directly, or screen-record
yourself using it as a demo video.
"""
import gradio as gr
from src.pipeline import run_pipeline

MODEL_PATH = "best.pt"
DEFAULT_CUE_CLASS = "white ball"


def process_video(video_path, cue_class_name, distance_factor):
    if video_path is None:
        return None, "### Upload a clip first."

    cue_class_name = (cue_class_name or DEFAULT_CUE_CLASS).strip()

    try:
        out_path, summary = run_pipeline(
            model_path=MODEL_PATH,
            video_path=video_path,
            cue_class_name=cue_class_name,
            distance_factor=distance_factor,
            out_path="gradio_output.mp4",
        )
    except Exception as e:
        return None, f"### ⚠️ Error while processing\n```\n{e}\n```"

    if summary.startswith("LEGAL HIT"):
        headline = "## ✅ LEGAL HIT\nThe cue ball made contact with another ball."
    elif summary.startswith("FOUL"):
        headline = "## ❌ FOUL\nThe cue ball made no contact with another ball."
    else:
        headline = "## ⚠️ Could not evaluate"

    details = f"<details><summary>Technical details</summary>\n\n```\n{summary}\n```\n\n</details>"
    return out_path, f"{headline}\n\n{details}"


with gr.Blocks() as demo:
    gr.Markdown("# 🎱 Pool Shot Legality Detector")
    gr.Markdown(
        "Upload a short clip of a pool shot. The system detects the balls, "
        "tracks them, and checks whether the cue ball made legal contact "
        "with another ball.\n\n"
        "*Scope note: this checks contact only (legal hit vs. foul), not "
        "pocketing or full scoring - see README for details.*"
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload a pool shot clip")
            with gr.Accordion("Advanced settings", open=False):
                cue_class_input = gr.Textbox(
                    value=DEFAULT_CUE_CLASS, label="Cue ball class name",
                    info="Must match your model's class name exactly",
                )
                distance_input = gr.Slider(
                    minimum=0.8, maximum=2.5, value=1.3, step=0.1,
                    label="Collision sensitivity (ball-diameters)",
                    info="Lower = stricter about what counts as touching",
                )
            run_btn = gr.Button("Run Detection", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Annotated result")
            result_output = gr.Markdown()

    run_btn.click(
        fn=process_video,
        inputs=[video_input, cue_class_input, distance_input],
        outputs=[video_output, result_output],
    )

if __name__ == "__main__":
    demo.launch()
