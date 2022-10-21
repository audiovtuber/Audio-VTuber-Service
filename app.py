from argparse import ArgumentParser

import gradio as gr

from model import TalkingFaceTorchScript


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--listen-all",
        action="store_true",
        help="Set this flag to expose gradio on all network interfaces. Useful if running in a docker container",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to listen on",
        default=7860,
    )
    parser.add_argument(
        "--head-image",
        type=str,
        help="Image to load and overlay the animation upon",
        default="commish_mouthy_small.png",
    )
    return parser.parse_args()


def build_streaming_interface(model):
    demo = gr.Interface(
        fn=model.predict_stream,
        inputs=[gr.Microphone(streaming=True), "state"],
        outputs=["audio", "plot", "state"],
        live=True,
    )
    return demo


def build_static_block(predict_fn):
    # TODO: upload flagged logs (and their blobs) somewhere
    flag_callback = gr.CSVLogger()
    with gr.Blocks(css="#instruction_example {width: 300px;}") as demo:
        gr.Markdown(
            """
# Audio VTuber
This app will generate an animated expression using only your voice!
"""
        )
        gr.Image("instruction.png", tool=False, elem_id="instruction_example")
        gr.Markdown(
            """
## How-To, Step-by-Step
1. Press `Record from microphone`. If prompted, allow your browser to access the microphone
2. Say whatever you like! I recommend keeping it under 10 seconds so you don't have to wait long
3. Press `Stop recording`
4. Press the `Run` button and wait patiently for the video to be loaded
5. Play the video :)
"""
        )
        with gr.Row():
            inp = gr.Microphone(label="audio_path", show_label=False)
            with gr.Column():
                out = gr.Video(
                    label="video_path",
                    show_label=False,
                    interactive=False,
                )
                flag_btn = gr.Button("Flag", visible=False)
                flag_thanks = gr.Markdown(
                    "Thanks for flagging and helping us improve our performance!",
                    visible=False,
                )
        btn = gr.Button("Run")

        def predict(args):
            print(f"args: {args}")
            return {
                flag_btn: gr.update(visible=True),
                flag_thanks: gr.update(visible=False),
                out: predict_fn(*args),
            }

        btn.click(
            fn=lambda *args: predict(args),
            inputs=inp,
            outputs=[flag_btn, flag_thanks, out],
        )
        flag_callback.setup([inp, out], "flagged_data_points")

        def flag_and_reset(*args):
            flag_callback.flag(args)
            return {
                flag_btn: gr.update(visible=False),
                flag_thanks: gr.update(visible=True),
            }

        flag_btn.click(
            # lambda *args: flag_callback.flag(args),
            flag_and_reset,
            inputs=[inp, out],
            outputs=[flag_btn, flag_thanks],
            preprocess=False,
        )

        with gr.Row():
            gr.Markdown(
                """
## Feedback
If you're unhappy with the result, please click the `Flag` button to share the audio and video with us so we can improve our model! It will not be used for any other purpose
## About
This app was created as my contribution to the [Full Stack Deep Learning 2022 Cohort](https://fullstackdeeplearning.com/). Code for this app can be found in [github](https://github.com/audiovtuber/Audio-VTuber-Service)
and training code can also be found [in github](https://github.com/audiovtuber/Talking-Face-Landmarks-from-Speech)
## Credit
The training code is a reimplementation of [Generating Talking Face Landmarks from Speech](https://link.springer.com/chapter/10.1007/978-3-319-93764-9_35)
(Eskimez, et. al. ; Original code [here](https://github.com/audiovtuber/Talking-Face-Landmarks-from-Speech))
The [GRID Corpus](https://spandh.dcs.shef.ac.uk//gridcorpus) was also used in the training of the model
"""
            )

    return demo


def main(args):
    # TODO: refactor model loading to pull from wandb or other location(s)
    # TODO: configure mouth offset in UI
    model = TalkingFaceTorchScript(
        model_path="./torchscript_model.pt",
        head_image=args.head_image,
        # mouth_offset=(5, 0),
        # mouth_angle=3,
        # mouth_stretch=-10,
        mouth_offset=(45, 50),
        mouth_angle=13.0,
        mouth_stretch=30,
    )

    # demo = build_static_block(model.predict)
    demo = build_static_block(model.predict_animation)

    # server_name is a workaround to gradio not always exposing the port when working locally in a docker container (e.g. when using WSL2)
    server_name = "0.0.0.0" if args.listen_all else None
    demo.launch(server_name=server_name, server_port=args.port)


if __name__ == "__main__":
    args = parse_args()
    main(args)
