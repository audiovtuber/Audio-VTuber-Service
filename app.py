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
        '--port',
        type=int,
        help="Port to listen on",
        default=7860,
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
    with gr.Blocks() as demo:
        gr.Markdown("Record some audio and see the animation")
        with gr.Row():
            inp = gr.Microphone(label="audio_path", show_label=False)
            with gr.Column():
                # NOTE: for whatever reason, regular videos get mirrored by gradio once we add a second button for flagging. Hacky fix: pass `mirror_webcam=False`
                out = gr.Video(
                    label="video_path", show_label=False, mirror_webcam=False
                )
                flag_btn = gr.Button("Flag", visible=False)
        btn = gr.Button("Run")

        def predict(args):
            print(f"args: {args}")
            return {flag_btn: gr.update(visible=True), out: predict_fn(*args)}

        btn.click(fn=lambda *args: predict(args), inputs=inp, outputs=[flag_btn, out])
        flag_callback.setup([inp, out], "flagged_data_points")
        flag_btn.click(
            lambda *args: flag_callback.flag(args),
            inputs=[inp, out],
            outputs=None,
            preprocess=False,
        )

    return demo


def main(args):
    # TODO: refactor model loading to pull from wandb or other location(s)
    # TODO: configure mouth offset in UI
    model = TalkingFaceTorchScript(
        model_path="./torchscript_model.pt",
        head_image="commish_mouthy_small.png",
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
