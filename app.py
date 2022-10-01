from argparse import ArgumentParser

import gradio as gr

from model import TalkingFaceTorchScript


def greet(name):
    return "Hello " + name + "!"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--listen-all", action="store_true", help="Set this flag to expose gradio on all network interfaces. Useful if running in a docker container")
    return parser.parse_args()
    
def main(args):
    # TODO: refactor model loading to pull from wandb or other location(s)
    model = TalkingFaceTorchScript(model_path='./torchscript_model.pt')

    with gr.Blocks() as demo:
        gr.Markdown("Record some audio and see the predicted face shape")
        with gr.Row():
            inp = gr.Microphone()
            out = gr.Textbox()
        btn = gr.Button("Run")
        btn.click(fn=model.predict, inputs=inp, outputs=out)

    # demo = gr.Interface(fn=model.predict, inputs=gr.Microphone(), outputs="text")
    # server_name is a workaround to gradio not always exposing the port when working locally in a docker container (e.g. when using WSL2)
    server_name = "0.0.0.0" if args.listen_all else None
    demo.launch(server_name=server_name)

if __name__ == '__main__':
    args = parse_args()
    main(args)