import numpy.typing as npt
import gradio as gr

from gradio_webapp import GradioWebapp

class ForegroundSelectorWebapp(GradioWebapp):
    def __init__(self, netpath: str, port: int):
        super().__init__(netpath, port)

    def select_foreground(self, img: npt.NDArray = None, prompt: str = None):
        raise NotImplementedError

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_input_image = gr.Image(label="Input Image", value="data/sunflower/input.png")
                    gr_segment_prompt = gr.Textbox(label="Segment Prompt", value="sunflower")
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_bg = gr.Image(label="Background")

            generate_button.click(
                self.select_foreground,
                inputs=[gr_input_image, gr_segment_prompt],
                outputs=[gr_bg])

        return gr_app
