import numpy.typing as npt
import gradio as gr

from gradio_webapp import GradioWebapp

class Text2imgWebapp(GradioWebapp):

    def __init__(self, netpath: str, port: int):
        super().__init__(netpath, port)

    def generate_image(self, prompt: str = None) -> npt.NDArray:
        raise NotImplementedError

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_text_prompt = gr.Textbox(label="Text Prompt", value="a sunflower in the garden")
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_generated_image = gr.Image(label="Edited Image")

            generate_button.click(
                self.generate_image,
                inputs=[gr_text_prompt],
                outputs=[gr_generated_image])

        return gr_app
