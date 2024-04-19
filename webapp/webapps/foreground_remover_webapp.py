import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy.typing as npt
import gradio as gr

from gradio_webapp import GradioWebapp

class ForegroundRemoverWebapp(GradioWebapp):

    def __init__(self, netpath: str, port: int):
        super().__init__(netpath=netpath, port=port)

    def remove_foreground(self, img: npt.NDArray = None, fg_mask: npt.NDArray = None, dilation: int = 3) -> npt.NDArray:
        raise NotImplementedError

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_input_image = gr.Image(label="Input Image")
                    gr_fg_mask = gr.Image(label="Foreground Mask")
                    gr_dilation = gr.Number(label="Forground Mask Dilation", precision=0, value=3, minimum=0, maximum=100)
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_bg = gr.Image(label="Background")

            generate_button.click(
                self.remove_foreground,
                inputs=[gr_input_image, gr_fg_mask, gr_dilation],
                outputs=[gr_bg])

        return gr_app
