import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy.typing as npt
import gradio as gr
from gradio_hdrimage import HDRImage

from gradio_webapp import GradioWebapp

class DepthEstimatorWebapp(GradioWebapp):

    def __init__(self, netpath: str, port: int):
        super().__init__(netpath, port)

    def estimate_depth(self, img: npt.NDArray = None) -> npt.NDArray:
        raise NotImplementedError

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_input_image = gr.Image(label="Input Image")
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_depth = HDRImage(label="Depth")

            generate_button.click(
                self.estimate_depth,
                inputs=[gr_input_image],
                outputs=[gr_depth])

        return gr_app
