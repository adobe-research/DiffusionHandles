import sys

import gradio as gr
from fastapi import FastAPI
import uvicorn

class GradioWebapp:
    def __init__(self, netpath: str, port: int):
        self.netpath = netpath
        self.port = port

    def build_gradio_app(self):
        raise NotImplementedError

    def cleanup(self):
        # in the derived classes, put any cleanup tasks after the server has stopped here
        pass
    
    def start(self):

        gr_app = self.build_gradio_app()
        gr_app = gr_app.queue()

        app = FastAPI()
        # app = gr.mount_gradio_app(app, gr_app, path=self.netpath)
        app = gr.mount_gradio_app(app, gr_app, path='/', root_path=self.netpath)

        try:
            uvicorn.run(app, host="0.0.0.0", port=self.port)
        except KeyboardInterrupt:
            self.cleanup()
            sys.exit()
