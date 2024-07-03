import time
from typing import Tuple

import gradio_client

class Text2imgClient():
    def __init__(self, url: str, timeout_seconds: float = None):
        self.url = url
        self.client = gradio_client.Client(url, download_files=True)
        self.timeout_seconds = timeout_seconds

    def text2img(self, prompt: str):
        
        job = self.client.submit(prompt)

        job_time = 0
        while not job.done():
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")
        
        edited_image_path = job.outputs()[0]

        return edited_image_path

if __name__ == '__main__':
    client = Text2imgClient(url="http://localhost:6006/text2img")
    edited_image_path = client.text2img(
        prompt="a sunflower in the garden"
    )
    import os
    import shutil
    os.makedirs('results/sunflower', exist_ok=True)
    shutil.copyfile(edited_image_path, "results/sunflower/text2img_test.png")
    print('done')
