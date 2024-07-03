import time
from typing import Tuple

import gradio_client

class DiffhandlesClient():
    def __init__(self, url: str, timeout_seconds: float = None):
        self.url = url
        self.client = gradio_client.Client(url, download_files=True)
        self.timeout_seconds = timeout_seconds

    def edit_image(
            self, prompt: str, img_path: str, fg_mask_path: str, depth_path: str, bg_depth_path: str,
            rot_angle: float = 0.0, rot_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0),
            translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        
        job = self.client.submit(
            prompt,
            gradio_client.file(img_path), gradio_client.file(fg_mask_path),
            gradio_client.file(depth_path), gradio_client.file(bg_depth_path),
            rot_angle, rot_axis[0], rot_axis[1], rot_axis[2],
            translation[0], translation[1], translation[2])
        
        job_time = 0
        while not job.done():
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")
        
        edited_image_path = job.outputs()[0]

        return edited_image_path

if __name__ == '__main__':
    client = DiffhandlesClient(url="http://localhost:6006/diffhandles")
    edited_image_path = client.edit_image(
        prompt="a sunflower in the garden",
        img_path="data/sunflower/input.png",
        fg_mask_path="data/sunflower/mask.png",
        depth_path="data/sunflower/depth.exr",
        bg_depth_path="data/sunflower/bg_depth.exr",
        rot_angle=15.0,
        rot_axis=(0.0, 1.0, 0.0),
        translation=(0.0, 0.0, 0.0),
    )
    import os
    import shutil
    os.makedirs('results/sunflower', exist_ok=True)
    shutil.copyfile(edited_image_path, "results/sunflower/edit_test.png")
    print('done')
