import time

import gradio_client

class ForegroundSelectorClient():
    def __init__(self, url: str, timeout_seconds: float = None):
        self.url = url
        self.client = gradio_client.Client(url, download_files=True)
        self.timeout_seconds = timeout_seconds

    def estimate_segment(self, img_path: str, prompt: str):
        
        job = self.client.submit(
            gradio_client.file(img_path),
            prompt)
        
        job_time = 0
        while not job.done():
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")
        
        mask_path = job.outputs()[0]

        return mask_path

if __name__ == '__main__':
    client = ForegroundSelectorClient(url="http://localhost:6010/foreground_selector")
    mask_path = client.estimate_segment(
        img_path="data/sunflower/input.png",
        prompt="sunflower",
    )
    import os
    import shutil
    os.makedirs('results/sunflower', exist_ok=True)
    shutil.copyfile(mask_path, "results/sunflower/mask.png")
    print('done')
