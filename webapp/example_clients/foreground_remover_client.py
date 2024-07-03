import time

import gradio_client

class ForegroundRemoverClient():
    def __init__(self, url: str, timeout_seconds: float = None):
        self.url = url
        self.client = gradio_client.Client(url, download_files=True)
        self.timeout_seconds = timeout_seconds

    def remove_foreground(self, img_path: str, fg_mask_path: str, dilation: int = 3):
        
        job = self.client.submit(
            gradio_client.file(img_path), gradio_client.file(fg_mask_path),
            dilation)
        
        job_time = 0
        while not job.done():
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")
        
        bg_path = job.outputs()[0]

        return bg_path

if __name__ == '__main__':
    # client = ObjectPeelingClient(url="http://localhost:6008/foreground_remover")
    client = ForegroundRemoverClient(url="http://localhost:8891/g3i-dev/guerrero-dev/lama_inpainter")
    bg_path = client.remove_foreground(
        img_path="data/cube_toy/input.png",
        fg_mask_path="data/cube_toy/mask.png",
    )
    import os
    import shutil
    os.makedirs('results/cube_toy', exist_ok=True)
    shutil.copyfile(bg_path, "results/cube_toy/bg.png")
    print('done')
