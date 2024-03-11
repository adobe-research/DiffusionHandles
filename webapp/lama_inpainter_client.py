import time

from gradio_client import Client

class LamaInpainterClient():
    def __init__(self, url: str, timeout_seconds: float = None):
        self.url = url
        self.client = Client(url, upload_files=True, download_files=True)
        self.timeout_seconds = timeout_seconds

    def remove_foreground(self, img_path: str, fg_mask_path: str, dilation: int = 3):
        
        job = self.client.submit(img_path, fg_mask_path, dilation)
        
        job_time = 0
        while not job.done():
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")
        
        bg_path = job.outputs()[0]

        return bg_path

if __name__ == '__main__':
    client = LamaInpainterClient(url="http://localhost:6008/lama_inpainter")
    bg_path = client.remove_foreground(
        img_path="../test/data/photogen/sunflower/input.png",
        fg_mask_path="../test/data/photogen/sunflower/mask.png",
    )
    import os
    import shutil
    os.makedirs('../test/results/webapp/sunflower', exist_ok=True)
    shutil.copyfile(bg_path, "../test/results/webapp/sunflower/bg.png")
    print('done')
