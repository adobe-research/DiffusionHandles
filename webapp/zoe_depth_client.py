import time

from gradio_client import Client

class ZoeDepthClient():
    def __init__(self, url: str, timeout_seconds: float = None):
        self.url = url
        self.client = Client(url, upload_files=True, download_files=True)
        self.timeout_seconds = timeout_seconds

    def estimate_depth(self, img_path: str):

        job = self.client.submit(img_path)

        job_time = 0
        while not job.done():
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")

        depth_path = job.outputs()[0]

        return depth_path

if __name__ == '__main__':
    client = ZoeDepthClient(url="http://localhost:6007/zoe_depth")
    depth_path = client.estimate_depth(
        img_path="../test/data/photogen/sunflower/input.png"
    )
    import os
    import shutil
    os.makedirs('../test/results/webapp/sunflower', exist_ok=True)
    shutil.copyfile(depth_path, "../test/results/webapp/sunflower/depth.exr")
    print('done')
