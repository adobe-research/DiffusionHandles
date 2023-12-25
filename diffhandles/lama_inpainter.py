import os

from diffhandles.inpainter import Inpainter

class LaMaInpainter(Inpainter):
    def __init__(self):
        super().__init__()
        
        # download pre-trained model
        if not (
            os.path.exists(os.path.expanduser('~/.cache/lama/config.yaml')) and
            os.path.exists(os.path.expanduser('~/.cache/lama/models/best.ckpt'))):

            os.makedirs(os.path.expanduser('~/.cache/lama'), exist_ok=True)
            with urlopen('https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip') as remote_file:
                with ZipFile(BytesIO(remote_file.read())) as zfile:
                    zfile.extractall(os.path.expanduser('~/.cache/lama'))

        predict_config.model.path = os.path.expanduser('~/.cache/lama/big-lama')