# DiffusionHandles

## Installation

Create conda environment:
```bash
conda create -n diffusionhandles python=3.9
conda activate diffusionhandles
```

Install PyTorch:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```
The NVCC compiler is needed for the CUDA version that PyTorch is using (12.1 in the example above). Make sure `nvcc --version` shows the same CUDA version that you used to install PyTorch, otherwise install CUDA as described here for example: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html

Clone the Diffusion Handles repository:
```bash
git clone https://github.com/karranpandey/diffusionhandles.git
cd diffusionhandles
```

Install Diffusion Handles as editable package:
```bash
pip install -e .
```
