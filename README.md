# DiffusionHandles

## Installation

### Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) with nvcc compiler. The version needs to be compatible with [PyTorch](https://pytorch.org/).

### Installation Steps

Create conda environment:
```bash
conda create -n diffusionhandles python=3.9
conda activate diffusionhandles
```

Install PyTorch using the CUDA version you have installed on your system (we use 12.1 as example here):
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

Clone the Diffusion Handles repository:
```bash
git clone https://github.com/karranpandey/diffusionhandles.git
cd diffusionhandles
```

Install Diffusion Handles as editable package:
```bash
pip install -e .
```