# DiffusionHandles

## Installation

### Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) with nvcc compiler. The version needs to be compatible with [PyTorch](https://pytorch.org/).

### Install as Package (Experimental)

Install Diffusion Handles as pip package:
```bash
pip install git+https://github.com/karranpandey/diffusionhandles.git
```
TODO: Make sure this installs all necessary dependencies.

### Install for Development

Create conda environment:
```bash
conda create -n diffusionhandles python=3.9
conda activate diffusionhandles
```

**Optional:** Install PyTorch explicitly, so you can select the CUDA version you have installed on your system (check their [webpage](https://pytorch.org/) for selecting a cuda version). Using the latest CUDA version, for example:
```bash
pip install torch torchvision
```

Clone the Diffusion Handles repository:
```bash
git clone https://github.com/karranpandey/diffusionhandles.git
```

Install as editable package with development dependencies:
```bash
cd diffusionhandles
pip install -e .[dev]
```
