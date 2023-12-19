# DiffusionHandles

## Installation

Create environment:
```bash
conda create -n diffusionhandles python=3.9
conda activate diffusionhandles
```

Optional: Install Cuda Toolkit if the correct nvcc version is not already installed:
(The nvcc cuda version needs to match the pytorch version you are going to install latern on.)
```bash
conda install cuda-toolkit=11.7 -c nvidia
```

(Just as reminder, when using multiple cuda versions, make sure you have the right one active, for example with `update-alternatives`:)
```bash
sudo update-alternatives --config cuda
```

Install PyTorch:
(The PyTorch cuda version needs to match the nvcc cuda version you installed earlier)
(PyTorch<2.0 is required by ZoeDepth - updating it to work with PyTorch >= 2.0 would probably be easy, but would require modifying the ZoeDepth source code)
```bash
pip install torch==1.13 torchvision
```

Install Diffusion Handles as editable package:
```bash
pip install -e .
```

TODO: create internal repo for ZoeDepth fixed for PyTorch >= 2.0 and install it from there
TODO: create internal repo for lang-sam with fixed dependencies (huggingface-hub version can also be larger) and install it from there