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

Optional: The NVCC compiler is needed for the CUDA version that PyTorch is using (12.1 in the example above). If the correct version is not already installed, you can install the Cuda Toolkit for the right version as conda package (this should automatically change the nvcc path when the conda evironment is active):
```bash
conda install cudatoolkit-dev=12.1 -c conda-forge
```

Install Diffusion Handles as editable package:
```bash
pip install -e .
```
