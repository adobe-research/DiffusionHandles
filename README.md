# DiffusionHandles

## Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html)

## Install as Package (Experimental)

Install Diffusion Handles as pip package.

```bash
pip install git+https://github.com/karranpandey/diffusionhandles.git
```
TODO: make sure no dependencies are missing

## Install for Development

Also installs dependencies required to run test scripts and notebooks in the `test` folder. This should allow running all scripts and notebooks in the test folder.

Create a conda environment:
```bash
conda create -n diffusionhandles python=3.9
conda activate diffusionhandles
```

> **Optional Explicit CUDA & PyTorch Installation**
>
> If a CUDA version compatible with PyTorch is not installed on your system, install PyTorch with conda to make sure you have a CUDA version that works with PyTorch:
> ```bash
> conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
> ```
>
> If a suitable CUDA dev environment including nvcc is not installed on your system, install CUDA dev environment matching the CUDA runtime version:
> ```bash
> conda install cuda-libraries-dev=12.1 cuda-nvcc=12.1 cuda-nvtx=12.1 cuda-cupti=12.1 -c nvidia
> ```

Clone the Diffusion Handles repository:
```bash
git clone https://github.com/karranpandey/diffusionhandles.git
```

Install as editable package with development dependencies:
```bash
cd diffusionhandles
pip install -e .[dev]
```

## Install as Web App (Experimental)

Follow the steps to install for development, replacing the following step to use the `webapp` extras instead of `dev` extras:
```bash
cd diffusionhandles
pip install -e .[webapp]
```

Start the server (use [tmux](https://github.com/tmux/tmux/wiki) or similar to keep the server running in the background):
```bash
cd webapp
python diffhandles_webapp.py
```
