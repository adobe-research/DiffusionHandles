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

Start the full Diffusion Handles Pipeline WebApp in [tmux](https://github.com/tmux/tmux/wiki), where `netpath` is the base network path from the root of the server (for example `/demo` for a server at `https://my_server.com/demo`):
```bash
sudo apt install tmux
tmux
cd webapp
source start_webapps_in_tmux.sh <netpath>
```
Check `start_webapps_in_tmux.sh` to adjust configuration details like the distribution of ports and GPUs among services.
