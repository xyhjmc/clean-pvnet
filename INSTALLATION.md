# Installation and Environment Requirements

This project targets GPU training and evaluation of PVNet. The steps below help
set up a repeatable environment on Linux with CUDA-enabled NVIDIA GPUs.

## Prerequisites

- **OS:** Ubuntu 20.04+ (other modern Linux distributions should work with minor changes).
- **Python:** 3.10 (matching the recommended `conda` environment). Earlier Python
  versions are not tested.
- **GPU:** NVIDIA GPU with a recent driver and a CUDA toolkit that matches your
  PyTorch build (e.g., CUDA 12.6 for the commands below).
- **Build tools:** `gcc`, `g++`, and `make` are required for compiling the
  custom CUDA/C++ extensions under `lib/csrc/`.
- **System libraries:**
  - `libglfw3-dev` and `libglfw3`
  - Optional (for uncertainty-driven PnP): `libgoogle-glog-dev`,
    `libsuitesparse-dev`, `libatlas-base-dev`

If you prefer a containerized setup, the repository includes a ready-to-use
Docker configuration under [`docker/`](docker/).

## Create a Python environment

```bash
conda create -n pvnet python=3.10
conda activate pvnet

# Example: install PyTorch and torchvision built for CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install the remaining Python dependencies
pip install -r requirements.txt
```

If you need a different CUDA version or CPU-only binaries, follow the official
PyTorch installation selector and adjust the `pip install torch torchvision`
command accordingly.

## Compile native extensions

Several components rely on CUDA/C++ extensions for performance. Compile them
inside the repo after setting the `CUDA_HOME` path to your toolkit location:

```bash
ROOT=/path/to/clean-pvnet
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-12.6"

cd nn && python setup.py build_ext --inplace
cd ../fps && python setup.py build_ext --inplace

# The DCN-based heads rely on torchvision's native
# `deform_conv2d`/`DeformRoIPool` kernels, so no extra build is required for
# deformable convolutions.

# Optional: required for uncertainty-driven PnP
cd ../uncertainty_pnp
python setup.py build_ext --inplace
```

The RANSAC voting module uses the pure PyTorch implementation at
`lib/csrc/ransac_voting/ransac_voting_gpu.py` and does not require a separate
build step.

## Dataset symlinks

Create symbolic links under `data/` that point to your local copies of the
required datasets:

```bash
ROOT=/path/to/clean-pvnet
cd $ROOT/data
ln -s /path/to/linemod linemod
ln -s /path/to/linemod_orig linemod_orig
ln -s /path/to/occlusion_linemod occlusion_linemod
ln -s /path/to/tless tless
ln -s /path/to/cache cache
ln -s /path/to/SUN2012pascalformat sun
```

Refer to the README for dataset download links and usage notes.

## Quick validation

After installation, ensure PyTorch detects your GPU and the extensions load:

```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
PY
```

You can also run a short script to import PVNet modules:

```bash
python - <<'PY'
import importlib
for mod in [
    'lib.csrc.nn._ext',
    'lib.csrc.fps._ext',
]:
    try:
        importlib.import_module(mod)
        print(mod, 'loaded')
    except Exception as exc:
        print(mod, 'unavailable:', exc)
PY
```

If optional modules are missing, verify that you built them and that `CUDA_HOME`
points to the correct toolkit installation.
