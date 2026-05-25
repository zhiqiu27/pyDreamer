#!/bin/bash
# Setup pydreamer conda environment
# Usage: bash setup_env.sh [cuda_version]
#   cuda_version: 118 (default) or 121
# Example: bash setup_env.sh 121

CUDA=${1:-118}
ENV_NAME=pydreamer

echo "==> Creating conda environment: $ENV_NAME (Python 3.10)"
conda create -n $ENV_NAME python=3.10 -y

echo "==> Installing PyTorch (CUDA $CUDA)"
conda run -n $ENV_NAME pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu${CUDA}

echo "==> Installing requirements"
conda run -n $ENV_NAME pip install -r requirements.txt

echo "==> Installing pydreamer"
conda run -n $ENV_NAME pip install -e .

echo "==> Patching gym for NumPy 2.0 compatibility"
GYM_PATH=$(conda run -n $ENV_NAME python -c "import gym, os; print(os.path.dirname(gym.__file__))")
sed -i 's/np\.bool8/np.bool_/g' "$GYM_PATH/utils/passive_env_checker.py"

echo "==> Setting MUJOCO_GL=egl"
conda env config vars set MUJOCO_GL=egl -n $ENV_NAME

echo ""
echo "Done! Activate with: conda activate $ENV_NAME"
echo "Check CUDA version with: nvidia-smi"
echo "  CUDA 11.x -> bash setup_env.sh 118"
echo "  CUDA 12.x -> bash setup_env.sh 121"
