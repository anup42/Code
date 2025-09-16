#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="yolo11-tf"
PY_VER="3.10"
TF_VER="2.17.1"

usage() {
  echo "Usage: $0 [-n env_name] [-p python_ver] [-t tf_version]" >&2
  echo "Defaults: -n $ENV_NAME -p $PY_VER -t $TF_VER" >&2
}

while getopts ":n:p:t:h" opt; do
  case $opt in
    n) ENV_NAME="$OPTARG" ;;
    p) PY_VER="$OPTARG" ;;
    t) TF_VER="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) usage; exit 1 ;;
  esac
done

echo "[1/5] Creating Conda env '$ENV_NAME' (Python $PY_VER)..."
conda create -n "$ENV_NAME" python="$PY_VER" -y >/dev/null

echo "[2/5] Upgrading pip..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip >/dev/null

echo "[3/5] Installing TensorFlow $TF_VER (bundled CUDA/cuDNN)..."
conda run -n "$ENV_NAME" pip install "tensorflow[and-cuda]==$TF_VER" >/dev/null

echo "[4/5] Installing project dependencies..."
conda run -n "$ENV_NAME" pip install numpy pillow pyyaml >/dev/null

echo "[5/5] Verifying TensorFlow GPU visibility..."
conda run -n "$ENV_NAME" python - <<'PY'
import tensorflow as tf
print('TF:', tf.__version__)
try:
    bi = tf.sysconfig.get_build_info()
    print('Build CUDA:', bi.get('cuda_version'), 'cuDNN:', bi.get('cudnn_version'))
except Exception as e:
    print('Build info unavailable:', e)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY

echo "Done. Activate the env with:"
echo "  conda activate $ENV_NAME"
echo "Optional (stability): disable XLA in this session:"
echo "  export TF_XLA_FLAGS=--tf_xla_auto_jit=0"

