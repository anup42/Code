param(
  [string]$EnvName = "yolo11-tf",
  [string]$Python = "3.10",
  [string]$TFVersion = "2.17.1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "[1/5] Creating Conda env '$EnvName' (Python $Python)..." -ForegroundColor Cyan
conda create -n $EnvName python=$Python -y | Out-Null

Write-Host "[2/5] Upgrading pip..." -ForegroundColor Cyan
conda run -n $EnvName python -m pip install --upgrade pip | Out-Null

Write-Host "[3/5] Installing TensorFlow $TFVersion (bundled CUDA/cuDNN)..." -ForegroundColor Cyan
conda run -n $EnvName pip install "tensorflow[and-cuda]==$TFVersion" | Out-Null

Write-Host "[4/5] Installing project dependencies..." -ForegroundColor Cyan
conda run -n $EnvName pip install numpy pillow pyyaml | Out-Null

Write-Host "[5/5] Verifying TensorFlow GPU visibility..." -ForegroundColor Cyan
$tmp = New-TemporaryFile
@"
import tensorflow as tf
print('TF:', tf.__version__)
try:
    bi = tf.sysconfig.get_build_info()
    print('Build CUDA:', bi.get('cuda_version'), 'cuDNN:', bi.get('cudnn_version'))
except Exception as e:
    print('Build info unavailable:', e)
print('GPUs:', tf.config.list_physical_devices('GPU'))
"@ | Set-Content -Path $tmp -Encoding UTF8
conda run -n $EnvName python "$tmp"
Remove-Item $tmp -Force

Write-Host "Done. Activate the env with: `n  conda activate $EnvName" -ForegroundColor Green
Write-Host "Optional (stability): disable XLA in this session: `n  $env:TF_XLA_FLAGS=\"--tf_xla_auto_jit=0\"" -ForegroundColor DarkGray

