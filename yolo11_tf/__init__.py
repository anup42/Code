"""Package init for YOLO11-TF utilities.

Sets safe defaults to avoid known TensorFlow GPU crashes with
UnsortedSegment kernels when XLA JIT is enabled via environment.
"""

import os as _os

# Disable global XLA JIT unless user explicitly opted in before import.
_os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

