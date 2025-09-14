import sys
print('PY', sys.version)
try:
    import tensorflow as tf
    print('TF', tf.__version__)
except Exception as e:
    print('TF_IMPORT_ERROR', repr(e))
