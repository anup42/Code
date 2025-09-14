import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def autopad(k, p=None):
    # Pad to maintain same shape for odd kernels
    if p is None:
        p = k // 2
    return p


class ConvBNAct(layers.Layer):
    def __init__(self, c_out, k=1, s=1, act=True, name=None):
        super().__init__(name=name)
        p = autopad(k)
        self.conv = layers.Conv2D(c_out, k, s, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization(epsilon=1e-5, momentum=0.97)
        self.act = layers.Activation(tf.nn.silu) if act else layers.Activation("linear")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)


class Bottleneck(layers.Layer):
    def __init__(self, c_out, shortcut=True, e=0.5, name=None):
        super().__init__(name=name)
        c_ = int(c_out * e)
        self.cv1 = ConvBNAct(c_, 1, 1)
        self.cv2 = ConvBNAct(c_out, 3, 1)
        self.add_shortcut = shortcut

    def call(self, x, training=False):
        y = self.cv2(self.cv1(x, training=training), training=training)
        if self.add_shortcut and x.shape[-1] == y.shape[-1]:
            y = x + y
        return y


class C2f(layers.Layer):
    """YOLOv8/YOLO11-style C2f with partial concatenation.
    Simplified faithful port for TensorFlow.
    """

    def __init__(self, c_out, n=1, shortcut=True, e=0.5, name=None):
        super().__init__(name=name)
        c_ = int(c_out * e)
        self.cv1 = ConvBNAct(c_out, 1, 1)
        self.m = [Bottleneck(c_out, shortcut, e=e) for _ in range(n)]
        self.cv2 = ConvBNAct(c_out, 1, 1)

    def call(self, x, training=False):
        y = self.cv1(x, training=training)
        outs = [y]
        for b in self.m:
            y = b(y, training=training)
            outs.append(y)
        y = layers.Concatenate(axis=-1)(outs)
        return self.cv2(y, training=training)


class SPPF(layers.Layer):
    def __init__(self, c_out, k=5, name=None):
        super().__init__(name=name)
        self.cv1 = ConvBNAct(c_out // 2, 1, 1)
        self.k = k
        self.cv2 = ConvBNAct(c_out, 1, 1)

    def call(self, x, training=False):
        x = self.cv1(x, training=training)
        y1 = layers.MaxPool2D(pool_size=self.k, strides=1, padding="same")(x)
        y2 = layers.MaxPool2D(pool_size=self.k, strides=1, padding="same")(y1)
        y3 = layers.MaxPool2D(pool_size=self.k, strides=1, padding="same")(y2)
        y = layers.Concatenate(axis=-1)([x, y1, y2, y3])
        return self.cv2(y, training=training)


class UpSample(layers.Layer):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def call(self, x):
        h = tf.shape(x)[1] * self.scale
        w = tf.shape(x)[2] * self.scale
        return tf.image.resize(x, (h, w), method="nearest")

