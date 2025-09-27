import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L


def ConvBnSiLU(filters, k=1, s=1, p="same", name=None):
    return keras.Sequential([
        L.Conv2D(filters, k, s, padding=p, use_bias=False, kernel_initializer="he_normal"),
        L.BatchNormalization(momentum=0.97, epsilon=1e-3),
        L.Activation("swish"),  # SiLU
    ], name=name)


class Bottleneck(keras.Model):
    def __init__(self, ch, shortcut=True, e=0.5, name=None):
        super().__init__(name=name)
        hidden = int(ch * e)
        self.cv1 = ConvBnSiLU(hidden, 1, 1)
        self.cv2 = ConvBnSiLU(ch, 3, 1)
        self.add = shortcut

    def call(self, x, training=False):
        y = self.cv2(self.cv1(x, training=training), training=training)
        return x + y if self.add else y


class C2f(keras.Model):
    """C2f block similar to Ultralytics C2f (YOLOv8/YOLO11-style).

    Splits channels, processes multiple Bottlenecks, and concatenates.
    """

    def __init__(self, ch, n=2, e=0.5, name=None):
        super().__init__(name=name)
        hidden = int(ch * e)
        self.cv1 = ConvBnSiLU(hidden, 1, 1)
        self.cv2 = ConvBnSiLU(ch, 1, 1)
        # Each bottleneck operates on half-channel chunks
        self.m = [Bottleneck(hidden // 2, shortcut=True, e=1.0) for _ in range(n)]

    def call(self, x, training=False):
        y = self.cv1(x, training=training)
        chunks = tf.split(y, num_or_size_splits=2, axis=-1)
        y_list = [chunks[0], chunks[1]]
        for b in self.m:
            y_list.append(b(y_list[-1], training=training))
        y = tf.concat(y_list, axis=-1)
        return self.cv2(y, training=training)


class SPPF(keras.Model):
    def __init__(self, ch, k=5, name=None):
        super().__init__(name=name)
        hidden = ch // 2
        self.cv1 = ConvBnSiLU(hidden, 1, 1)
        self.cv2 = ConvBnSiLU(ch, 1, 1)
        self.k = k

    def call(self, x, training=False):
        x = self.cv1(x, training=training)
        y1 = L.MaxPool2D(pool_size=self.k, strides=1, padding="same")(x)
        y2 = L.MaxPool2D(pool_size=self.k, strides=1, padding="same")(y1)
        y3 = L.MaxPool2D(pool_size=self.k, strides=1, padding="same")(y2)
        x = tf.concat([x, y1, y2, y3], axis=-1)
        return self.cv2(x, training=training)


class DetectHead(keras.Model):
    """Decoupled detection head for 3 scales (P3, P4, P5).

    Outputs per-scale tensors with shape [B, H, W, 5 + num_classes].
    """

    def __init__(self, num_classes, ch=(256, 256, 256), name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        prior_prob = 0.01
        prior_logit = -math.log((1.0 - prior_prob) / prior_prob)
        cls_bias_init = tf.keras.initializers.Constant(prior_logit)
        obj_bias_vector = np.concatenate([np.zeros(4, dtype=np.float32), np.full((1,), prior_logit, dtype=np.float32)])
        stems, cls_convs, reg_convs, cls_preds, box_preds = [], [], [], [], []
        for i, c in enumerate(ch):
            stem = ConvBnSiLU(c, 1, 1, name=f"stem_{i}")
            cls_conv = keras.Sequential([ConvBnSiLU(c, 3, 1), ConvBnSiLU(c, 3, 1)], name=f"cls_conv_{i}")
            reg_conv = keras.Sequential([ConvBnSiLU(c, 3, 1), ConvBnSiLU(c, 3, 1)], name=f"reg_conv_{i}")
            cls_pred = L.Conv2D(num_classes, 1, 1, padding="same", bias_initializer=cls_bias_init, name=f"cls_pred_{i}")
            box_pred = L.Conv2D(4 + 1, 1, 1, padding="same", bias_initializer=tf.keras.initializers.Constant(obj_bias_vector), name=f"box_pred_{i}")  # 4 box + 1 obj
            stems.append(stem)
            cls_convs.append(cls_conv)
            reg_convs.append(reg_conv)
            cls_preds.append(cls_pred)
            box_preds.append(box_pred)
        self.stems = stems
        self.cls_convs = cls_convs
        self.reg_convs = reg_convs
        self.cls_preds = cls_preds
        self.box_preds = box_preds

    def call(self, feats, training=False):
        outputs = []
        for i, f in enumerate(feats):
            x = self.stems[i](f, training=training)
            cls_feat = self.cls_convs[i](x, training=training)
            reg_feat = self.reg_convs[i](x, training=training)
            cls_out = self.cls_preds[i](cls_feat)
            box_out = self.box_preds[i](reg_feat)
            out = tf.concat([box_out, cls_out], axis=-1)
            outputs.append(out)
        return outputs


