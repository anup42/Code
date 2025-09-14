import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from .blocks import ConvBnSiLU, C2f, SPPF, DetectHead


def build_backbone(width_mult=0.50, depth_mult=0.33):
    def ch(x):
        return max(16, int(round(x * width_mult / 8)) * 8)

    def n(x):
        return max(1, int(round(x * depth_mult)))

    inp = L.Input(shape=(None, None, 3))
    # Stem
    x = ConvBnSiLU(ch(64), 3, 2)(inp)
    x = ConvBnSiLU(ch(64), 3, 1)(x)

    # Stage 1
    x = ConvBnSiLU(ch(128), 3, 2)(x)
    x = C2f(ch(128), n(3))(x)
    p3 = x

    # Stage 2
    x = ConvBnSiLU(ch(256), 3, 2)(x)
    x = C2f(ch(256), n(6))(x)
    p4 = x

    # Stage 3
    x = ConvBnSiLU(ch(512), 3, 2)(x)
    x = C2f(ch(512), n(6))(x)
    x = SPPF(ch(512))(x)
    p5 = x

    return keras.Model(inp, [p3, p4, p5], name="backbone")


def build_neck(feat_channels, width_mult=0.50, depth_mult=0.33):
    def ch(x):
        return max(16, int(round(x * width_mult / 8)) * 8)

    def n(x):
        return max(1, int(round(x * depth_mult)))

    p3, p4, p5 = [L.Input(shape=(None, None, c)) for c in feat_channels]

    # Top-down
    p5_td = ConvBnSiLU(ch(256), 1, 1)(p5)
    p5_up = L.UpSampling2D(size=2, interpolation="nearest")(p5_td)
    p4_l = ConvBnSiLU(ch(256), 1, 1)(p4)
    p4_td = C2f(ch(256), n(3))(L.Concatenate(axis=-1)([p4_l, p5_up]))

    p4_up = L.UpSampling2D(size=2, interpolation="nearest")(p4_td)
    p3_l = ConvBnSiLU(ch(128), 1, 1)(p3)
    p3_out = C2f(ch(128), n(3))(L.Concatenate(axis=-1)([p3_l, p4_up]))

    # Bottom-up
    p3_dn = ConvBnSiLU(ch(128), 3, 2)(p3_out)
    p4_out = C2f(ch(256), n(3))(L.Concatenate(axis=-1)([p3_dn, p4_td]))

    p4_dn = ConvBnSiLU(ch(256), 3, 2)(p4_out)
    p5_out = C2f(ch(256), n(3))(L.Concatenate(axis=-1)([p4_dn, p5_td]))

    return keras.Model([p3, p4, p5], [p3_out, p4_out, p5_out], name="neck")


def build_yolo11(num_classes, width_mult=0.50, depth_mult=0.33):
    inp = L.Input(shape=(None, None, 3), name="images")

    backbone = build_backbone(width_mult=width_mult, depth_mult=depth_mult)
    p3, p4, p5 = backbone(inp)

    neck = build_neck([p3.shape[-1], p4.shape[-1], p5.shape[-1]], width_mult=width_mult, depth_mult=depth_mult)
    n3, n4, n5 = neck([p3, p4, p5])

    head = DetectHead(num_classes, ch=(int(n3.shape[-1]), int(n4.shape[-1]), int(n5.shape[-1])))
    out_p3, out_p4, out_p5 = head([n3, n4, n5])

    return keras.Model(inp, [out_p3, out_p4, out_p5], name="yolo11_tf")
