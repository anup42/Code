import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from .blocks import ConvBnSiLU, C2f, SPPF
from tensorflow.keras import layers as L


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

    # Bottom-up path produces strides {8, 16}
    p3_dn = ConvBnSiLU(ch(128), 3, 2)(p3_out)
    p4_out = C2f(ch(256), n(3))(L.Concatenate(axis=-1)([p3_dn, p4_td]))

    p4_dn = ConvBnSiLU(ch(256), 3, 2)(p4_out)
    p5_out = C2f(ch(256), n(3))(L.Concatenate(axis=-1)([p4_dn, p5_td]))

    # Return strides (8, 16, 32) for the detection head
    return keras.Model([p3, p4, p5], [p3_out, p4_out, p5_out], name="neck")


class DetectHeadDFL(keras.Model):
    """YOLO-style decoupled head with DFL regression (anchor-free).

    Outputs dict with:
      - 'cls': list of [B, HW, C]
      - 'reg': list of [B, HW, 4*(bins)]
      - 'grids': list of [HW, 2] (pixel centers)
      - 'strides': list of ints in image pixels (e.g. 8, 16, 32)
    """

    def __init__(self, num_classes, ch=(256, 256, 256), reg_max=16, strides=(8, 16, 32), name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.bins = reg_max + 1
        self.strides = strides
        stems, cls_convs, reg_convs, obj_convs = [], [], [], []
        cls_preds, reg_preds, obj_preds = [], [], []
        for i, c in enumerate(ch):
            stem = ConvBnSiLU(c, 1, 1, name=f"stem_{i}")
            cls_conv = keras.Sequential([ConvBnSiLU(c, 3, 1), ConvBnSiLU(c, 3, 1)], name=f"cls_conv_{i}")
            reg_conv = keras.Sequential([ConvBnSiLU(c, 3, 1), ConvBnSiLU(c, 3, 1)], name=f"reg_conv_{i}")
            cls_pred = L.Conv2D(num_classes, 1, 1, padding="same", name=f"cls_pred_{i}")
            reg_pred = L.Conv2D(4 * self.bins, 1, 1, padding="same", name=f"reg_pred_{i}")
            obj_conv = keras.Sequential([ConvBnSiLU(c, 3, 1)], name=f"obj_conv_{i}")
            obj_pred = L.Conv2D(1, 1, 1, padding="same", name=f"obj_pred_{i}")
            stems.append(stem)
            cls_convs.append(cls_conv)
            reg_convs.append(reg_conv)
            obj_convs.append(obj_conv)
            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            obj_preds.append(obj_pred)
        self.stems = stems
        self.cls_convs = cls_convs
        self.reg_convs = reg_convs
        self.obj_convs = obj_convs
        self.cls_preds = cls_preds
        self.reg_preds = reg_preds
        self.obj_preds = obj_preds

    def call(self, feats, training=False):
        cls_list = []
        reg_list = []
        obj_list = []
        grids = []
        # Build per-scale outputs
        for i, f in enumerate(feats):
            x = self.stems[i](f, training=training)
            cls_feat = self.cls_convs[i](x, training=training)
            reg_feat = self.reg_convs[i](x, training=training)
            cls_out = self.cls_preds[i](cls_feat)  # [B,H,W,C]
            reg_out = self.reg_preds[i](reg_feat)  # [B,H,W,4*bins]
            obj_feat = self.obj_convs[i](x, training=training)
            obj_out = self.obj_preds[i](obj_feat)  # [B,H,W,1]

            B = tf.shape(cls_out)[0]
            H = tf.shape(cls_out)[1]
            W = tf.shape(cls_out)[2]

            # Flatten to [B,HW,*]
            cls_list.append(tf.reshape(cls_out, [B, H * W, -1]))
            reg_list.append(tf.reshape(reg_out, [B, H * W, -1]))
            obj_list.append(tf.reshape(obj_out, [B, H * W, -1]))

            # Grid centers in pixels
            stride = float(self.strides[i])
            yy = tf.range(H, dtype=tf.float32)
            xx = tf.range(W, dtype=tf.float32)
            yy, xx = tf.meshgrid(yy, xx, indexing="ij")
            gy = (yy + 0.5) * stride
            gx = (xx + 0.5) * stride
            grid = tf.stack([gx, gy], axis=-1)  # [H,W,2]
            grids.append(tf.reshape(grid, [H * W, 2]))

        return {"cls": cls_list, "reg": reg_list, "obj": obj_list, "grids": grids, "strides": list(self.strides)}


def build_yolo11(num_classes, width_mult=0.50, depth_mult=0.33, reg_max=16):
    inp = L.Input(shape=(None, None, 3), name="images")

    backbone = build_backbone(width_mult=width_mult, depth_mult=depth_mult)
    p3, p4, p5 = backbone(inp)

    neck = build_neck([p3.shape[-1], p4.shape[-1], p5.shape[-1]], width_mult=width_mult, depth_mult=depth_mult)
    n3, n4, n5 = neck([p3, p4, p5])

    head = DetectHeadDFL(
        num_classes,
        ch=(int(n3.shape[-1]), int(n4.shape[-1]), int(n5.shape[-1])),
        reg_max=reg_max,
        strides=(8, 16, 32),
    )

    # Keras functional model can return dict; use subclass wrapper
    class Wrapper(keras.Model):
        def __init__(self, backbone, neck, head):
            super().__init__(name="yolo11_tf")
            self.backbone = backbone
            self.neck = neck
            self.head = head

        def call(self, x, training=False):
            p3, p4, p5 = self.backbone(x, training=training)
            n3, n4, n5 = self.neck([p3, p4, p5], training=training)
            return self.head([n3, n4, n5], training=training)

    return Wrapper(backbone, neck, head)

