import tensorflow as tf


class ArcFaceLoss(tf.keras.losses.Loss):
    def __init__(self, loss_func, m=0.5, s=30, name="arcface_loss"):
        """
        Args:
            loss_func: 元々の損失関数
            m: マージン
            s: 倍率
        """
        super().__init__()

        self.loss_func = loss_func
        self.mergin = m
        self.s = s
        self.enable = True

    def call(self, y_true, y_pred):
        # y_predはcos(θ)
        sin = tf.keras.backend.sqrt(1.0 - tf.keras.backend.square(y_pred))
        # cos(θ+m)の加法定理
        phi = y_pred * tf.math.cos(self.mergin) - sin * tf.math.sin(self.mergin)
        phi = tf.where(y_pred > 0, phi, y_pred)

        # 正解クラス: cos(θ+m) 他のクラス: cosθ
        output = (y_true * phi) + ((1.0 - y_true) * y_pred)
        output = self.s * output

        # scaling and shift -> [0, 1]
        output = tf.sigmoid(output)

        return self.loss_func(y_true, output)
