import tensorflow as tf


class ArcFaceAccuracy(tf.keras.metrics.Mean):
    def __init__(self, metrics_func, s=30, name="arcface_accuracy"):
        super().__init__()

        self.metrics_func = metrics_func
        self.s = s

    def update_state(self, y_true, y_pred, sample_weight=None):
        output = tf.nn.softmax(y_pred * self.s)
        matches = self.metrics_func(y_true, output)

        return super().update_state(matches, sample_weight=sample_weight)
