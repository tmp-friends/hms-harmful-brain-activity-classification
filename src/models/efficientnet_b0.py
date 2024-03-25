import tensorflow as tf
from tensorflow.keras.models import load_model

# import efficientnet.tfkeras as efn


class EfficientNetB0:
    @staticmethod
    def build_model():
        inp = tf.keras.Input(shape=(512, 512, 3))
        base_model = load_model("/kaggle/input/efficientnetb-tf-keras/EfficientNetB0.h5")

        # OUTPUT
        x = base_model(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

        # compile
        model = tf.keras.Model(inputs=inp, outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence
        loss = tf.keras.losses.KLDivergence()

        model.compile(loss=loss, optimizer=opt)

        return model
