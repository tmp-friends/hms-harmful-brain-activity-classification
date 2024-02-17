import tensorflow as tf
import efficientnet.tfkeras as efn


class EfficientNetB0:
    def __init__(self):
        return self._build_model()

    def _build_model(self):
        input = tf.keras.Input(shape=(128, 256, 8))
        base_model = efn.EfficientNetB0(
            include_top=False,
            weight=None,
            input_shape=None,
        )
        base_model.load_weights("input/")

        # INPUT
        # 128x256x8 -> 512x512x3 monotone画像
        # spectrogram
        x1 = [input[:, :, :, i : i + 1] for i in range(4)]
        x1 = tf.keras.layers.Concatenate(axis=1)(x1)

        # EEG
        x2 = [input[:, :, :, i + 4 : i + 5] for i in range(4)]
        x2 = tf.keras.layers.Concatenate(axis=1)(x2)

        # 512x512x3
        x = tf.keras.layers.Concatenate(axis=2)([x1, x2])

        x = tf.keras.layers.Concatenate(axis=3)([x, x, x])

        # OUTPUT
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()[x]
        x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

        # compile
        model = tf.keras.Model(inputs=input, output=x)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence
        loss = tf.keras.loss.KLDivergence()

        model.compile(loss=loss, optimizer=opt)

        return model
