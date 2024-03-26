import tensorflow as tf

import efficientnet.tfkeras as efn


class EfficientNetB0:
    @staticmethod
    def build_model():
        inp = tf.keras.Input(shape=(512, 512, 3))
        base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
        base_model.load_weights(
            "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
        )

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
