import tensorflow as tf

import efficientnet.tfkeras as efn

from src.losses.arcface_loss import ArcFaceLoss


def build_model():
    input = tf.keras.Input(shape=(128, 256, 8))
    base_model = efn.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=None,
    )
    base_model.load_weights(
        "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
    )

    # base_model = unfreeze_model(base_model)

    # INPUT
    # 128x256x8 -> 512x512x3 monotone画像
    # spectrogram
    x1 = [input[:, :, :, i : i + 1] for i in range(4)]
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)

    # EEG
    x2 = [input[:, :, :, i + 4 : i + 5] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)

    # Make512x512x3
    x = tf.keras.layers.Concatenate(axis=2)([x1, x2])

    x = tf.keras.layers.Concatenate(axis=3)([x, x, x])

    # OUTPUT
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Dropout(0.25)(x)

    # x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)
    arcface_layer = ArcFaceLayer0(num_outputs=6, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    x = arcface_layer(x)

    # compile
    model = tf.keras.Model(inputs=input, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence
    # loss = tf.keras.losses.KLDivergence()
    # loss = ArcFaceLoss(loss_func=tf.keras.losses.KLDivergence())
    loss = ArcFaceLoss(loss_func=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    model.compile(loss=loss, optimizer=opt)

    return model


def unfreeze_model(model):
    """
    Unfreeze layers while leaving BatchNorm layers frozen

    PetFinder 6th place solution
    https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/301015
    """
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    return model


class ArcFaceLayer0(tf.keras.layers.Layer):
    def __init__(self, num_outputs, kernel_regularizer=None):
        super().__init__()

        self.num_outputs = num_outputs
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        weight_shape = (input_shape[-1], self.num_outputs)
        self.kernel = self.add_weight(
            name="kernel",
            shape=weight_shape,
            initializer=tf.keras.initializers.TruncatedNormal(),
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, input):
        n_inputs = tf.math.l2_normalize(input, axis=1)
        n_kernels = tf.math.l2_normalize(self.kernel, axis=0)

        return tf.matmul(n_inputs, n_kernels)  # W.Txの内積
