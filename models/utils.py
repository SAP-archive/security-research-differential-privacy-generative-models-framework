import tensorflow as tf


class Activation:
    ReLU = "relu"
    LeakyReLU = "leaky_relu"
    ELU = "elu"
    TANH = "tanh"
    SIGMOID = "sigmoid"


TF_NON_LAYER_ACTIVATIONS = [Activation.TANH, Activation.SIGMOID]

ACTIVATION_MAP = {
    "relu": tf.keras.layers.ReLU,
    "leaky_relu": tf.keras.layers.LeakyReLU,
    "elu": tf.keras.layers.ELU,
    "tanh": tf.keras.layers.Activation,
    "sigmoid": tf.keras.layers.Activation,
}
