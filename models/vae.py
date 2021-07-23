import tensorflow as tf
from tensorflow.keras import layers

from models.utils import ACTIVATION_MAP, TF_NON_LAYER_ACTIVATIONS


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the data."""

    def __init__(self, epsilon_std=1.0, **kwargs):
        super().__init__(**kwargs)
        self._epsilon_std = epsilon_std

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), stddev=self._epsilon_std)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps input data to a triplet (z_mean, z_log_var, z)."""

    def __init__(
        self,
        latent_dim,
        compress_dims,
        activations,
        activation_kwargs=None,
        batch_norm=False,
        dropout_rate=None,
        epsilon_std=1.0,
        name="encoder",
        **kwargs
    ):
        super(Encoder, self).__init__(name=name, **kwargs)
        if len(compress_dims) != len(activations):
            raise ValueError(
                "number of dimensions must be the same as the number of activations"
            )
        if activation_kwargs is None:
            activation_kwargs = {}
        seq = []
        for dim, act in list(zip(compress_dims, activations)):
            seq.append(layers.Dense(dim, activation="linear"))
            if batch_norm:
                seq.append(layers.BatchNormalization())
            if act in activation_kwargs:
                act_kwargs = activation_kwargs[act]
            else:
                act_kwargs = {}
            if act in TF_NON_LAYER_ACTIVATIONS:
                seq.append(layers.Activation(act))
            else:
                seq.append(ACTIVATION_MAP[act](**act_kwargs))
        if dropout_rate:
            if (
                not isinstance(dropout_rate, (float,))
                or dropout_rate > 1
                or dropout_rate < 0
            ):
                raise ValueError("dropout rate must be a float between 0 and 1")
            else:
                seq.append(layers.Dropout(dropout_rate))
        self.seq = tf.keras.Sequential(seq)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling(epsilon_std=epsilon_std)

    def call(self, inputs, training=None):
        x = self.seq(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded data vector, back into a inout data space."""

    def __init__(
        self,
        original_dim,
        decompress_dims,
        activations,
        activation_kwargs=None,
        batch_norm=False,
        name="decoder",
        **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        if len(decompress_dims) != len(activations):
            raise ValueError(
                "number of dimensions must be the same as the number of activations"
            )
        if activation_kwargs is None:
            activation_kwargs = {}
        seq = []
        for dim, act in list(zip(decompress_dims, activations)):
            seq.append(layers.Dense(dim, activation="linear"))
            if batch_norm:
                seq.append(layers.BatchNormalization())
            if act in activation_kwargs:
                act_kwargs = activation_kwargs[act]
            else:
                act_kwargs = {}
            if act in TF_NON_LAYER_ACTIVATIONS:
                seq.append(layers.Activation(act))
            else:
                seq.append(ACTIVATION_MAP[act](**act_kwargs))
        self.seq = tf.keras.Sequential(seq)
        self.dense_output = layers.Dense(original_dim)

    def call(self, inputs, training=None):
        x = self.seq(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        compress_dims,
        compress_activations,
        decompress_dims,
        decompress_activations,
        latent_dim,
        activation_kwargs=None,
        dropout_rate=None,
        batch_norm=False,
        epsilon_std=1.0,  # previous implementation had user-specified
        name="vae",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.encoder = Encoder(
            latent_dim=latent_dim,
            compress_dims=compress_dims,
            activations=compress_activations,
            activation_kwargs=activation_kwargs,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            epsilon_std=epsilon_std,
        )

        self.decoder = Decoder(
            original_dim,
            decompress_dims=decompress_dims,
            activations=decompress_activations,
            batch_norm=batch_norm,
        )

    def call(self, inputs, training=None, **kwargs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=1
        )
        self.add_loss(kl_loss)
        return reconstructed
