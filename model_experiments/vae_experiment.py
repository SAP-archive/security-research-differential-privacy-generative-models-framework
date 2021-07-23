import tensorflow as tf

from general.experiment import (
    Experiment,
    GenerativeModelType,
    TransformerType,
    OptimizerType,
)
from general.utils import OutputActivation, DataTypes, apply_activation
from models.utils import Activation
from models.vae import VariationalAutoEncoder


class VaeExperiment(Experiment):
    """
    VaeExperiment handles the training and sample procedure for a VariationalAutoEncoder.

    See detailed documentation in superclass.
    """

    def __init__(self, user_config):
        super().__init__(user_config)

    def fit_model(self, train_data):

        original_dim = self.pre_fit_transformer.output_dim
        output_info = self.pre_fit_transformer.output_info
        cont_output_act = self.config["model_config"]["output_activations"][
            "continuous"
        ]
        categorical_output_act = self.config["model_config"]["output_activations"][
            "categorical"
        ]

        vae_config = {
            k: v
            for k, v in self.config["model_config"].items()
            if k != "output_activations"
        }
        vae = VariationalAutoEncoder(original_dim, **vae_config)
        optimizer, persistent = self.specify_optimizer_and_tape_persistence()

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data.astype("float32"))
        train_dataset = train_dataset.batch(self.config["model_train"]["batch_size"])

        if self.config["model_train"]["early_stop_epsilon"]["enabled"]:
            self.determine_final_iteration()

        epochs = self.config["model_train"]["epochs"]
        for epoch in range(epochs):
            print("Start of epoch {}".format(epoch))
            # Iterate over the batches of the train data.
            for step, x_batch_train in enumerate(train_dataset):
                with tf.GradientTape(persistent=persistent) as tape:
                    recon_x = vae(x_batch_train)
                    var_list = vae.trainable_variables
                    x_batch_eval = None

                    def loss_fn():
                        if x_batch_eval is not None:
                            x_batch = x_batch_eval
                        else:
                            x_batch = x_batch_train
                        recon_x = vae(x_batch, training=True)
                        st = 0
                        loss = []
                        for item in output_info:
                            ed = st + item[0]
                            if item[1] == DataTypes.CONTINUOUS:
                                feat = apply_activation(recon_x[:, st], cont_output_act)
                                loss.append((x_batch[:, st] - feat) ** 2 / 2)
                            elif item[1] == DataTypes.CATEGORICAL:
                                if categorical_output_act == OutputActivation.SOFTMAX:
                                    loss.append(
                                        tf.nn.softmax_cross_entropy_with_logits(
                                            labels=x_batch[:, st:ed],
                                            logits=recon_x[:, st:ed],
                                        )
                                    )
                                else:
                                    raise ValueError(
                                        "{} is not a handled activation".format(
                                            categorical_output_act
                                        )
                                    )
                            else:
                                raise ValueError(
                                    "{} is not a handled data type".format(item[1])
                                )
                            st = ed

                        if st != recon_x.shape[1]:
                            raise ValueError("transformer output is invalid")

                        loss.append(vae.losses[0])  # Add KLD regularization loss
                        loss = [tf.expand_dims(x, 1) for x in loss]
                        loss = tf.reduce_sum(tf.concat(loss, axis=1), axis=1)
                        return loss

                    if self.config["diff_priv"]["enabled"]:
                        grads_and_vars = optimizer.compute_gradients(
                            loss_fn, var_list, gradient_tape=tape
                        )
                    else:
                        grads = tape.gradient(tf.reduce_mean(loss_fn()), var_list)

                    if (
                        step % 100 == 0
                        and self.config["model_train"]["record_gradients"]["enabled"]
                    ):
                        for group, info in self.gradient_norms.items():
                            if info["support"] > 0:
                                # sample to make x_batch_eval size equal for all groups?
                                x_batch_eval = tf.convert_to_tensor(
                                    train_data[info["idx"]], dtype=tf.float32
                                )
                                self.record_gradient_norms(
                                    group, optimizer, tape, loss_fn, var_list,
                                )
                        # must be reset to None after recording
                        x_batch_eval = None

                if self.config["diff_priv"]["enabled"]:
                    optimizer.apply_gradients(grads_and_vars)
                else:
                    optimizer.apply_gradients(zip(grads, var_list))

                if step % 100 == 0:
                    loss = tf.reduce_mean(loss_fn()).numpy()
                    print("step {}: mean loss = {:0.2f}".format(step, loss))
                    self.losses.append(loss)

                if self.config["diff_priv"]["enabled"] and step == self.steps_per_epoch:
                    self.record_epsilon(epoch)

                if self.config["model_train"]["early_stop_epsilon"][
                    "enabled"
                ] and self.is_final_iteration(epoch, step):
                    break

            if self.do_early_stop_epsilon:
                self.record_epsilon(epoch, step=step)
                break

            if self.config["model_train"]["early_stop"]["enabled"]:
                loss = tf.reduce_mean(loss_fn()).numpy()
                if self.do_early_stop(epoch, loss):
                    break

            if self.is_loss_nan():
                break
        return vae

    def sample_model(self, model):
        noise = tf.random.normal(
            (self.n_training_samples, self.config["model_config"]["latent_dim"])
        )
        fake_data_raw = model.decoder(noise, training=False).numpy()
        output_info = self.pre_fit_transformer.output_info
        cont_output_act = self.config["model_config"]["output_activations"][
            "continuous"
        ]
        categorical_output_act = self.config["model_config"]["output_activations"][
            "categorical"
        ]

        st = 0
        for item in output_info:
            ed = st + item[0]
            if item[1] == DataTypes.CONTINUOUS:
                fake_data_raw[:, st] = apply_activation(
                    fake_data_raw[:, st], cont_output_act
                )
            elif item[1] == DataTypes.CATEGORICAL:
                fake_data_raw[:, st:ed] = apply_activation(
                    fake_data_raw[:, st:ed], categorical_output_act
                )
            else:
                raise ValueError("{} is not a handled data type".format(item[1]))

            st = ed
        if st != fake_data_raw.shape[1]:
            raise ValueError("transformer output is invalid")

        fake_data = self.pre_fit_transformer.inverse_transform(fake_data_raw)
        return fake_data


if __name__ == "__main__":
    config = {
        "dataset": "uci_credit_card.csv",
        "model_type": GenerativeModelType.VAE,
        "name": "vae_experiment",
        "data_path": "../data/",
        "data_processing": {
            "categorical_columns": [
                "SEX",
                "EDUCATION",
                "MARRIAGE",
                "PAY_0",
                "PAY_1",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
                "DEFAULT",
            ],
        },
        "model_config": {
            "compress_dims": [128, 128],
            "decompress_dims": [128, 128],
            "compress_activations": [Activation.LeakyReLU, Activation.LeakyReLU],
            "decompress_activations": [Activation.LeakyReLU, Activation.LeakyReLU],
            "latent_dim": 128,
            "batch_norm": True,
            "output_activations": {
                "continuous": OutputActivation.TANH,
                "categorical": OutputActivation.SOFTMAX,
            },
        },
        "model_train": {
            "seed": 1,
            "test_pct": 0.3,
            "k_fold": False,
            "stratified_by_col": None,
            "epochs": 50,
            "batch_size": 64,
            "dp_optimizer_type": OptimizerType.DPAdamGaussianOptimizer,
            "record_gradients": {"enabled": True, "subset": ["SEX", "EDUCATION"],},
            "transformer": {
                "type": TransformerType.GENERAL,
                "kwargs": {"outlier_clipping": True},
            },
            "early_stop": {"enabled": False},
            "early_stop_epsilon": {"enabled": True, "value": 2.5},
        },
        "diff_priv": {
            "enabled": True,
            "microbatches": 1,
            "l2_norm_clip": 2.0,
            "noise_multiplier": 0.8,
        },
    }
    vae_experiment = VaeExperiment(config)
    vae_experiment.run()
