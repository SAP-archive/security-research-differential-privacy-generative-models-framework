import pandas as pd
import numpy as np
import tensorflow as tf
import json
import hashlib
import os
import pickle
import logging
from sys import exit
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow_privacy import (
    DPAdamGaussianOptimizer,
    DPGradientDescentGaussianOptimizer,
    DPAdagradGaussianOptimizer,
)
from tensorflow_privacy.privacy.analysis.rdp_accountant import (
    compute_rdp,
    get_privacy_spent,
)

from general.utils import (
    load_df,
    label_encode_categorical_columns,
    shuffle_data,
    train_test_indices,
    subset_indices,
)
from models.vae import VariationalAutoEncoder
from transformers.bgm_transformer import BGMTransformer
from transformers.general_transformer import GeneralTransformer


class GenerativeModelType:
    VAE = 0


GENERATIVE_MODEL_MAP = {0: VariationalAutoEncoder}


class TransformerType:
    GENERAL = 0
    BGM = 1


TRANSFORMER_MAP = {0: GeneralTransformer, 1: BGMTransformer}


class OptimizerType:
    # tf-privacy optimizers
    DPGradientDescentGaussianOptimizer = 0
    DPAdagradGaussianOptimizer = 1
    DPAdamGaussianOptimizer = 2

    # tf optimizers
    GradientDescentOptimizer = 3
    AdagradOptimizer = 4
    AdamOptimizer = 5


OPTIMIZER_MAP = {
    0: DPGradientDescentGaussianOptimizer,
    1: DPAdagradGaussianOptimizer,
    2: DPAdamGaussianOptimizer,
    3: SGD,
    4: Adagrad,
    5: Adam,
}


class Experiment:
    """
    Experiment handles the full process of data generation across different model
    configurations.

    The class shouldn't be instantiated directly. Rather, it serves as the superclass
    for a ModelExperiment (i.e. VaeExperiment). In order to run an experiment,
    the ModelExperiment must implement the fit_model() and sample_model() methods.

    Parameters
    ----------
    user_config : dict
        dataset : str
            File name of the dataset to use for the experiment.

        model_type : GenerativeModelType
            Type of model to use for the experiment.

        name : str (default="model_experiment")
            Name of the experiment (e.g. vae_experiment).

        data_path : str (default="../data/")
            Path for where to access the data and store the experiment output.

        model_config : dict
            Contains parameters to be passed to the model class.

        data_processing : dict
            Contains parameters for pre-processing data transformations.

            read_csv_kwargs : dict (default={})
                Keyword arguments to pass to pd.read_csv().

            columns_to_drop : list (default=[])
                Whether to drop certain columns when loading the data.

            columns_rename_map : dict (default={})
                A map between old_column_name and new_column_name.

            categorical_columns : list (default=[])
                The categorical columns names of the data. If passed to the config,
                it assumes that all categorical columns are included. If not passed
                to the config, then the experiment will consider columns to be
                categorical if the data type is neither an int nor float.

        model_train : dict
            Contains parameters that govern the model training procedure.

            seed : int (default=None)
                The seed to set for experiments. Both the numpy and tensorflow seed
                will be set.

            test_pct : float (default=0.3)
                Percentage of data to use for the test set.

            k_fold : bool (default=False)
                Whether to run k-fold cross validation (i.e. k model_experiments will be run).
                If True, then the number of folds is determined by int(1 / test_pct).

            stratified_by_col : str (default=None)
                The column to use for stratification in KFoldStratification. The default
                is to use no stratification.

            batch_size : int (default=64)
                Size of batches to use in the gradient descent optimization.

            epochs : int (default=50)
                Maximum number of epochs to train the model. Can be cut short if early
                stopping is enabled.

            learning_rate : float (default=0.001)
                Learning rate to use in the gradient descent optimization.

            optimizer_type : OptimizerType (default=OptimizerType.AdamOptimizer)
                Optimizer type if training without differential privacy (e.g. with_dp=False)

            dp_optimizer_type : OptimizerType (default=OptimizerType.DPAdamGaussianOptimizer)
                Optimizer type if training with differential privacy (e.g. with_dp=True)

            transformer : dict
                The Transformer class converts data that is already pre-processed into a
                format suitable for model training. It also does the inverse
                transformation to convert generated data back to the original format.

                type : TransformerType (default=TransformerType.GENERAL)
                    Type of transformer to use in the experiment.

                kwargs : dict
                    Any keyword arguments that should be passed to the Transformer class.

            early_stop : dict
                Contains parameters to govern the early stopping procedure.

                enabled : bool (default=True)
                    Whether to use early stopping.

                first_epoch : int (default=4)
                    First epoch to begin checking whether to stop the training process.

                n_previous_losses : int (default=10)
                    Number of previously saved losses to evaluate whether to early stop.

                loss_pct : float (default=0.8)
                    Percentage of losses in the previous n that should be lower than the
                    current loss in order to stop the training process.

            early_stop_epsilon : dict
                Contains parameters to govern early stopping when a maximum epsilon value
                is reached. Should only be used when training with differential privacy.

                enabled : bool (default=False)
                    Whether to use early stopping by epsilon.

                value : float (default=5.0)
                    The maximum value of epsilon to allow.

            record_gradients : dict
                Contains parameters that govern how the gradients should be
                monitored during training.

                enabled : bool (default=True)

                subset : list (default=[])
                    The default is to record the gradient norms of the entire training
                    data. Passing in column names will record gradient norms for each
                    of the resulting combinations of the columns.

                clip_gradient_per_sample : bool (default=False)
                    When training with dp-SGD, the num_microbatches parameter controls
                    the grouping size to use during the gradient clipping procedure.
                    As the gradients are recorded per subgroup, often the originally
                    specified num_microbatches will not divide evenly with the number
                    of examples per subgroup. When True, it is equivalent to
                    num_microbatches=number of examples per subgroup. When False,
                    it is equivalent to num_micbrobatches=1. Setting to True will
                    significantly increase the time needed to record the norms, while
                    providing a more accurate measurement.

        diff_priv : dict
            Contains parameters to govern the differential privacy aspect of training.
            Descriptions taken from:
            https://github.com/tensorflow/privacy/tree/master/tutorials

            enabled : bool (default=True)
                Whether to train with differential privacy.

            microbatches : int (default=1)
                The input data for each step (i.e., batch) of your original training
                algorithm is split into this many microbatches. Generally, increasing
                this will improve your utility but slow down your training in terms of
                wall-clock time. The total number of examples consumed in one global
                step remains the same. This number should evenly divide your input
                batch size.

            l2_norm_clip : float (default=1.0)
                The cumulative gradient across all network parameters from each
                microbatch will be clipped so that its L2 norm is at most this value.
                You should set this to something close to some percentile of what you
                expect the gradient from each microbatch to be.

            noise_multiplier : float (default=1.1)
                This governs the amount of noise added during training. Generally,
                more noise results in better privacy and lower utility. This generally
                has to be at least 0.3 to obtain rigorous privacy guarantees, but smaller
                values may still be acceptable for practical purposes.

    Examples
    --------
    config = {
        'dataset': 'adult.csv',
        'model_type': GenerativeModelType.VAE,
        'model_config': {
            'compress_dims': [128, 128],
            'decompress_dims': [128, 128],
            'compress_activations': ['elu', 'elu'],
            'decompress_activations': ['elu', 'elu'],
            'latent_dim': 128,
            "output_activations": {
                "continuous": OutputActivation.TANH,
                "categorical": OutputActivation.SOFTMAX,
            }
        }
    }
    vae_experiment = VaeExperiment(config)
    vae_experiment.run()

    """

    def __init__(self, user_config):
        self.config = user_config
        self._default_config = {
            "dataset": None,
            "model_type": None,
            "name": "model_experiment",
            "data_path": "../data/",  # default is when run from /model_experiments
            "model_config": None,
            "data_processing": {
                "read_csv_kwargs": {},
                "columns_to_drop": [],
                "columns_rename_map": {},
                "categorical_columns": [],
            },
            "model_train": {
                "seed": None,
                "test_pct": 0.3,
                "k_fold": False,
                "stratified_by_col": None,
                "batch_size": 64,
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer_type": OptimizerType.AdamOptimizer,
                "dp_optimizer_type": OptimizerType.DPAdamGaussianOptimizer,
                "transformer": {"type": TransformerType.BGM, "kwargs": {}},
                "early_stop": {
                    "enabled": True,
                    "first_epoch": 4,
                    "n_previous_losses": 10,
                    "loss_pct": 0.8,
                },
                "early_stop_epsilon": {"enabled": False, "value": 5.0},
                "record_gradients": {
                    "enabled": True,
                    "subset": [],
                    "clip_gradient_per_sample": False,
                },
            },
            "diff_priv": {
                "enabled": True,
                "microbatches": 1,
                "l2_norm_clip": 1.0,
                "noise_multiplier": 1.1,
            },
        }
        self._required_parameters = ["dataset", "model_type", "model_config"]
        self.build_config()
        self.set_seed()
        self._data_path = os.path.join(os.getcwd(), self.config["data_path"])
        self.check_microbatch_size_compatibility()
        self._config_json = json.dumps(self.config, sort_keys=True)
        self.config_file = "{}.pkl".format(
            hashlib.sha1(self._config_json.encode()).hexdigest()
        )
        self.config_hash_path = os.path.join(self._data_path, self.config_file)

        self.column_names = None
        self.categorical_indices = None
        self.num_categories_per_col = None
        self.shuffle_idx = None

        self.pre_fit_transformer = None
        self.n_training_samples = None
        self.steps_per_epoch = None
        self.losses = []
        self.epsilons = []
        self.gradient_norms = {}
        self.final_iteration = None
        self.do_early_stop_epsilon = False

    def build_config(self):
        """
        First, ensures that the user specified config:
            1. Is a dictionary.
            2. Includes the required parameters.
            3. Has a model_type that is in GenerativeModelType.
            4. Has a model_config that is of type dict.

        Then, add any item to the user config that is in the default config
        but is not present in the user config. It assumes that none of the default
        config values are three layer or more embedded dicts containing items.
        """
        if not isinstance(self.config, dict):
            raise ValueError("the config must be a dictionary")

        missing_keys = [k for k in self._required_parameters if k not in self.config]
        if len(missing_keys) > 0:
            raise ValueError(
                "the following required keys are missing: {}".format(missing_keys)
            )

        if self.config["model_type"] not in GENERATIVE_MODEL_MAP:
            raise ValueError("new model must be added to mapping")

        if not isinstance(self.config["model_config"], dict):
            raise ValueError(
                "the model config must be a dictionary, passed as: {}".format(
                    type(self.config["model_config"])
                )
            )

        for k, v in self._default_config.items():
            if k not in self.config.keys():
                self.config[k] = v
            elif isinstance(self._default_config[k], dict):
                for k2, v2 in self._default_config[k].items():
                    if k2 not in self.config[k].keys():
                        self.config[k][k2] = v2
                    elif isinstance(self._default_config[k][k2], dict):
                        for k3, v3 in self._default_config[k][k2].items():
                            if k3 not in self.config[k][k2].keys():
                                self.config[k][k2][k3] = v3

    def set_seed(self):
        if self.config["model_train"]["seed"] is not None:
            np.random.seed(self.config["model_train"]["seed"])
            tf.random.set_seed(self.config["model_train"]["seed"])

    def check_microbatch_size_compatibility(self):
        """
        If training with differential privacy, then the the following conditions about
        the batch_size and microbatch size must be must:
            1. The last batch is larger than the microbatch size.
            2. The last batch is divisible by the microbatch size.
            3. The batch size is divisible by the microbatch size.

        If the specified microbatch size does not meet the above criteria, then
        the method will progressively check whether values one less are satisfactory.
        If none of the values are suitable, then a a microbatch size of 1 is used.
        """
        if (
            self.config["diff_priv"]["enabled"]
            and self.config["diff_priv"]["microbatches"] is not None
        ):
            batch_size = self.config["model_train"]["batch_size"]
            num_microbatches = self.config["diff_priv"]["microbatches"]

            n = len(pd.read_csv(os.path.join(self._data_path, self.config["dataset"])))
            train_indices, _ = train_test_indices(
                np.arange(n),
                self.config["model_train"]["test_pct"],
                self.config["model_train"]["k_fold"],
            )
            n_trains = [len(train_idx) for train_idx in train_indices]

            if all(np.array(n_trains) == n_trains[0]):
                n_train = n_trains[0]
            else:
                num_microbatches = 1

            while num_microbatches > 1:
                valid_cond = all(
                    [
                        n_train % batch_size >= num_microbatches,
                        batch_size % num_microbatches == 0,
                        (n_train % batch_size) % num_microbatches == 0,
                    ]
                )
                if valid_cond:
                    break
                num_microbatches -= 1

            if num_microbatches < 1:
                raise ValueError(
                    "error in microbatch re-sizing procedure, size cannot be less than 1"
                )

            if num_microbatches != self.config["diff_priv"]["microbatches"]:
                msg = " Resetting number of microbatches from {} to {}.".format(
                    self.config["diff_priv"]["microbatches"], num_microbatches
                )
                logging.warning(msg)
                self.config["diff_priv"]["microbatches"] = num_microbatches

    def preprocess_data(self, path):
        """
        Progressively calls the necessary basic pre-processing steps and stores
        relevant information.

        Steps:
            1. Load the original data.
            2. Label encode the categorical features.
            3. Shuffle the dataset.

        Parameters
        ----------
        path : str
            The path for the dataset file.
        """
        data = load_df(
            path,
            False,
            self.config["data_processing"]["columns_to_drop"],
            self.config["data_processing"]["columns_rename_map"],
            self.config["data_processing"]["read_csv_kwargs"],
        )
        self.column_names = list(data.columns)

        res = label_encode_categorical_columns(
            data,
            True,
            categorical_columns=self.config["data_processing"]["categorical_columns"],
        )
        data = res[0]
        self.categorical_indices = res[1]
        self.num_categories_per_col = res[2]

        res = shuffle_data(data, return_index=True)
        data = res[0]
        self.shuffle_idx = res[1]
        return data

    def transform_data(self, data):
        """
        Instantiates the transformer class with the correct parameters, transforms
        the data, and stores the pre-fit-transformer.

        Parameters
        ----------
        data : np.array
            The data after it has gone through preprocess_data()
        """
        transformer_cls = TRANSFORMER_MAP[
            self.config["model_train"]["transformer"]["type"]
        ]
        transformer = transformer_cls(
            **self.config["model_train"]["transformer"]["kwargs"]
        )
        transformer.fit(data, self.categorical_indices, self.num_categories_per_col)
        data = transformer.transform(data)

        self.pre_fit_transformer = transformer
        return data

    def run(self, automatic_overwrite=False):
        """
        The core method of the Experiment class. It runs the entire procedure:
            1. Verify if a config run already exists and whether to overwrite.
            2. Preprocess the data.
            3. Determine the train/test indices for each model run (>1 if k-fold).
            4. Transform the data to be suitable for model training.
            5. Calls the fit_model() method (implemented in subclass).
            6. Calls the sample_model() method (implemented in sublcass).
            7. Verify integrity of output and save.

        Parameters
        ----------
        automatic_overwrite : bool (default=False)
            If True, the program will automatically overwrite the saved files when
            they exist.
        """
        if not automatic_overwrite and os.path.isfile(self.config_hash_path):
            overwrite = input("Config run already exists. Overwrite? y = yes, n = no\n")
            if overwrite.lower() == "n":
                exit()
            elif overwrite.lower() == "y":
                pass
            else:
                raise ValueError("must be either y or n")

        dataset_path = os.path.join(self._data_path, self.config["dataset"])
        data = self.preprocess_data(dataset_path)

        out = {
            "dataset": self.config["dataset"],
            "shuffle_idx": self.shuffle_idx,
            "column_names": self.column_names,
            "categorical_indices": self.categorical_indices,
            "num_categories_per_col": self.num_categories_per_col,
            "model_runs": [],
        }

        if self.config["model_train"]["stratified_by_col"] is not None:
            stratified_col_idx = self.column_names.index(
                self.config["model_train"]["stratified_by_col"]
            )
        else:
            stratified_col_idx = None
        train_indices, test_indices = train_test_indices(
            data,
            self.config["model_train"]["test_pct"],
            self.config["model_train"]["k_fold"],
            stratified_col_idx,
        )
        if self.config["model_train"]["k_fold"]:
            print("Training model with {} fold validation".format(len(train_indices)))
        else:
            print("Training single model")

        for i, (train_idx, test_idx) in enumerate(
            list(zip(train_indices, test_indices))
        ):
            print("Model number: {}".format(i))
            train_data = data[train_idx]

            if self.config["model_train"]["record_gradients"]["enabled"]:
                self.record_subset_indices(train_data)

            transformed_train_data = self.transform_data(train_data)
            self.n_training_samples = transformed_train_data.shape[0]
            self.steps_per_epoch = (
                self.n_training_samples // self.config["model_train"]["batch_size"]
            )

            model = self.fit_model(transformed_train_data)
            fake_data = self.sample_model(model)
            self.generated_data_sanity_check(fake_data)

            out["model_runs"].append(
                {
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "fake_data": fake_data,
                    "epsilons": self.epsilons,
                    "losses": self.losses,
                    "gradient_norms": self.gradient_norms,
                }
            )

            # these values must be reset at the start of each training fold
            self.losses = []
            self.epsilons = []
            self.gradient_norms = {}
            if self.config["model_train"]["early_stop_epsilon"]["enabled"]:
                self.do_early_stop_epsilon = False

        self.save_results(out)

    def save_results(self, out):
        out["config"] = self._config_json
        path = self.config_hash_path

        with open(path, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("experiment saved to {}".format(self.config_file))

    def generated_data_sanity_check(self, data):
        """
        Verifies the integrity of the generated data with the following checks:
            1. Do nans exist?
            2. Do any continuous columns have values of zero greater than
               the zero_value_threshold?
        """
        if np.any(np.isnan(data)):
            out_dict = dict(
                zip(self.column_names, np.sum(np.isnan(data), 0) / data.shape[0])
            )
            msg = (
                "Generated data fails sanity check because nans exist. Percent "
                "nans: {}".format({k: v for k, v in out_dict.items() if v > 0})
            )
            logging.warning(msg)

        mask = np.ones(data.shape[1], dtype=bool)
        mask[self.categorical_indices] = False
        num_zero_values = np.sum(data[:, mask] == 0, 0)
        if np.any(num_zero_values != 0):
            out_dict = dict(
                zip(np.array(self.column_names)[mask], num_zero_values / data.shape[0])
            )
            msg = "Continuous cols with 0 values. Percent equal to 0: {}".format(
                {k: v for k, v in out_dict.items() if v != 0}
            )
            logging.warning(msg)

    def specify_optimizer_and_tape_persistence(self):
        """
        If training with differential privacy then the to utilize the tf-privacy features
        the GradientTape must be persistent. The optimizer will also be a DP one.
        """
        if self.config["diff_priv"]["enabled"]:
            optimizer = OPTIMIZER_MAP[self.config["model_train"]["dp_optimizer_type"]](
                l2_norm_clip=self.config["diff_priv"]["l2_norm_clip"],
                noise_multiplier=self.config["diff_priv"]["noise_multiplier"],
                num_microbatches=self.config["diff_priv"]["microbatches"],
                learning_rate=self.config["model_train"]["learning_rate"],
            )
            persistent = True
        else:
            optimizer = OPTIMIZER_MAP[self.config["model_train"]["optimizer_type"]](
                learning_rate=self.config["model_train"]["learning_rate"]
            )
            if self.config["model_train"]["record_gradients"]["enabled"]:
                persistent = True
            else:
                persistent = False
        return optimizer, persistent

    def compute_epsilon(self, steps, sampling_probability):
        """
        Computes epsilon value for given hyperparameters.
        """
        if self.config["diff_priv"]["noise_multiplier"] == 0.0:
            return float("inf")
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(
            q=sampling_probability,
            noise_multiplier=self.config["diff_priv"]["noise_multiplier"],
            steps=steps,
            orders=orders,
        )
        # should always be at least the inverse of data size
        delta = 1 / self.n_training_samples
        return get_privacy_spent(orders, rdp, target_delta=delta)[0]

    def record_epsilon(self, epoch, step=None, do_print=True):
        """
        Saves the step of having to call compute_epsilon() and then record it
        yourself in the training procedure. It saves the epsilon value to
        the recorded list.

        Parameters
        ----------
        epoch : int
            The current epoch in training procedure.

        do_print : bool (default=True)
            Whether to print epsilon.
        """
        if step is not None:
            steps = epoch * self.steps_per_epoch + step
        else:
            steps = (epoch + 1) * self.steps_per_epoch
        sampling_probability = (
            self.config["model_train"]["batch_size"] / self.n_training_samples
        )
        eps = self.compute_epsilon(
            steps=steps, sampling_probability=sampling_probability
        )
        if do_print:
            print("epsilon = {:0.2f}".format(eps))
        self.epsilons.append(eps)

    def do_early_stop(self, epoch, current_loss):
        """
        Decides whether to stop the training process due to early stopping based
        on the early stopping criteria (parameters explained in the config documentation).
        """
        if (epoch + 1) < self.config["model_train"]["early_stop"]["first_epoch"]:
            return False
        else:
            losses_to_check = self.losses[
                -self.config["model_train"]["early_stop"]["n_previous_losses"] :
            ]
            pct_losses_lower_than_current_loss = np.sum(
                losses_to_check < current_loss
            ) / len(losses_to_check)
            if (
                pct_losses_lower_than_current_loss
                > self.config["model_train"]["early_stop"]["loss_pct"]
            ):
                print("Ending training procedure with early stopping")
                print(
                    "Current loss of {:0.2f} > {:0.2f}% of previous {}".format(
                        current_loss,
                        pct_losses_lower_than_current_loss * 100,
                        self.config["model_train"]["early_stop"]["n_previous_losses"],
                    )
                )
                self.losses.append(current_loss)
                return True
            else:
                return False

    def determine_final_iteration(self):
        """
        If early_stop_epsilon is enabled, this method should be called before the
        training begins in order to determine the final epoch/step for the requested
        maximum epsilon value.
        """
        if not self.config["diff_priv"]["enabled"]:
            raise ValueError(
                "early_stop_epsilon cannot be enabled when running without differential privacy"
            )

        sampling_probability = (
            self.config["model_train"]["batch_size"] / self.n_training_samples
        )
        max_eps = self.config["model_train"]["early_stop_epsilon"]["value"]

        epoch = 0
        eps = -np.inf
        while eps < max_eps:
            steps = (epoch + 1) * self.steps_per_epoch
            eps = self.compute_epsilon(
                steps=steps, sampling_probability=sampling_probability
            )
            epoch += 1

        final_epoch = epoch - 1
        eps = -np.inf
        step = 0
        while eps < max_eps:
            steps = final_epoch * self.steps_per_epoch + (step + 1)
            eps = self.compute_epsilon(
                steps=steps, sampling_probability=sampling_probability
            )
            step += 1

        final_step = step - 1
        self.final_iteration = (final_epoch, final_step)
        print(
            "Max Epsilon of {} reached at Epoch: {}, Step: {}".format(
                max_eps, final_epoch, final_step
            )
        )

    def is_final_iteration(self, epoch, step):
        """
        Determines whether the current epoch/step is the final iteration.

        If early_stop_epsilon is enabled, this method should be called after
        each step within an epoch. If the returned value is True, then
        the training proecedure should end.

        Parameters
        ----------
        epoch : int
            The current epoch.
        step : int
            The current step within the epoch.
        """
        if (epoch, step) == self.final_iteration:
            self.do_early_stop_epsilon = True
            return True
        else:
            return False

    def is_loss_nan(self):
        """
        Check if the losses contain nan. If yes, there is a problem with training
        so end the experiment.
        """
        if any(np.isnan(self.losses)):
            return True
        else:
            return False

    def record_subset_indices(self, data):
        """
        Based on the subsets passed to record gradients, this method will identify
        indices in the data that correspond to each subgroup. If no subset is passed
        it will consider the subset to be the entire dataset.

        It is called during the training method and enables the recording of
        gradients for different subgroups.
        """
        subset_indices_dict = subset_indices(
            data=data,
            subset=self.config["model_train"]["record_gradients"]["subset"],
            col_names=self.column_names,
        )
        for group, info in subset_indices_dict.items():
            info["values"] = []
            self.gradient_norms[group] = info

    # TODO change to global_norm instead of sum of norms
    def record_gradient_norms(self, group, optimizer, tape, loss_fn, var_list):
        """
        Records the gradient norms. When training with dp it will record the gradient
        both before and after the dp procedure. The gradients norms are saved as
        a tuple (norm_before_dp, norm_after_dp), with norms_after_dp=None when
        training without differential privacy.

        The method should be called in fit_model() of a model_experiment.
        """

        if self.config["diff_priv"]["enabled"]:
            # change the num_microbatches to work for the specific group
            # must be reset to train_num_microbatches after recording!
            train_num_microbatches = optimizer._num_microbatches
            if self.config["model_train"]["record_gradients"][
                "clip_gradient_per_sample"
            ]:
                num_microbatches = tf.constant(self.gradient_norms[group]["support"])
            else:
                num_microbatches = tf.constant(1)
            optimizer._num_microbatches = num_microbatches

            grads_and_vars = optimizer.compute_gradients(
                loss_fn, var_list, gradient_tape=tape
            )
            grads_dp = [item[0] for item in grads_and_vars]
            grads_dp_sum = tf.reduce_sum([tf.norm(grad) for grad in grads_dp]).numpy()
        else:
            grads_dp_sum = None
        grads = tape.gradient(tf.reduce_mean(loss_fn()), var_list)
        grads_sum = tf.reduce_sum([tf.norm(grad) for grad in grads]).numpy()
        self.gradient_norms[group]["values"].append((grads_sum, grads_dp_sum))

        if self.config["diff_priv"]["enabled"]:
            # reset to the num_microbatches used during training
            optimizer._num_microbatches = train_num_microbatches

    def fit_model(self, train_data):
        """
        This method should handle the entire training procedure for your model.

        Returns
        -------
        Fitted model class.
        """
        raise NotImplementedError

    def sample_model(self, model):
        """
        This method should sample fake data from your fitted model.

        Returns
        -------
        Fake data.
        """
        raise NotImplementedError
