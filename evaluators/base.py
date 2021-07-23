import pickle
from sys import exit
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from general.utils import (
    load_df,
    label_encode_categorical_columns,
    shuffle_data,
    apply_metatype,
    dict_contains_sub_dict,
)


class Evaluator:
    """
    Evaluator handles the full scope of evaluating the generated data after it is saved.

    Parameters
    ----------
    data_path : str
        Path for where to access the data and experiment output.

    config_file : str
        File name for the experiment output. Will be of the form "(config_hash).pkl".
    """

    def __init__(self, data_path, config_file):
        self._data_path = data_path
        self._config_file = config_file
        self.config_file = self._config_file

    @staticmethod
    def existing_configs(data_path, test_config=None):
        """
        See the experiment outputs that are saved containing user-specified parameters.

        Parameters
        ----------
        data_path : str
            Path for where to look for the experiment output data.

        test_config : dict (default=None)
            The parameters, structured in the same way as the config, that
            the experiment configs should contain.

        Returns
        -------
        out : dict
            Contains the configs for each of the valid (config_hash).pkl.

        Examples
        --------
        existing_configs = Evaluator.existing_configs(
            '../data/', {
            'model_train': {
                'transformer': {
                    'kwargs': {'std_outlier_threshold': 3},
                    'type': TransformerType.GENERAL
                }
            'diff_priv': {'enabled': True}
            }
        })
        """
        if test_config is None:
            test_config = {}

        out = {}
        pickle_files = [f for f in os.listdir(data_path) if f.split(".")[-1] == "pkl"]
        for file_name in pickle_files:
            data = pickle.load(open(os.path.join(data_path, file_name), "rb"))
            if not isinstance(data, dict):
                continue
            try:
                config = json.loads(data["config"])
            except (TypeError, json.decoder.JSONDecodeError):
                continue

            if dict_contains_sub_dict(config, test_config):
                out[file_name] = config

        return out

    @staticmethod
    def delete_config(file_path, data_path=None):
        if data_path:
            file_path = os.path.join(data_path, file_path)
        try:
            os.remove(file_path)
            print("{} removed".format(file_path))
        except FileNotFoundError:
            print("no file found")

    def get_experiment_data(self):
        """
        Get the data from a saved experiment using the path and config.
        """
        path = os.path.join(self._data_path, self.config_file)
        if not os.path.isfile(path):
            print("No file exists with this config -- exiting process")
            exit()
        return pickle.load(open(os.path.join(path), "rb"))

    def get_dataset(self, original_df=False):
        """
        Load the original data (or dataframe) used in an experiment.

        The label encoding and shuffling, with the same indexes as in the experiment,
        is applied. It is necessary to apply the saved shuffle index because that way,
        when the data is split with the saved train/test indices, the partitions  will
        correspond to the exact same partition used to train fake data. This allows
        for an accurate comparison of fake data and the specific train data partition
        used to train the generative model.
        """
        experiment_data = self.get_experiment_data()
        try:
            data_processing_config = json.loads(experiment_data["config"])[
                "data_processing"
            ]
        except KeyError:
            # to be compatible for configs before there was data_processing_config
            data_processing_config = {
                "columns_to_drop": None,
                "columns_rename_map": None,
                "read_csv_kwargs": None,
            }

        dataset_path = os.path.join(self._data_path, experiment_data["dataset"])
        if original_df:
            data = load_df(
                dataset_path,
                as_array=False,
                columns_to_drop=data_processing_config["columns_to_drop"],
                columns_rename_map=data_processing_config["columns_rename_map"],
                read_csv_kwargs=data_processing_config["read_csv_kwargs"],
            )
            data = data.iloc[experiment_data["shuffle_idx"]].reset_index(drop=True)
        else:
            data = load_df(
                dataset_path,
                as_array=True,
                columns_to_drop=data_processing_config["columns_to_drop"],
                columns_rename_map=data_processing_config["columns_rename_map"],
                read_csv_kwargs=data_processing_config["read_csv_kwargs"],
            )
            data = label_encode_categorical_columns(
                data, categorical_indices=experiment_data["categorical_indices"]
            )
            data = shuffle_data(data, shuffle_idx=experiment_data["shuffle_idx"])
        return data

    def fake_df(self, model_run=0):
        """
        The generated fake data from an experiment is returned as a np.array.
        This function will return the fake data as a dataframe with the categorical
        column values as the original string representations, as well as integer valued
        columns without decimal points.

        Parameters
        ----------
        model_run : int
            A model run number from the experiment.
        """
        original_df = self.get_dataset(original_df=True)
        data = self.get_experiment_data()
        fake_df = pd.DataFrame(
            data["model_runs"][model_run]["fake_data"], columns=data["column_names"]
        )

        label_encoder = LabelEncoder()
        for idx in data["categorical_indices"]:
            label_encoder.fit(original_df.iloc[:, idx])
            fake_df.iloc[:, idx] = label_encoder.inverse_transform(
                fake_df.iloc[:, idx].astype(int)
            )

        continuous_indices = np.setdiff1d(
            np.arange(len(data["column_names"])), data["categorical_indices"]
        )
        fake_df.iloc[:, continuous_indices] = fake_df.iloc[:, continuous_indices].apply(
            apply_metatype, axis=0
        )
        return fake_df
