from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from evaluators.base import Evaluator
from general.utils import subset_indices


class EvaluatorModelType:
    # regression models
    LinearRegression = LinearRegression
    RandomForestRegressor = RandomForestRegressor
    GradientBoostingRegressor = GradientBoostingRegressor
    AdaBoostRegressor = AdaBoostRegressor

    # classification models
    LogisticRegression = LogisticRegression
    RandomForestClassifier = RandomForestClassifier
    GradientBoostingClassifier = GradientBoostingClassifier
    AdaBoostClassifier = AdaBoostClassifier


class ScalerType:
    MINMAX = MinMaxScaler
    ROBUST = RobustScaler
    STANDARD = StandardScaler


class MetricType:
    # regression metrics
    R2_SCORE = "r2_score"
    MSE = "mse"

    # classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1_score"


METRIC_MAP = {
    "r2_score": r2_score,
    "mse": mean_squared_error,
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
}


# TODO Add H2o AutoML as an evaluation option
# TODO Add categorical correlations with separate calculation to heatmap
# TODO Add a distance evaluation between generated and ground truth continuous dists
class MachineLearningEvaluator(Evaluator):
    """
    MachineLearningEvaluator evaluates the quality of the data by testing its
    usefulness for machine learning tasks and examining the generated distributions.

    Parameters
    -----------
    data_path : str
        Path for where to access the data and experiment output.

    config_file : str
        File name for the experiment output. Will be of the form "(config_hash).pkl".

    target_col : str (default=None)
        Target column to be used to evaluate the fake data for machine learning.
        Only the plotting functions can be utilized if target_col is None.
    """

    def __init__(self, data_path, config_file, target_col=None):
        super().__init__(data_path=data_path, config_file=config_file)
        self._target_col = target_col

    @staticmethod
    def transform_data(transformer, data, categorical_indices, num_categories):
        transformer.fit(data, categorical_indices, num_categories)
        return transformer.transform(data)

    def shift_categorical_indices(self, x, target_col):
        temp = np.setdiff1d(x, target_col)
        return np.where(temp > target_col, temp - 1, temp)

    def data_for_evaluation(self, scaler, holdout=None, subset=None):
        """
        Prepare the real and fake data for evaluation:
            1. Split the features and target.
            2.  Identify the possible shift in categorical indices after removing
               the target_col.
            3. One hot encode categorical features and scale continuous features.
            4. Save the train/test splits for the real and fake data.

        Parameters
        ----------
        scaler : sklearn scaler object

        holdout : np.ndarray
            A set of holdout data in the same form as the train data.

        subset : list of strings
            The name of a column in the original data to subset the scores. It should
            be a categorical column. Only the test scores can be subset.
        """
        if subset is None:
            subset = []

        experiment_data = self.get_experiment_data()
        column_names = experiment_data["column_names"]
        if self._target_col not in column_names:
            raise ValueError("{} not present in column names".format(self._target_col))
        target_col = experiment_data["column_names"].index(self._target_col)
        full_data = self.get_dataset()

        categorical_indices = experiment_data["categorical_indices"]
        shifted_categorical_indices = self.shift_categorical_indices(
            categorical_indices, target_col
        )
        is_target_col = np.isin(np.arange(full_data.shape[1]), [target_col])
        is_categorical_col = np.isin(
            np.arange(full_data.shape[1] - 1), shifted_categorical_indices
        )

        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        out = []
        for model_run in experiment_data["model_runs"]:
            train = full_data[model_run["train_idx"]]
            fake_train = model_run["fake_data"]
            if holdout is None:
                test = full_data[model_run["test_idx"]]
            else:
                test = holdout

            model_run_out = defaultdict(dict)
            for train_type, train_data in [("real", train), ("fake", fake_train)]:
                X_train, y_train = (
                    train_data[:, ~is_target_col],
                    train_data[:, is_target_col],
                )
                enc.fit(X_train[:, is_categorical_col])
                X_train_cat = enc.transform(X_train[:, is_categorical_col])
                scaler.fit(X_train[:, ~is_categorical_col])
                X_train_cont = scaler.transform(X_train[:, ~is_categorical_col])
                X_train = np.concatenate([X_train_cat, X_train_cont], axis=1)

                X_test, y_test = test[:, ~is_target_col], test[:, is_target_col]
                X_test_cat = enc.transform(X_test[:, is_categorical_col])
                X_test_cont = scaler.transform(X_test[:, ~is_categorical_col])
                X_test = np.concatenate([X_test_cat, X_test_cont], axis=1)

                model_run_out[train_type]["X_train"] = X_train
                model_run_out[train_type]["y_train"] = y_train
                model_run_out[train_type]["X_test"] = X_test
                model_run_out[train_type]["y_test"] = y_test

            model_run_out["subset_indices"] = subset_indices(
                data=test, subset=subset, col_names=column_names, include_names=True
            )
            out.append(dict(model_run_out))
        return out

    @staticmethod
    def predict_score(x, y, mdl, metric, metric_kwargs):
        return METRIC_MAP[metric](y, mdl.predict(x), **metric_kwargs[metric])

    def data_utility_scores(
        self,
        model,
        metrics,
        scaler_type=None,
        scaler_kwargs=None,
        model_kwargs=None,
        metric_kwargs=None,
        holdout=None,
        subset=None,
        include_train=False,
    ):
        """
        The quality of generated data can be evaluated with the following strategy:
            1. Train a model using real train data.
            2. Train a model using fake data.
            3. Predict a test set of real data using both models.
            4. Compare the quality of the predictions.

        The models and metrics should be accessed through EvaluationModelType and
        MetricType classes. They contain scikit-learn estimators and metrics. New ones
        can easily be added. Model parameters should be passed through model kwargs.

        Parameters
        ----------
        model : sklearn model as EvaluatorModelType
            The scikit-learn model to be used.
        metrics : sklearn metric(s) as MetricType or list of MetricType
            The scikit-learn metric to be used.
        scaler_type : sklearn scaler as ScalerType
            The scikit-learn scaler to be used for continuous feature scaling.
        scaler_kwargs : dict
            The arguments for the scaler.
        model_kwargs : dict
            The model parameters.
        metric_kwargs : dict
            The metric parameters. Can include parameters for only a single metric
            i.e. metric_kwargs={MetricType.F1: {"average": "weighted"}}
        holdout : np.ndarray
            A set of holdout data in the same form as the train data. If holdout is
            None, then the scores are validation scores.
        subset : list of strings
            The name of a column in the original data to subset the scores. It should
            be a categorical column. The model is fit using the whole dataset, but
            the returned scores will be given for each of the classes in the subset col.
            Only the test scores can be subset.
        include_train : bool (default=False)
            Whether to include train (in-sample) scores.
        """
        if scaler_type is None:
            scaler_type = ScalerType.MINMAX

        if scaler_kwargs is None:
            scaler_kwargs = {}

        if model_kwargs is None:
            model_kwargs = {}

        if not isinstance(metrics, list):
            metrics = [metrics]

        if metric_kwargs is None:
            metric_kwargs = {}
        for metric in metrics:
            if metric not in metric_kwargs:
                metric_kwargs[metric] = {}

        if holdout is not None and not isinstance(holdout, np.ndarray):
            raise ValueError("If passing a holdout set it should be an np.array")

        if subset is None:
            subset = []

        scaler = scaler_type(**scaler_kwargs)
        eval_data = self.data_for_evaluation(scaler, holdout, subset)

        all_scores = []
        for model_run in eval_data:
            scores = {}
            for train_type in ["fake", "real"]:
                # instantiate and fit model
                mdl = model(**model_kwargs)
                mdl.fit(
                    model_run[train_type]["X_train"],
                    model_run[train_type]["y_train"].ravel(),
                )
                data_splits = [("test", ("X_test", "y_test"))]
                if include_train:
                    data_splits.append(("train", ("X_train", "y_train")))
                for split_name, (x_key, y_key) in data_splits:
                    x = model_run[train_type][x_key]
                    y = model_run[train_type][y_key].ravel()
                    for metric in metrics:
                        col_name = "{}_{}_{}".format(train_type, split_name, metric)
                        if subset and split_name == "test":
                            for group, info in model_run["subset_indices"].items():
                                if info["support"]:
                                    x_subset = x[info["idx"], :]
                                    y_subset = y[info["idx"]]
                                    col_name_subset = "{}_{}".format(
                                        col_name, info["name"]
                                    )
                                    scores[col_name_subset] = self.predict_score(
                                        x_subset, y_subset, mdl, metric, metric_kwargs
                                    )
                        else:
                            scores[col_name] = self.predict_score(
                                x, y, mdl, metric, metric_kwargs
                            )
            all_scores.append(scores)

        # handle case where scores are returned for each target class (average=None)
        new_all_scores = []
        for score_dict in all_scores:
            new_score_dict = {}
            for k, v in score_dict.items():
                if isinstance(v, np.ndarray):
                    for i, sc in enumerate(v):
                        new_score_dict[k + "_target{}".format(i)] = sc
                else:
                    new_score_dict[k] = v
            new_all_scores.append(new_score_dict)

        scores_df = pd.DataFrame(new_all_scores)
        return scores_df.reindex(
            sorted(scores_df.columns, key=lambda col: col.split("_")[1:]), axis=1
        )

    @staticmethod
    def plot_data_utility_scores(
        target_col,
        data_path,
        configs,
        model,
        metrics,
        include_real=False,
        scaler_type=None,
        scaler_kwargs=None,
        model_kwargs=None,
        metric_kwargs=None,
        holdout=None,
        subset=None,
    ):
        """
        Wrapper function for data_utility_scores to plot a group, or groups, of configs.
        All of the configurations should be using the same dataset.

        Only the parameters that are not included in data_utility_scores() are
        described.

        Parameters
        ----------
        data_path : str
            Path for where to access the data and experiment output.

        target_col : str (default=None)
            Target column to be used to evaluate the fake data for machine learning.
            Only the plotting functions can be utilized if target_col is None.

        configs : dict of dict(s)
            Key should be the name of the group of configs. It is the label that will
            appear in the legend.
            Value should be the output of the method in Base, existing_configs. All the
            scores will be considered as one group.

        include_real : bool (default=False)
            Whether to include the score for the model fit with real data in the plots.
            Since this has the same expectation across all runs it would be redundant
            to include for all config groups. Therefore, the scores from the last group
            of configs to be processed are used.
        """

        all_group_scores = {}
        for group_name, group_configs in configs.items():
            group_scores = []
            for config_hash, config in group_configs.items():
                eval_input = {
                    "target_col": target_col,
                    "data_path": data_path,
                    "config_file": config_hash,
                }
                evaluator = MachineLearningEvaluator(**eval_input)
                scores = evaluator.data_utility_scores(
                    model=model,
                    metrics=metrics,
                    scaler_type=scaler_type,
                    scaler_kwargs=scaler_kwargs,
                    model_kwargs=model_kwargs,
                    metric_kwargs=metric_kwargs,
                    holdout=holdout,
                    subset=subset,
                    include_train=False,
                )
                group_scores.append(scores)
            all_group_scores[group_name] = pd.concat(group_scores, axis=0)
        columns = list(scores.columns)
        cols_to_plot = [col for col in columns if "fake" in col]
        img_per_row = 3
        rows = int(np.ceil(len(cols_to_plot) / img_per_row))
        fig, axes = plt.subplots(rows, img_per_row, figsize=(20, rows * 5))
        if axes.ndim == 1:
            axes = np.expand_dims(axes, 0)
        fig.subplots_adjust(hspace=0.35, wspace=0.2)
        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                col_idx = i * img_per_row + j
                if col_idx < len(cols_to_plot):
                    col_name = cols_to_plot[col_idx]
                    for group_name, score_df in all_group_scores.items():
                        sns.distplot(
                            score_df.loc[:, col_name].values, label=group_name, ax=ax
                        )
                    if include_real:
                        col_name_real = col_name.replace("fake", "real")
                        sns.distplot(
                            score_df.loc[:, col_name_real].values, label="real", ax=ax
                        )
                    ax.set_title("_".join(col_name.split("_")[2:]))
                    ax.legend()
                else:
                    break
        plt.show()

    def plot_distributions(
        self, model_run=None, include_test=False, specific_dist=None
    ):
        """
        Display the histograms/kdes of the features. With all of the default parameters
        the function will plot the train, fake train, and test distributions of the
        first model run. Only one of model_run and specific_dist should be passed.

        Parameters
        ----------
        model_run: int (default=None)
            An experiment can contain multiple model runs if k-fold.
        include_test : bool (default=True)
            Whether to include the test data in the plots. It can sometimes be
            difficult to see the train/fake train comparison clearly with test included.
        specific_dist : str (default=None)
            The specific distribution to plot over each of the model runs. For example,
            if "fake_train" is passed, it will plot the fake train data for each of the
            features over all runs. This is only useful if k_fold=True in the config.
        """
        if model_run is not None and specific_dist is not None:
            raise ValueError("only one of model_run or specific_dist should be passed")
        elif model_run is None and specific_dist is None:
            model_run = 0

        experiment_data = self.get_experiment_data()
        df = self.get_dataset()
        img_per_row = 3
        fig, axes = plt.subplots(
            int(np.ceil(df.shape[1] / img_per_row)), img_per_row, figsize=(20, 15)
        )
        fig.subplots_adjust(hspace=0.35, wspace=0.2)

        if model_run is not None:
            data = [
                df[experiment_data["model_runs"][model_run]["train_idx"], :],
                experiment_data["model_runs"][model_run]["fake_data"],
            ]
            labels = ["train", "fake_train"]
            if include_test:
                test_idx = experiment_data["model_runs"][model_run]["test_idx"]
                if test_idx is None:
                    msg = "cannot include test because experiment was run with no test set"
                    logging.warning(msg)
                else:
                    data.append(df[test_idx, :])
                    labels.append("test")
        else:
            num_model_runs = len(experiment_data["model_runs"])
            data = []
            if specific_dist == "fake_train":
                for i in range(num_model_runs):
                    data.append(experiment_data["model_runs"][i]["fake_data"])
            elif specific_dist == "train":
                for i in range(num_model_runs):
                    data.append(df[experiment_data["model_runs"][i]["train_idx"], :])
            else:
                raise ValueError("{} is not a handled dist".format(specific_dist))
            labels = ["fold_{}".format(i) for i in range(num_model_runs)]

        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                col_idx = i * img_per_row + j
                if col_idx < df.shape[1]:
                    with_kde = (
                        False
                        if col_idx in experiment_data["categorical_indices"]
                        else True
                    )
                    for nd, d in enumerate(data):
                        sns.distplot(
                            d[:, col_idx], label=labels[nd], ax=ax, kde=with_kde
                        )
                    ax.legend()
                    ax.set_title(experiment_data["column_names"][col_idx])
                else:
                    break
        plt.show()

    def plot_heatmap(self):
        """
        Display a heatmap of the correlation coefficient between continuous features.
        """
        experiment_data = self.get_experiment_data()
        mask = np.ones(len(experiment_data["column_names"]), dtype=bool)
        mask[experiment_data["categorical_indices"]] = False
        df = self.get_dataset(original_df=True).iloc[:, mask]

        img_per_row = 3
        model_runs = experiment_data["model_runs"]
        fig, axes = plt.subplots(len(model_runs), img_per_row, figsize=(20, 15))
        if len(model_runs) == 1:
            axes = axes.reshape(1, img_per_row)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        for i, row in enumerate(axes):
            model_run = model_runs[i]
            fake_train = pd.DataFrame(
                model_run["fake_data"], columns=experiment_data["column_names"]
            ).iloc[:, mask]
            train = df.iloc[model_run["train_idx"]]
            test = df.iloc[model_run["test_idx"]]
            for j, (title, data, ax) in enumerate(
                list(
                    zip(["fake_train", "train", "test"], [fake_train, train, test], row)
                )
            ):
                sns.heatmap(data.corr(), annot=True, cmap="RdYlGn", square=True, ax=ax)
                ax.set_title("{} {}".format(title, i))
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()


if __name__ == "__main__":
    # see configs that exists with specific parameters
    test_config = {
        "diff_priv": {"enabled": True},
        "model_train": {"transformer": {"kwargs": {"outlier_clipping": True}}},
    }
    existing_configs = MachineLearningEvaluator.existing_configs(
        "../data/", test_config
    )

    # instantiate evaluator
    eval_input = {
        "target_col": "default",
        "data_path": "../data/",
        "config_file": "587636d21c44fc0a3262cdf36fdb4801481a9603.pkl",
    }
    evaluator = MachineLearningEvaluator(**eval_input)

    # load generated data
    fake_data = evaluator.fake_df()

    # calculate the scores for each model run
    scores = evaluator.data_utility_scores(
        EvaluatorModelType.LogisticRegression,
        [MetricType.F1],
        ScalerType.ROBUST,
        model_kwargs={"solver": "lbfgs", "max_iter": 200, "class_weight": "balanced"},
        metric_kwargs={MetricType.F1: {"average": None}},
        subset="SEX",
        include_train=False,
    )
    print(scores)
