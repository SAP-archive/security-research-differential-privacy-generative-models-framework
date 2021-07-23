from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from general.utils import MetaDataTypes, DataTypes
from transformers.base import Transformer


class BGMTransformer(Transformer):
    """
    Apply mode-specific normalization to continuous features. Section 4.2 of reference:
    https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf .

    Convert categorical features to a one-hot vector.

    Parameters
    ----------
    clip_outliers_first : bool (default=False)
        If outlier_clipping=True, then outliers can be clipped either before the
        mode-specific normalization procedure or after. When clip_outliers_first=False
        the outlier clipping occurs after the transformation.

    normalize_between_true_range : bool (default=False)
        Applying the inverse transformation may result in generated features falling
        outside the range of the original feature space. If True, the generated feature
        will be re-scaled for the range to be the same as the original feature.

    n_clusters : int (default=5)
        The upper bound on the number of modes for the Bayesian Gaussian Mixture Model.

    eps : float (default=0.005)
        The minimum value of a mode weight for the BGM such that it will be
        included in the output components.
    """

    def __init__(
        self,
        outlier_clipping=True,
        outlier_std_threshold=4,
        cont_feature_range=(-1, 1),
        clip_outliers_first=False,
        cont_range_clipping=False,
        normalize_between_true_range=False,
        n_clusters=6,
        eps=0.005,
    ):
        super().__init__(
            outlier_clipping=outlier_clipping,
            outlier_std_threshold=outlier_std_threshold,
            cont_feature_range=cont_feature_range,
        )
        self._clip_outliers_first = clip_outliers_first
        self._cont_range_clipping = cont_range_clipping
        self._normalize_between_true_range = normalize_between_true_range
        self._n_clusters = n_clusters
        self._eps = eps

        self._components = []
        self._model = []

    def fit(self, data, categorical_columns, num_categories):
        """
        Fit the BGM for each of the continuous features and save the relevant data.
        """
        self._meta = self.get_metadata(data, categorical_columns, num_categories)
        model = []

        for id_, info in enumerate(self._meta):
            if info["type"] == DataTypes.CONTINUOUS:
                gm = BayesianGaussianMixture(
                    self._n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    n_init=1,
                )
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                comp = gm.weights_ > self._eps
                self._components.append(comp)

                self.output_info += [
                    (1, DataTypes.CONTINUOUS),
                    (np.sum(comp), DataTypes.CATEGORICAL),
                ]
                self.output_dim += 1 + np.sum(comp)
            else:
                model.append(None)
                self._components.append(None)
                self.output_info += [(info["size"], DataTypes.CATEGORICAL)]
                self.output_dim += info["size"]

        self._model = model

    def transform(self, data):
        """
        For continuous features apply the following transformation:
            1. For the current feature, extract the means and standard deviations
               from the BGM modes.
            2. If specified, clip outliers before transformation.
            3. Standardize the feature with respect to each of the modes by applying the
               transformation: (feature - mode_mean) / (mode_std * 4).
            4. For each value of a given feature, sample one of the mode specific feature
               transformations using the probabilities that the value comes from each of
               the modes.
            5. If specified, clip outliers after mode-specific normalization.
            6. The feature needs to be within the cont_feature_range. Either clip
               all values outside the bound or apply min-max scaling.
            7. One-hot encode a vector indicating the selected mode.

        Categorical features are one-hot encoded.

        """
        values = []
        for id_, info in enumerate(self._meta):
            current = data[:, id_]
            if info["type"] == DataTypes.CONTINUOUS:
                current = current.reshape([-1, 1])
                if self._outlier_clipping and self._clip_outliers_first:
                    current = self.clip_outliers(current)

                means = self._model[id_].means_.reshape((1, self._n_clusters))
                stds = np.sqrt(self._model[id_].covariances_).reshape(
                    (1, self._n_clusters)
                )
                features = (current - means) / (4 * stds)

                probs = self._model[id_].predict_proba(current.reshape([-1, 1]))

                n_opts = sum(self._components[id_])
                features = features[:, self._components[id_]]
                probs = probs[:, self._components[id_]]

                opt_sel = np.zeros(len(data), dtype="int")
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])

                if self._outlier_clipping and not self._clip_outliers_first:
                    features = self.clip_outliers(features)

                if self._cont_range_clipping:
                    features = np.clip(
                        features,
                        self._cont_feature_range[0],
                        self._cont_feature_range[1],
                    )
                else:
                    scaler = MinMaxScaler(feature_range=self._cont_feature_range)
                    scaler.fit(features)
                    self._scaler_fits[id_] = scaler
                    features = scaler.transform(features)

                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                values += [features, probs_onehot]
            else:
                col_t = np.zeros([len(data), info["size"]])
                col_t[np.arange(len(data)), current.astype("int32")] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        """
        Apply the inverse of the transformation procedure to continuous features and
        take the argmax of generated categorical feature vectors.
        """
        data_t = np.zeros([len(data), len(self._meta)])

        st = 0
        for id_, info in enumerate(self._meta):
            if info["type"] == DataTypes.CONTINUOUS:
                u = data[:, st]
                v = data[:, st + 1 : st + 1 + np.sum(self._components[id_])]

                v_t = np.ones((data.shape[0], self._n_clusters)) * -100
                v_t[:, self._components[id_]] = v
                v = v_t
                st += 1 + np.sum(self._components[id_])
                means = self._model[id_].means_.reshape([-1])
                stds = np.sqrt(self._model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]

                if self._cont_range_clipping:
                    u = np.clip(
                        u, self._cont_feature_range[0], self._cont_feature_range[1]
                    )
                else:
                    u = (
                        self._scaler_fits[id_]
                        .inverse_transform(u.reshape(-1, 1))
                        .ravel()
                    )

                tmp = u * 4 * std_t + mean_t

                if self._normalize_between_true_range:
                    tmp = (
                        MinMaxScaler(feature_range=(info["min"], info["max"]))
                        .fit_transform(tmp.reshape(-1, 1))
                        .ravel()
                    )

                if info["meta_type"] == MetaDataTypes.INT:
                    tmp = np.rint(tmp)
                data_t[:, id_] = tmp
            else:
                current = data[:, st : st + info["size"]]
                st += info["size"]
                data_t[:, id_] = np.argmax(current, axis=1)

        return data_t
