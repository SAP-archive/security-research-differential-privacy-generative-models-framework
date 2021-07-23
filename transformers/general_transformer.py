import numpy as np
from sklearn.preprocessing import MinMaxScaler

from general.utils import MetaDataTypes, DataTypes
from transformers.base import Transformer


class GeneralTransformer(Transformer):
    """
    Continuous features are scaled with min-max scaling between the cont_feature_range.
    Categorical features are converted to a one-hot vector.
    """

    def __init__(
        self,
        outlier_clipping=False,
        outlier_std_threshold=4,
        cont_feature_range=(-1, 1),
    ):
        super().__init__(
            outlier_clipping=outlier_clipping,
            outlier_std_threshold=outlier_std_threshold,
            cont_feature_range=cont_feature_range,
        )

    def fit(self, data, categorical_columns, num_categories):
        self._meta = self.get_metadata(data, categorical_columns, num_categories)
        for info in self._meta:
            if info["type"] in [DataTypes.CONTINUOUS]:
                self.output_dim += 1
                self.output_info.append((1, DataTypes.CONTINUOUS))
            else:
                self.output_dim += info["size"]
                self.output_info.append((info["size"], DataTypes.CATEGORICAL))

    def transform(self, data):
        """
        Scale continuous features with min-max scaling.

        Convert categorical features to one-hot encoding.
        """
        data_t = []
        for id_, info in enumerate(self._meta):
            col = data[:, id_]
            if info["type"] == DataTypes.CONTINUOUS:
                if self._outlier_clipping:
                    col = self.clip_outliers(col)

                scaler = MinMaxScaler(feature_range=self._cont_feature_range)
                col = col.reshape(-1, 1)
                scaler.fit(col)
                self._scaler_fits[id_] = scaler
                col = scaler.transform(col)
                data_t.append(col)
            else:
                col_t = np.zeros([len(data), info["size"]])
                col_t[np.arange(len(data)), col.astype("int32")] = 1
                data_t.append(col_t)

        return np.concatenate(data_t, axis=1)

    def inverse_transform(self, data):
        """
        Re-scale generated continuous features back to original range and round if
        the original feature is an int.

        Take the argmax of generated categorical feature vectors to represent the
        generated category.
        """
        data_t = np.zeros([len(data), len(self._meta)])

        data = data.copy()
        for id_, info in enumerate(self._meta):
            if info["type"] == DataTypes.CONTINUOUS:
                current = data[:, 0]
                data = data[:, 1:]
                current = (
                    self._scaler_fits[id_]
                    .inverse_transform(current.reshape(-1, 1))
                    .ravel()
                )
                if info["meta_type"] == MetaDataTypes.INT:
                    current = np.rint(current)
                data_t[:, id_] = current
            else:
                current = data[:, : info["size"]]
                data = data[:, info["size"] :]
                data_t[:, id_] = np.argmax(current, axis=1)

        return data_t
