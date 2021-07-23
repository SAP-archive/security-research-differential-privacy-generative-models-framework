import numpy as np
import pandas as pd
from general.utils import get_metatype, DataTypes


class Transformer:
    """
    The Transformer class converts data that is already pre-processed into a
    format suitable for model training. It also does the inverse transformation
    to convert generated data back to the original format.

    It is important to note that although transformers and models are independent, it
    generally makes sense to scale continuous features within the same range as the
    output activation that will be applied to them. For example, if a config uses
    the parameter `cont_feature_range` to scale the original data between -1 and 1,
    then tanh is a suitable continuous feature output activation function to apply.

    The following code is adapted from SDGYM - an open source framework
    for training and evaluating generative models on tabular data. The major issue with
    the open source version is that it breaks when the generated data does not include
    all of the possible categories for categorical variables. Other changes have also
    been made.

    https://github.com/sdv-dev/SDGym/blob/master/sdgym/synthesizers/utils.py

    Parameters
    ----------
    outlier_clipping : bool (default=False)
        Whether to clip outliers for continuous features.

    outlier_std_threshold : float (default=4)
        Continuous scaling methods, such as min-max scaling, are highly influenced by
        the presence of outliers. It may be beneficial to clip the outliers before
        scaling. This parameter controls how many standard deviations away from the mean
        you would like to clip.

    cont_feature_range : tuple of floats (default=(-1, 1))
        Continuous feature should be on the same scale for the model optimization
        procedure to work properly. This parameter specifies the range bounds.

    Attributes
    ----------
    output_dim : int
        The total number of dimensions in the transformed data.
    output_info : list of tuples
        Contains the transformed dimension of an original feature and
        the output activation to apply.
    """

    def __init__(
        self,
        outlier_clipping=False,
        outlier_std_threshold=4,
        cont_feature_range=(-1, 1),
    ):
        self._outlier_clipping = outlier_clipping
        self._outlier_std_threshold = outlier_std_threshold
        self._cont_feature_range = cont_feature_range

        self._meta = None
        self.output_dim = 0
        self.output_info = []
        self._scaler_fits = {}

    @staticmethod
    def get_metadata(data, categorical_columns, num_categories):
        meta = []
        num_categories = dict(zip(categorical_columns, num_categories))

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]
            index_meta = {
                "name": index,
            }

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                index_meta.update(
                    {
                        "type": DataTypes.CATEGORICAL,
                        "size": num_categories[index],  # the important change
                        "i2s": mapper,
                    }
                )
                meta.append(index_meta)
            else:
                index_meta.update(
                    {
                        "type": DataTypes.CONTINUOUS,
                        "meta_type": get_metatype(column.values),
                        "min": column.min(),
                        "max": column.max(),
                    }
                )
                meta.append(index_meta)

        return meta

    def clip_outliers(self, x):
        mean = np.mean(x)
        std = np.std(x)
        pos_outlier_mask = x > (mean + std * self._outlier_std_threshold)
        neg_outlier_mask = x < (mean - std * self._outlier_std_threshold)
        return np.clip(x, np.min(x[~neg_outlier_mask]), np.max(x[~pos_outlier_mask]))

    def fit(self, data, categorical_columns, num_categories):
        """
        Save relevant information about the transformation.

        Parameters
        ----------
        data : np.array
        categorical_columns : list
        num_categories : list
        """
        raise NotImplementedError

    def transform(self, data):
        """
        Transform the original data into a format to be passed to the model.

        Parameters
        ----------
        data : np.array
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        Apply the inverse transformation to convert data generated from the model
        to the original format.

        Parameters
        ----------
        data : np.array
        """
        raise NotImplementedError


class PassThroughTransformer(Transformer):
    def fit(self, data, categorical_columns, num_categories):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data
