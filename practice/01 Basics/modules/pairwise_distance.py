import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        if self.metric == 'euclidean':
            return ED_distance
        elif self.metric == 'norm_euclidean':
            return norm_ED_distance
        elif self.metric == 'dtw':
            return DTW_distance
        else:
            raise ValueError("Unsupported metric. Choose 'euclidean', 'norm_euclidean' or 'dtw'.")


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        
        if self.is_normalize and self.metric != 'norm_euclidean':
            input_data = np.array([z_normalize(ts) for ts in input_data])
        
        dist_func = self._choose_distance()
        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)
        
        for i in range(input_data.shape[0]):
            for j in range(i, input_data.shape[0]):
                distance = dist_func(input_data[i], input_data[j])
                matrix_values[i, j] = distance
                matrix_values[j, i] = distance
        
        return matrix_values
