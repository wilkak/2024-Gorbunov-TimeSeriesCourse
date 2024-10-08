import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    return np.sqrt(np.sum((ts1 - ts2) ** 2))



def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)
    
    # Вычисление среднего арифметического и стандартного отклонения для каждого ряда
    mu_ts1 = np.mean(ts1)
    mu_ts2 = np.mean(ts2)
    sigma_ts1 = np.std(ts1)
    sigma_ts2 = np.std(ts2)
    
    # Вычисление скалярного произведения
    dot_product = np.dot(ts1, ts2)
    
    # Вычисление нормализованного евклидова расстояния
    norm_ed_dist = np.sqrt(np.abs(2 * n * (1 - (dot_product - n * mu_ts1 * mu_ts2) / (n * sigma_ts1 * sigma_ts2))))
    
    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)
    
    # Initialize the cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Apply the warping window size
    window = max(int(r * max(n, m)), 1)
    
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window)):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )
    
    return dtw_matrix[n, m]