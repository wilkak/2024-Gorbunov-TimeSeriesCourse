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
    
    ed_dist = 0

    # Вычисляем разность между временными рядами
    difference = ts1 - ts2
    # Возводим каждую разность в квадрат и суммируем
    squared_diff = np.sum(difference ** 2)
    # Берем квадратный корень из суммы
    ed_dist = np.sqrt(squared_diff)

    return ed_dist


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

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size (not used in this simple version)
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    n, m = len(ts1), len(ts2)

    # Инициализация матрицы расстояний
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Заполнение матрицы расстояний
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (ts1[i-1] - ts2[j-1]) ** 2  # Квадрат разности между элементами
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Вставка
                dtw_matrix[i, j-1],    # Удаление
                dtw_matrix[i-1, j-1]   # Замена
            )

    return dtw_matrix[n, m]  # Без извлечения квадратного корня

