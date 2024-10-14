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
    Вычисление расстояния DTW с учетом полосы Сако-Чиба.

    Параметры
    ----------
    ts1: первый временной ряд
    ts2: второй временной ряд
    r: размер полосы искажения как доля от длины временных рядов
    
    Возвращает
    -------
    dtw_dist: расстояние DTW между ts1 и ts2
    """

    n = len(ts1)
    m = len(ts2)

    # Инициализация матрицы затрат
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Применение размера полосы искажения
    window = max(int(r * max(n, m)), 0)

    for i in range(1, n + 1):
        # Определяем диапазон j на основе полосы Сако-Чиба
        if window == 0:  # Если r = 0, сравниваем только диагональные элементы
            j = i
            if j <= m:
                cost = (ts1[i - 1] - ts2[j - 1]) ** 2
                dtw_matrix[i, j] = cost + dtw_matrix[i - 1, j - 1]
        else:
            start_j = max(1, i - window)
            end_j = min(m + 1, i + window)

            for j in range(start_j, end_j):
                cost = (ts1[i - 1] - ts2[j - 1]) ** 2  # Вычисление стоимости
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # вставка
                    dtw_matrix[i, j - 1],      # удаление
                    dtw_matrix[i - 1, j - 1]   # совпадение
                )

    # Возвращаем без взятия корня, чтобы сравнить с функцией из sktime
    return dtw_matrix[n, m]
