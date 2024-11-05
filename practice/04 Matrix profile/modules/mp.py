import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config

import numpy as np
import stumpy

def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """
    
  
     # Вычисляем матричный профиль: если ts2 не указан, используем ts1 для self-join
    if ts2 is None:
        mp_result = stumpy.stump(ts1, m)
    else:
        mp_result = stumpy.stump(ts1, m, T_B=ts2)

    # Возвращаем структуру с результатами
    return {
        'mp': mp_result[:, 0],          # Матричный профиль
        'mpi': mp_result[:, 1].astype(int), # Индексы матричного профиля
        'm': m,                         # Длина подпоследовательности
        'data': {'ts1': ts1, 'ts2': ts2} # Данные временных рядов
    }
