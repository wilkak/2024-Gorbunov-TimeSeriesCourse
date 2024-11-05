import numpy as np

from modules.utils import *

def top_k_discords(matrix_profile: dict, excl_zone: int, top_k: int = 3) -> dict:
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []
    
    # Копируем массив значений профиля
    mp_copy = matrix_profile['mp'].copy()

    for _ in range(top_k):
        discord_idx = np.argmax(mp_copy)  # Индекс максимального значения в профиле
        discord_dist = mp_copy[discord_idx]  # Расстояние до ближайшего соседа
        nn_idx = int(matrix_profile['mpi'][discord_idx])  # Индекс ближайшего соседа
        
        # Сохраняем результаты
        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)
        
        # Применяем зону исключения
        mp_copy = apply_exclusion_zone(mp_copy, discord_idx, excl_zone, -np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }





def top_k_discords2(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []


    # Предполагается, что mp — это массив, а не словарь
    mp_copy = matrix_profile[:, 0].copy()  # Здесь находятся значения матричного профиля

    for _ in range(top_k):
        discord_idx = np.argmax(mp_copy)  # Находим индекс максимума (дискорд)
        discord_dist = mp_copy[discord_idx]  # Расстояние до ближайшего соседа
        
        # Получаем индекс ближайшего соседа
        nn_idx = int(mp[discord_idx, 1])
        
        # Сохраняем результаты
        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)
        
        # Применяем зону исключения
        mp_copy = apply_exclusion_zone(mp_copy, discord_idx, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
