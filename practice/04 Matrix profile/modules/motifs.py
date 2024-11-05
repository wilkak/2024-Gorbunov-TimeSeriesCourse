import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

     # Копируем матричный профиль и индексы, чтобы не изменять исходные данные
    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi'].copy()
    
    excl_zone = matrix_profile.get('excl_zone', 0)
    m = matrix_profile['m']
    
    for _ in range(top_k):
        # Находим индекс минимального значения в матричном профиле
        min_idx = np.argmin(mp)
        min_val = mp[min_idx]
        partner_idx = mpi[min_idx]
        
        # Добавляем найденную пару мотивов и их расстояние
        motifs_idx.append((min_idx, partner_idx))
        motifs_dist.append(min_val)
        
        # Применяем зону исключения вокруг обоих индексов
        mp = apply_exclusion_zone(mp, min_idx, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, partner_idx, excl_zone, np.inf)
    

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }

