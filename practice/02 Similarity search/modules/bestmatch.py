import numpy as np
import math
import copy

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    
    Parameters
    ----------
    array: the array to apply the exclusion zone to
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone
    
    Returns
    -------
    array: the array which is applied the exclusion zone
    """

    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start : zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    
    Parameters
    ----------
    dist_profile: distances between query and subsequences of time series
    excl_zone: size of the exclusion zone
    topK: count of the best match subsequences
    max_distance: maximum distance between query and a subsequence `S` for `S` to be considered a match
    
    Returns
    -------
    topK_match_results: dictionary containing results of algorithm
    """

    topK_match_results = {
        'indices': [],
        'distances': []
    } 

    dist_profile_len = len(dist_profile)
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder
    
    Parameters
    ----------
    excl_zone_frac: exclusion zone fraction
    topK: number of the best match subsequences
    is_normalize: z-normalize or not subsequences before computing distances
    r: warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """ 
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r


    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone
        
        Parameters
        ----------
        m: length of subsequence
        
        Returns
        -------
        excl_zone: exclusion zone
        """

        excl_zone = math.ceil(m * self.excl_zone_frac)

        return excl_zone


    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class NaiveBestMatchFinder
        """


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Поиск подпоследовательностей в временном ряде, которые наиболее близко соответствуют запросу с использованием наивного алгоритма
        
        Параметры
        ----------
        ts_data: временной ряд
        query: запрос, короче временного ряда

        Возвращает
        -------
        best_match: словарь, содержащий результаты наивного алгоритма
        """

        # Создаем копию запроса
        query = copy.deepcopy(query)

        # Если временной ряд не двумерный, преобразуем его в двумерный с помощью скользящего окна
        if (len(ts_data.shape) != 2):
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape  # N - количество подстрок, m - длина запроса
        excl_zone = self._calculate_excl_zone(m)  # Вычисляем размер исключающей зоны

        # Инициализируем профиль расстояний
        dist_profile = np.ones((N,)) * np.inf
        bestmatch = {
            'index': [],
            'distance': []
        }

        # Проходим по каждому началу подпоследовательности
        for start in range(N):
            if start + m <= N:  # Убедимся, что подпоследовательность помещается в временной ряд
                subsequence = ts_data[start]  # Извлекаем текущую подпоследовательность

                # Вычисляем расстояние DTW между запросом и текущей подпоследовательностью
                distance = DTW_distance(query, subsequence)

                # Обновляем профиль расстояний
                dist_profile[start] = distance

        # Используем функцию topK_match для получения лучших совпадений с учетом исключающей зоны
        topK_results = topK_match(dist_profile, excl_zone, self.topK)

        # Заполняем bestmatch с индексами и расстояниями
        bestmatch['index'] = topK_results['indices']
        bestmatch['distance'] = topK_results['distances']

        return bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        
        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """

        lb_Kim = 0
        
        lb_Kim = (subs1[0] - subs2[0]) ** 2 + (subs1[-1] - subs2[-1]) ** 2

        return lb_Kim


    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        r: warping window size
        
        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """

        lb_Keogh = 0

        n = len(subs1)
        lb_Keogh = 0.0

        # Compute the envelopes
        u = np.zeros(n)
        l = np.zeros(n)
        for i in range(n):
            window_start = max(0, i - r)
            window_end = min(n, i + r + 1)
            u[i] = max(subs1[window_start:window_end])
            l[i] = min(subs1[window_start:window_end])

        # Calculate LB_Keogh as the sum of squared differences for points outside the envelopes
        for i in range(n):
            if subs2[i] > u[i]:
                lb_Keogh += (subs2[i] - u[i]) ** 2
            elif subs2[i] < l[i]:
                lb_Keogh += (subs2[i] - l[i]) ** 2


        return lb_Keogh


    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using UCR-DTW algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of UCR-DTW algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape

        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf
        
        bestmatch = {
            'index' : [],
            'distance' : []
        }

        for start in range(N):
            subsequence = ts_data[start]
        if self.is_normalize:
            subsequence = z_normalize(subsequence)

        # Apply lower bound LB_Kim
        lb_Kim = self._LB_Kim(query, subsequence)
        if lb_Kim < bsf:
            # Apply lower bound LB_Keogh (QC)
            lb_Keogh_QC = self._LB_Keogh(query, subsequence, self.r)
            if lb_Keogh_QC < bsf:
                # Apply lower bound LB_Keogh (CQ)
                lb_Keogh_CQ = self._LB_Keogh(subsequence, query, self.r)
                if lb_Keogh_CQ < bsf:
                    # Calculate DTW distance
                    distance = DTW_distance(query, subsequence)
                    if distance < bsf:
                        dist_profile[start] = distance
                        bsf = distance

        # Extract top-K matches
        topK_results = topK_match(dist_profile, excl_zone, self.topK)
        bestmatch['index'] = topK_results['indices']
        bestmatch['distance'] = topK_results['distances']

        return bestmatch
