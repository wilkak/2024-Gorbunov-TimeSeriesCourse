import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from typing_extensions import Self

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class TimeSeriesHierarchicalClustering:
    """
    Hierarchical Clustering of time series

    Parameters
    ----------
    n_clusters: number of clusters
    method: linkage criterion.
            Options: {single, complete, average, weighted}
    """

    def __init__(self, n_clusters: int = 3, method: str = 'complete') -> None:

        self.n_clusters: int = n_clusters
        self.method: str = method
        self.model: AgglomerativeClustering | None = None
        self.linkage_matrix: np.ndarray | None = None


    def fit(self, distance_matrix: np.ndarray) -> Self:
        """
        Fit the agglomerative clustering model based on distance matrix

        Parameters
        ----------
        distance_matrix: distance matrix between instances of dataset with shape (ts_number, ts_number)
        
        Returns
        -------
        self: the fitted model
        """

        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage=self.method)
        self.model.fit(distance_matrix)
        self.linkage_matrix = linkage(distance_matrix, method=self.method)

        return self


    def fit_predict(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Fit the agglomerative clustering model based on distance matrix and predict classes

        Parameters
        ----------
        distance_matrix: distance matrix between instances of dataset with shape (ts_number, ts_number)
        
        Returns
        -------
            predicted labels 
        """

        self.fit(distance_matrix)

        return self.model.labels_


    def _draw_timeseries_allclust(self, dx: pd.DataFrame, labels: np.ndarray, leaves: list[int], gs: gridspec.GridSpec, ts_hspace: int) -> None:
        """ 
        Plot time series graphs beside dendrogram

        Parameters
        ----------
        dx: timeseries data with column "y" indicating cluster number
        labels: labels of dataset's instances
        leaves: leave node names from scipy dendrogram
        gs: gridspec configurations
        ts_hspace: horizontal space in gridspec for plotting time series
        """

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        margin = 7

        max_cluster = len(leaves)
        # flip leaves, as gridspec iterates from top down
        leaves = leaves[::-1]

        for cnt in range(len(leaves)):
            plt.subplot(gs[cnt:cnt+1, max_cluster-ts_hspace:max_cluster])
            plt.axis("off")

            # get leafnode name, which corresponds to original data index
            leafnode = leaves[cnt]
            ts = dx[leafnode]
            ts_len = ts.shape[0] - 1

            label = int(labels[leafnode])
            color_ts = colors[label]

            plt.plot(ts, color=color_ts)
            plt.text(ts_len+margin, 0, f'class = {label}')


    def plot_dendrogram(self, df: pd.DataFrame, labels: np.ndarray, ts_hspace: int = 12, title: str = 'Dendrogram') -> None:
        """ 
        Draw agglomerative clustering dendrogram with timeseries graphs for all clusters.

        Parameters
        ----------
        df: dataframe with each row being the time window of readings
        labels: labels of dataset's instances
        ts_hspace: horizontal space for timeseries graph to be plotted
        title: title of dendrogram
        """

        max_cluster = len(self.linkage_matrix) + 1

        plt.figure(figsize=(12, 9))

        # define gridspec space
        gs = gridspec.GridSpec(max_cluster, max_cluster)

        # add dendrogram to gridspec
        # add -1 to give timeseries graphs more space
        plt.subplot(gs[:, 0 : max_cluster - ts_hspace - 1])
        plt.xlabel("Distance")
        plt.ylabel("Cluster")
        plt.title(title, fontsize=16, weight='bold')

        ddata = dendrogram(self.linkage_matrix, orientation="left", color_threshold=None, show_leaf_counts=True)

        self._draw_timeseries_allclust(df, labels, ddata["leaves"], gs, ts_hspace)
