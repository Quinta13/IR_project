"""

Clustering
-------------------------
"""
from abc import ABC, abstractmethod
from statistics import mean
from typing import Dict

import numpy as np
from scipy.cluster._optimal_leaf_ordering import squareform
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from io_ import log


class Clusters:
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, labels: np.ndarray):
        """
        Initialize an instance Clusters.
        
        :param mat: feature matrix.
        :param labels: clustering labels.
        """

        # Create slices depending on cluster indices

        unique_labels = np.unique(labels)
        cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

        self._clusters: Dict[int, np.ndarray] = {
            label: mat[indices]
            for label, indices in cluster_indices.items()
        }

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return string representation for Clusters object.

        :return: string representation for the object.
        """

        return f"ClusterDataSplit [Data: {len(self)}, Clusters: {self.n_cluster}, " \
               f"Mean-per-Cluster: {self.mean_cardinality:.3f}]"

    def __repr__(self) -> str:
        """
        Return string representation for Clusters object.

        :return: string representation for the object.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Return total number of instances within all clusters.

        :return: total number of instances.
        """

        return sum(self.clusters_cardinalities.values())

    def __getitem__(self, cluster_idx: int) -> np.ndarray:
        """
        Get cluster corresponding to given index

        :param cluster_idx: cluster index
        :return: cluster
        """

        return self.clusters[cluster_idx]

    # STATS

    @property
    def clusters(self) -> Dict[int, np.ndarray]:
        """
        Return data split by clusters.

        :return: data split in cluster.
        """

        return self._clusters

    @property
    def n_cluster(self) -> int:
        """
        Returns the number of clusters found

        :return: number of clusters
        """

        return len(self.clusters)

    @property
    def clusters_cardinalities(self) -> Dict[int, int]:
        """
        Return the number of items for each cluster.

        :return: number of items for each cluster.
        """

        return {k: len(v) for k, v in self.clusters.items()}

    @property
    def mean_cardinality(self) -> float:
        """
        Return average cluster cardinality

        :return: average cluster cardinality
        """

        return mean(self.clusters_cardinalities.values())

    # MEDOIDS

    @property
    def medoids(self) -> Dict[int, np.ndarray]:
        """
        Return medoid for each cluster

        :return: cluster medoids
        """

        return {
            i: self._get_medoid(cluster_idx=i)
            for i in self.clusters.keys()
        }

    def _get_medoid(self, cluster_idx: int) -> np.ndarray:
        """
        Returns the medoid of a given set of points representing a cluster.

        :param cluster_idx: cluster index.
        :return: cluster medoid.
        """

        arr = self[cluster_idx]

        pairwise_distances = pdist(arr)

        # Convert the pairwise distances to a square distance matrix
        distance_matrix = squareform(pairwise_distances)

        # Calculate the sum of distances for each data point
        sum_distances = np.sum(distance_matrix, axis=0)

        # Find the index of the data point with the minimum sum of distances
        medoid_index = np.argmin(sum_distances)

        # The medoid is the data point at the medoid_index
        medoid = arr[medoid_index]

        return medoid

class ClusteringModel(ABC):
    """
    This abstract class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray):
        """
        Initialize an instance ClusteringModel.

        :param mat: feature matrix.
        """

        self._mat: np.ndarray = mat
        self._fitted: bool = False

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the KMeansClustering object.

        :returns: string representation of the object.
        """

        return f"Clustering[Items: {len(self)}; Fitted: {self._fitted}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the DocumentsCollection object.

        :returns: string representation of the object.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Return number of items (documents) in the collection.

        :return: items within the collection.
        """

        rows, _ = self._mat.shape

        return rows

    # FIT

    @abstractmethod
    def fit(self):
        """
        Fit the model
        """

        pass

    @property
    @abstractmethod
    def labels(self) -> np.ndarray:
        """
        Return clustering labels.

        :return: clustering labels.
        """

        pass

    @property
    def clusters(self) -> Clusters:
        """
        Return clusters found by the model.

        :return: clusters.
        """

        return Clusters(
            mat=self._mat,
            labels=self.labels
        )


class KMeansClustering(ClusteringModel):
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, k: int):
        """
        Initialize an instance of k-Means clustering.

        :param mat: feature matrix.
        :param k: number of clusters to find.
        """

        super().__init__(mat)

        self._k: int = k
        self._kmeans: KMeans = KMeans(n_clusters=k, n_init='auto')

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the KMeansClustering object.

        :returns: string representation of the object.
        """

        return f"KMeansClustering[Items: {len(self)}; k: {self._k};  Fitted: {self._fitted}]"

    # FIT

    def fit(self):
        """
        Fit the model
        """

        # Check if model was fitted
        if self._fitted:
            log(info="Model was already fitted. ")
            return

        # Fit the model
        log(info="Fitting K-Means model. ")
        self._kmeans.fit_predict(X=self._mat)
        self._fitted = True

    @property
    def labels(self) -> np.ndarray:
        """
        Return clustering labels.

        :return: clustering labels.
        """

        if not self._fitted:
            raise Exception("Model not fitted yet")

        return self._kmeans.labels_
