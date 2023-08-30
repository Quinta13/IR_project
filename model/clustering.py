"""

Clustering
-------------------------
"""
from abc import ABC, abstractmethod
from os import path
from statistics import mean
from typing import Dict

import numpy as np
from scipy.cluster._optimal_leaf_ordering import squareform
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from io_ import log, store_json, get_sample_dir, make_dir, load_json, store_dense_matrix, load_dense_matrix
from model.dataset import DataConfig, RCV1Loader


class Clusters:
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, labels: np.ndarray, name: str):
        """
        Initialize an instance Clusters.
        
        :param mat: feature matrix.
        :param labels: clustering labels.
        """

        # Create slices depending on cluster indices

        self._name = name

        unique_labels = np.unique(labels)
        cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

        self._clusters: Dict[int, np.ndarray] = {
            label: mat[indices]
            for label, indices in cluster_indices.items()
        }

        self._medoids = np.array([])

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
    def medoids(self) -> np.ndarray:
        """
        Return medoid for each cluster

        :return: cluster medoids
        """

        if len(self._medoids) != 0:
            return self._medoids

        if not self._medoid_available:

            log(info="Computing medoids")

            self._medoids = np.stack([
                self._get_medoid(cluster_idx=i) for i in self.clusters.keys()
            ])

        else:

            log(info="Loading medoids from disk")

            self._medoids = self._load_medoids()

        return self._medoids

    def _get_medoid(self, cluster_idx: int) -> np.ndarray:
        """
        Returns the medoid of a given set of points representing a cluster.

        :param cluster_idx: cluster index.
        :return: cluster medoid.
        """

        # Extract cluster array
        arr = self[cluster_idx]

        # Compute pairwise distances
        pairwise_distances = pdist(arr)

        # Convert the pairwise distances to a square distance matrix
        distance_matrix = squareform(pairwise_distances)

        # Calculate the sum of distances for each data point
        sum_distances = np.sum(distance_matrix, axis=0)

        # Find the index of the data point with the minimum sum of distances
        medoid_index = np.argmin(sum_distances)

        # Extract the medoid
        medoid = arr[medoid_index]

        return medoid

    # SAVE / LOAD

    @property
    def _medoid_fp(self) -> str:
        """
        Dataset name

        :param name: dataset name
        :return: path to medoid fp
        """

        sample_dir = get_sample_dir(sample_name=self._name)
        return path.join(sample_dir, "medoids.npy")

    @property
    def _medoid_available(self) -> bool:
        """
        If medoids file is avaialbable
        """

        return path.exists(self._medoid_fp)

    def save_medoids(self):
        """
        Save medoids to proper directory.

        :param name: dataset name
        """

        log(info="Saving medoids ")

        sample_dir = get_sample_dir(sample_name=self._name)
        make_dir(path_=sample_dir)

        store_dense_matrix(path_=self._medoid_fp, mat=self._medoids)

    def _load_medoids(self) -> np.ndarray:

        return load_dense_matrix(path_=self._medoid_fp)




class ClusteringModel(ABC):
    """
    This abstract class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, name: str):
        """
        Initialize an instance ClusteringModel.

        """

        self._mat: np.ndarray = mat
        self._name = name

        self._labels: np.ndarray = np.array([])

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the KMeansClustering object.

        :returns: string representation of the object.
        """

        return f"Clustering[Items: {len(self)}; Labels-available: {self._labeling_available}]"

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
            labels=self.labels,
            name=self._name
        )

    # SAVE / LOAD

    @property
    def _labeling_fp(self) -> str:
        """
        Return labeling file path associated to given file name.

        :return: labeling file path
        """

        return path.join(get_sample_dir(sample_name=self._name), "labeling.json")

    @property
    def _labeling_available(self) -> bool:
        """

        :return:
        """

        return path.exists(self._labeling_fp)

    def save_labels(self):
        """
        Stores output labels locally

        :param name: name of collection
        """

        labels_int = [int(label) for label in self.labels]

        store_json(path_=self._labeling_fp, obj=labels_int)

    def _load_labels(self):
        """
        Load labels
        """

        return np.array(load_json(path_=self._labeling_fp))


class KMeansClustering(ClusteringModel):
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, name: str, k: int):
        """
        Initialize an instance of k-Means clustering.

        :param config:
        :param k: number of clusters to find.
        """

        super().__init__(mat=mat, name=name)

        self._k: int = k
        self._kmeans: KMeans = KMeans(n_clusters=k, n_init='auto')

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the KMeansClustering object.

        :returns: string representation of the object.
        """

        return f"KMeansClustering[Items: {len(self)}; k: {self._k};  Labeling-available: {self._labeling_available}]"

    # LABELS

    @property
    def labels(self) -> np.ndarray:
        """
        Return clustering labels.

        :return: clustering labels.
        """

        if len(self._labels) != 0:
            return self._labels

        if not self._labeling_available:

            log(info="Fitting K-Means model. ")
            self._kmeans.fit_predict(X=self._mat)
            self._labels = self._kmeans.labels_

        else:
            log(info="Labels already computed. Loading from disk. ")

            self._labels = self._load_labels()

        return self._labels
