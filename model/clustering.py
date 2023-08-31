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

from io_ import log, store_json, get_collection_dir, make_dir, load_json, store_dense_matrix, load_dense_matrix
from model.settings import DataConfig


class CollectionClusters:
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, labels: np.ndarray, data_name: str, load_medoids: bool = True):
        """
        Initialize an instance Clusters.
        
        :param mat: feature matrix.
        :param labels: clustering labels.
        """

        self._data_name = data_name

        # Create slices depending on cluster indices
        unique_labels = np.unique(labels)
        cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

        self._clusters: Dict[int, np.ndarray] = {
            label: mat[indices]
            for label, indices in cluster_indices.items()
        }

        self._medoids = np.array([])
        self._medoids_computed: bool = False

        if load_medoids:
            log(info="Try to retrieve medoids info from disk")
            if self._medoid_available:
                self._load_medoids()
            else:
                log(info="Medoids not available, need to be computed")

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return string representation for Clusters object.

        :return: string representation for the object.
        """

        return f"ClusterDataSplit [Data: {len(self)}, Clusters: {self.n_cluster}, " \
               f"Mean-per-Cluster: {self.mean_cardinality:.3f}; Medoids computed: {self._medoids_computed}]"

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

        return {k: v.shape[0] for k, v in self.clusters.items()}

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

        if self._medoids_computed:
            return self._medoids

        log(info="Computing medoids")

        self._medoids = np.stack([
            self._get_medoid(cluster_idx=i) for i in self.clusters.keys()
        ])

        self._medoids_computed = True

        return self._medoids

    def _get_medoid(self, cluster_idx: int) -> np.ndarray:
        """
        Returns the medoid of a given set of points representing a cluster.

        :param cluster_idx: cluster index.
        :return: cluster medoid.
        """

        # Extract cluster array
        arr = self[cluster_idx]

        """# Compute pairwise distances
        pairwise_distances = pdist(arr)

        # Convert the pairwise distances to a square distance matrix
        distance_matrix = squareform(pairwise_distances)

        # Calculate the sum of distances for each data point
        sum_distances = np.sum(distance_matrix, axis=0)

        # Find the index of the data point with the minimum sum of distances
        medoid_index = np.argmin(sum_distances)

        # Extract the medoid
        medoid = arr[medoid_index]
        """

        # centroid
        arr = self[cluster_idx]

        return arr.mean(axis=0)

    # SAVE / LOAD

    @property
    def _medoid_fp(self) -> str:
        """
        Dataset name

        :param name: dataset name
        :return: path to medoid fp
        """

        sample_dir = get_collection_dir(collection_name=self._data_name)
        return path.join(sample_dir, "medoids.npy")

    @property
    def _medoid_available(self) -> bool:
        """
        If medoids file is available
        """

        return path.exists(self._medoid_fp)

    def save_medoids(self):
        """
        Save medoids to proper directory.

        :param name: dataset name
        """

        log(info="Saving medoids ")

        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        store_dense_matrix(path_=self._medoid_fp, mat=self._medoids)

    def _load_medoids(self):

        self._medoids = load_dense_matrix(path_=self._medoid_fp)
        self._medoids_computed = True


class ClusteringModel(ABC):
    """
    This abstract class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, data_name: str, load_labels: bool = True):
        """
        Initialize an instance ClusteringModel.

        """

        self._mat: np.ndarray = mat
        self._data_name = data_name

        self._labels: np.ndarray = np.array([])
        self._labels_computed = False

        if load_labels:
            log(info="Try to retrieve labels info from disk")
            if self._labeling_available:
                self._load_labels()
            else:
                log(info="Labeling is not available, need to be computed")

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the Clustering object.

        :returns: string representation of the object.
        """

        return f"Clustering[Items: {len(self)}; Labeling computed: {self._labels_computed}]"

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
    def clusters(self) -> CollectionClusters:
        """
        Return clusters found by the model.

        :return: clusters.
        """

        return CollectionClusters(
            mat=self._mat,
            labels=self.labels,
            data_name=self._data_name
        )

    # SAVE / LOAD

    @property
    def _labeling_fp(self) -> str:
        """
        Return labeling file path associated to given file name.

        :return: labeling file path
        """

        return path.join(get_collection_dir(collection_name=self._data_name), "labeling.json")

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

        self._labels = np.array(load_json(path_=self._labeling_fp))
        self._labels_computed = True


class KMeansClustering(ClusteringModel):
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray, data_name: str, k: int, load_labels: bool = True):
        """
        Initialize an instance of k-Means clustering.

        :param config:
        :param k: number of clusters to find.
        """

        super().__init__(mat=mat, data_name=data_name, load_labels=load_labels)

        self._k: int = k

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the KMeansClustering object.

        :returns: string representation of the object.
        """

        return f"KMeansClustering[Items: {len(self)}; k: {self._k};  Labeling computed: {self._labels_computed}]"

    # LABELS

    @property
    def labels(self) -> np.ndarray:
        """
        Return clustering labels.

        :return: clustering labels.
        """

        if self._labels_computed:
            return self._labels

        log(info="Fitting K-Means model. ")

        kmeans: KMeans = KMeans(n_clusters=self._k, n_init='auto')
        kmeans.fit_predict(X=self._mat)

        self._labels = kmeans.labels_
        self._labels_computed = True

        return self._labels
