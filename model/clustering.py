"""
Clustering
----------

This module provides classes for performing clustering analysis on data.

Classes:

    - RCV1Clusters: Class is designed to organize data into clusters and compute centroids for each cluster.
    - ClusteringModel: Abstract class represents a generic clustering model for grouping data points into clusters.
    - KMeansClustering: Class extends this functionality by implementing the k-Means clustering algorithm to
                        automatically find the optimal number of clusters and assign labels to data points.
"""

from abc import ABC, abstractmethod
from os import path
from statistics import mean
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from io_ import log, store_json, get_collection_dir, make_dir, load_json, store_dense_matrix, load_dense_matrix
from model.rcv1 import RCV1Collection


class RCV1Clusters:
    """
    This class represents clusters of data points in the RCV1 dataset.

    Attributes:
        _data_name (str): A name used to refer to a specific configuration of the dataset.
        _clusters (Dict[int, np.ndarray]): A dictionary mapping cluster labels to their corresponding data points.
        _centroids (np.ndarray): An array containing the centroid data point for each cluster.
        _centroids_computed (bool): A flag indicating whether centroids have been computed.
    """

    # CONSTRUCTOR

    def __init__(self, data: csr_matrix, labeling: np.ndarray, data_name: str):
        """
        Initialize an instance of RCV1Clusters.
        It provides some functionalities to compute cluster centroids.

        :param data: feature matrix.
        :labeling: clustering labels.
        :data_name: name used to refer to a specific configuration of the dataset.
                    If results were previously computed and saved,
                     they are loaded from disk instead of recomputed to save computation.
        """

        # Input attributes
        self._data_name: str = data_name

        # Create slices depending on cluster indices
        unique_labels = np.unique(labeling)
        cluster_indices = {label: np.where(labeling == label)[0] for label in unique_labels}

        self._clusters: Dict[int, np.ndarray] = {
            label: data[indices]
            for label, indices in cluster_indices.items()
        }

        # centroids computation
        self._centroids = np.array([])
        self._centroids_computed: bool = False

    # REPRESENTATION

    def __str__(self) -> str:
        """
         Return a human-readable string representation of the RCV1Clusters object.

        :return: string representation for the object.
        """

        return f"RCV1Clusters({self._data_name})[Data: {len(self)}, Clusters: {self.n_cluster}, " \
               f"Mean-per-Cluster: {self.mean_cardinality:.3f}; centroids computed: {self.centroids_computed}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the RCV1Clusters object.


        :return: string representation for the object.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Return total number of instances within all clusters.

        :return: total number of instances.
        """

        return sum(self.clusters_cardinalities.values())

    def __getitem__(self, cluster_idx: int) -> csr_matrix:
        """
        Return data split by cluster.

        :param cluster_idx: cluster index
        :return: cluster
        """

        return self.clusters[cluster_idx]

    # STATS

    @property
    def clusters(self) -> Dict[int, np.ndarray]:
        """
        Return data split by clusters.

        :return: data split into clusters, where keys are cluster labels, and values are data points.
        """

        return self._clusters

    @property
    def n_cluster(self) -> int:
        """
        Returns the number of clusters.

        :return: number of clusters.
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
        Return the average cluster cardinality

        :return: average cluster cardinality
        """

        return mean(self.clusters_cardinalities.values())

    # centroidS

    def compute_centroids(self, try_load: bool = True):
        """
        Compute the centroids for each cluster.

        :param try_load: whether to attempt loading precomputed centroids from disk.
        """

        # Check if stats were already computed
        if self.centroids_computed:
            log(info="centroids already computed. Use `centroids` to retrieve them. ")
            return

        # Try to load precomputed d_gaps if flag is enabled
        if try_load:
            if self._centroid_precomputed:
                log(info=f"Retrieving centroids from disk at {self._centroid_fp}. ")
                self._load_centroids()
                return
            else:
                log(info=f"Information not present at {self._centroid_fp}. ")

        # Computing centroids
        log(info="Computing centroids. ")

        # Using private helper function
        self._centroids = np.stack([
            self._get_centroid(cluster_idx=i) for i in tqdm(self.clusters.keys())
        ])

        # Enable computed flag
        self._centroids_computed = True

    @property
    def centroids(self) -> np.ndarray:
        """
        Return centroid for each cluster.

        :return: array containing the centroid data point for each cluster.
        """

        # Raise an exception if centroids were not computed
        if not self.centroids_computed:
            raise Exception("centroids were not computed yet. Use `compute_centroids()` first. ")

        return self._centroids

    def _get_centroid(self, cluster_idx: int) -> np.ndarray:
        """
        Returns the centroid of a given set of data points representing a cluster.

        :param cluster_idx: index of the cluster for which to compute the centroid.
        :return: centroid data point of the specified cluster.
        """

        cluster = self[cluster_idx]

        centroid = np.mean(cluster, axis=0)

        centroid_np = np.array(centroid)[0]

        return centroid_np

    @property
    def centroids_computed(self) -> bool:
        """
        Check if centroids have been computed.

        :return: `True` if centroids have been computed, `False` otherwise.
        """

        return self._centroids_computed

    # SAVE / LOAD

    @property
    def _centroid_fp(self) -> str:
        """
        Get the file path for storing centroids.

        This property returns the file path where the computed centroids will be saved or loaded from.
        The path is based on the dataset name.

        :return: file path for storing centroids.
        """

        sample_dir = get_collection_dir(collection_name=self._data_name)
        return path.join(sample_dir, "centroids.npy")

    @property
    def _centroid_precomputed(self) -> bool:
        """
        Check if centroids are precomputed on disk.

        :return: `True` if centroids are precomputed on disk, `False` otherwise.
        """

        return path.exists(self._centroid_fp)

    def save_centroids(self):
        """
        Save centroids to disk as a NPY file.
        """

        log(info=f"Saving centroids to {self._centroid_fp}. ")

        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        store_dense_matrix(path_=self._centroid_fp, mat=self._centroids)

    def _load_centroids(self):
        """
        Load centroids from a previously saved NPY file on disk.
        """

        self._centroids = load_dense_matrix(path_=self._centroid_fp)
        self._centroids_computed = True


class ClusteringModel(ABC):
    """
    This abstract class defines the common interface for clustering models used in RCV1 dataset analysis.

    Attributes:
        _data (numpy.ndarray): The feature matrix containing data to be clustered.
        _data_name (str): A name used to refer to a specific configuration of the dataset.
        _labeling (numpy.ndarray): An array storing cluster labels for each data point.
        _labeling_computed (bool): A flag indicating whether cluster labels have been computed.
    """

    # CONSTRUCTOR

    def __init__(self, collection: RCV1Collection, data_name: str):
        """
        Initializes an instance of the ClusteringModel.

        :param mat: The feature matrix containing data to be clustered.
        :param data_name: A name used to refer to a specific configuration of the dataset.
                          If results were previously computed and saved,
                           they are loaded from disk instead of recomputed to save computation.
        """

        # Input attributed
        self._data: csr_matrix = collection.data
        self._data_name: str = data_name

        # Labels computation
        self._labeling: np.ndarray = np.array([])
        self._labeling_computed = False

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the ClusteringModel object.

        :returns: string representation of the object.
        """

        return f"ClusteringModel({self._data_name})[Items: {len(self)}; Labeling computed: {self.labeling_computed}]"

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

        rows, _ = self._data.shape

        return rows

    # FIT

    def fit(self, try_load: bool = True):
        """
        Fits the clustering model to the provided data and computes clustering labels.

        :param try_load: Whether to attempt loading precomputed labels from disk if available.
                         If True and precomputed labels exist, they are loaded instead of recomputing.
                         Default is True.
        """

        # Check if d_gaps were already computed
        if self.labeling_computed:
            log(info="Labeling already computed. Use `labeling` to retrieve it. ")
            return

        # Try to load precomputed labeling if flag is enabled
        if try_load:
            if self._labeling_precomputed:
                log(info=f"Retrieving labeling from disk at {self._labeling_fp}. ")
                self._load_labels()
                return
            else:
                log(info=f"Information not present at {self._labeling_fp}. ")

        # Need to compute labeling
        log(info="Fitting K-Means model. ")

        self._fit()

    @abstractmethod
    def _fit(self):
        """
        Subclasses that implements this method must:
            - fit the model;
            - save labeling in the proper variable;
            - enable computed flag.
        """

        pass

    @property
    def labeling(self) -> np.ndarray:
        """
        Return the clustering labels assigned to each data point after fitting the model.

        :return: NumPy array containing the clustering labels for each data point.
        """

        # Raise an exception if d_gaps were not computed
        if not self.labeling_computed:
            raise Exception("Cluster labeling was not computed yet. Use `fit()` first. ")

        return self._labeling

    @property
    def labeling_computed(self) -> bool:
        """
        Indicates whether clustering labels have been computed and are available.

        :return: `True` if clustering labels have been computed, `False` otherwise.
        """

        return self._labeling_computed

    @property
    def clusters(self) -> RCV1Clusters:
        """
    Returns a dictionary where keys are cluster indices and values are numpy arrays representing the data points
    within each cluster.

    :return: dictionary of cluster indices mapped to numpy arrays of data points.
    """

        return RCV1Clusters(
            data=self._data,
            labeling=self.labeling,
            data_name=self._data_name
        )

    # SAVE / LOAD

    @property
    def _labeling_fp(self) -> str:
        """
        Get the file path for storing cluster labeling.

        This property returns the file path where the computed labeling will be saved or loaded from.
        The path is based on the dataset name.

        :return: file path for storing cluster labeling.
        """

        return path.join(get_collection_dir(collection_name=self._data_name), "labeling.json")

    @property
    def _labeling_precomputed(self) -> bool:
        """
        Check if cluster labeling is precomputed on disk.

        :return: `True` if cluster labeling is precomputed on disk, `False` otherwise.
        """

        return path.exists(self._labeling_fp)

    def save_labeling(self):
        """
        Save labeling to disk as a JSON file.
        """

        # Creating directory
        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        log(info=f"Saving labeling to {self._labeling_fp}. ")

        labels_int = [int(label) for label in self.labeling]

        store_json(path_=self._labeling_fp, obj=labels_int)

    def _load_labels(self):
        """
        Load labeling from a previously saved JSON file on disk.
        """

        self._labeling = np.array(load_json(path_=self._labeling_fp))
        self._labeling_computed = True


class KMeansClustering(ClusteringModel):
    """
    This class represents k-Means clustering for data.

    It uses the k-Means algorithm to partition the data into k clusters.

    Attributes:
        _data (numpy.ndarray): The feature matrix containing data to be clustered.
        _data_name (str): A name used to refer to a specific configuration of the dataset.
        _labeling (numpy.ndarray): An array storing cluster labels for each data point.
        _labeling_computed (bool): A flag indicating whether cluster labels have been computed.
        _k (int): Number of clusters to find.
    """

    # CONSTRUCTOR

    def __init__(self, collection: RCV1Collection, data_name: str, k: int):
        """
        Initializes an instance of the ClusteringModel.

        :param collection:  feature matrix containing data to be clustered.
        :param data_name: name used to refer to a specific configuration of the dataset.
                          If results were previously computed and saved,
                           they are loaded from disk instead of recomputed to save computation.
        :param k: number of clusters to find.
        """

        super().__init__(collection=collection, data_name=data_name)

        self._k: int = k

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the KMeansClustering object.

        :returns: string representation of the object.
        """

        return f"KMeansClustering({self._data_name})[Items: {len(self)}; k: {self._k}; "\
               f"Labeling computed: {self.labeling_computed}]"

    # LABELS

    def _fit(self):
        """
        Fit the k-Means clustering model to the data and compute clustering labels.
        """

        minibatch_kmeans: MiniBatchKMeans = MiniBatchKMeans(n_clusters=self._k, n_init='auto')
        minibatch_kmeans.fit(self._data)

        # Compute average
        self._labeling = minibatch_kmeans.labels_

        # Enable computed flag
        self._labeling_computed = True
