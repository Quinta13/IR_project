"""
DocID_Reassignment
-----------------
"""
from os import path
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from tqdm import tqdm

from io_ import get_collection_dir, log, make_dir, store_json, load_json
from model.clustering import CollectionClusters
from model.settings import DataConfig
from model.tsp import TravellingSalesmanProblem


class DocIdReordering:

    def __init__(self, collection_clusters: CollectionClusters, data_name: str, load_order: bool = True):

        self._data_name: str = data_name
        self._collection_clusters: CollectionClusters = collection_clusters

        self._medoids_order: np.ndarray = np.array([])
        self._clusters_order: Dict[int, np.ndarray] = dict()
        self._order_computed: bool = False

        if load_order:
            if self._order_available:
                self._load_order()
            else:
                log(info="Not avaialable, need to compute")

    def solve(self):

        if self._order_computed:
            return

        # 1. TSP over medoids
        log(info="----- TSP over MEDOIDS -----")
        medois = self._collection_clusters.medoids
        tsp_medoids = TravellingSalesmanProblem(mat=medois)
        self._medoids_order = tsp_medoids.order

        # 2. TSP over internal clusters
        log(info="----- TSP over CLUSTERS -----")
        clusters = self._collection_clusters.clusters

        self._clusters_order = {
            label: self._medoid_order_distance(label=label)
            for label, mat in tqdm(clusters.items())
        }

        self._order_computed = True

    def _medoid_order_distance(self, label):

        cluster = self._collection_clusters[label]
        medoid = self._collection_clusters.medoids[label]

        distances = np.array([distance.euclidean(point, medoid) for point in cluster])

        order = np.argsort(distances)

        return order

    @property
    def medoids_sol(self) -> np.ndarray:
        return self._medoids_order

    @property
    def clusters_sol(self) -> Dict[int, np.ndarray]:
        return self._clusters_order

    # SAVE / LOAD

    @property
    def _order_fp(self) -> str:

        sample_dir = get_collection_dir(collection_name=self._data_name)
        return path.join(sample_dir, "order.json")

    @property
    def _order_available(self) -> bool:

        return path.exists(self._order_fp)

    def save_order(self):
        """
        Save order to proper directory.
        """

        log(info="Saving order ")

        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        medoids_sol = [int(i) for i in self._medoids_order]
        clusters_sol = {int(k): [int(i) for i in v] for k, v in self._clusters_order.items()}

        solutions = {
            "medoids_sol": medoids_sol,
            "clusters_sol": clusters_sol
        }

        store_json(path_=self._order_fp, obj=solutions)

    def _load_order(self):

        loaded = load_json(path_=self._order_fp)

        self._medoids_order = np.array(loaded["medoids_sol"])
        self._clusters_order = {int(k): np.array(v) for k, v in loaded["clusters_sol"].items()}
        self._order_computed = True


class DocIdReassignment:

    def __init__(self, cluster: CollectionClusters, medoid_order: np.ndarray, clusters_order: Dict[int, np.ndarray]):

        self._cluster = cluster
        self._medoid_order = medoid_order
        self._clusters_order = clusters_order

    def reordered(self) -> np.ndarray:

        n_cluster = self._cluster.n_cluster

        clusters_reordered = [
            self._cluster[i][self._clusters_order[i]]
            for i in range(n_cluster)
        ]

        medoids_reordered = clusters_reordered[self._medoid_order]

        return medoids_reordered



