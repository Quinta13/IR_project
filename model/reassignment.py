"""
DocID Reassignment
-----------------
This module provides classes for solving the DocID Reassignment problem,
 which involves reordering document IDs according to clustering and solving the Travelling Salesman Problem (TSP).

Classes:
    - TravellingSalesmanProblem: A class for solving the TSP problem on a given set of data points.
    - DocIdReordering: A class for reordering document IDs based on clustering and TSP solutions.
"""

import math
from os import path
from typing import Dict

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver.pywrapcp import RoutingIndexManager, RoutingModel, DefaultRoutingSearchParameters
from scipy.sparse import vstack, csr_matrix
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from io_ import get_collection_dir, log, make_dir, store_json, load_json, SAVE
from model.clustering import RCV1Clusters, KMeansClustering
from model.d_gap import DGapComputation, DGapComputationReassigned, DGapInference
from model.rcv1 import RCV1Collection, RCV1Loader, DataConfig


class TravellingSalesmanProblem:
    """
    The `TravellingSalesmanProblem` class provides methods to solve the Travelling Salesman Problem (TSP) on a given
    set of data points. TSP is a classic combinatorial optimization problem where the goal is to find the shortest
    possible route that visits a set of cities and returns to the original city. In this context, the cities represent
    data points, and the goal is to find an optimal order in which to visit them.

    Attributes:
        _data (np.ndarray): A matrix of data points where each row represents a data point, and columns represent
                            features or coordinates.
        _dist_matrix (np.ndarray): A square matrix that represents the Euclidean distance between each pair of data
                                   points. This matrix is used as input to the TSP solver.
        _manager (RoutingIndexManager): An object that manages the routing index for data points in the TSP problem.
        _routing (RoutingModel): A routing model that defines the TSP problem and constraints.
        _search_parameters (DefaultRoutingSearchParameters): TSP solver's search configuration.
        _order (np.ndarray): An array representing the optimal order in which
                             to visit the data points as per the TSP solution.
        _order_computed (bool): A flag indicating whether the TSP order has been computed.
    """

    # CONSTRUCTOR

    def __init__(self, data: np.ndarray):
        """
        Initialize a TravellingSalesmanProblem instance.

        :param data: matrix of data points where each row represents a data point,
                     and columns represent features or coordinates.
        """

        # Input attributes
        self._data: np.ndarray = data

        # Compute distance matrix
        self._dist_matrix = self._get_euclidean_distance(data=self._data)

        # Ortools classes
        self._manager = RoutingIndexManager(len(self._dist_matrix), 1, 2)
        self._routing = RoutingModel(self._manager)

        transit_callback_index = self._routing.RegisterTransitCallback(self._distance_callback)
        self._routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        self._search_parameters = DefaultRoutingSearchParameters()
        self._search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Order computation
        self._order: np.ndarray = np.array([])
        self._order_computed: bool = False

    @staticmethod
    def _get_euclidean_distance(data: np.ndarray) -> np.ndarray:
        """
        Calculate the Euclidean distance between all pairs of data points.

        :param data: matrix of data points where each row represents a data point,
                     and columns represent features or coordinates.
        :return: A square matrix where element at position representing
                 the Euclidean distance between data point i and data point j.
        """

        # https://developers.google.com/optimization/routing/tsp

        distances = []

        for from_counter, from_node in enumerate(data):
            distances2 = []
            for to_counter, to_node in enumerate(data):
                if from_counter == to_counter:
                    distances2.append(0)
                else:
                    distances2.append(int(
                        math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
                    ))
            distances.append(distances2)
        return np.array(distances)

    def _distance_callback(self, from_index: int, to_index: int):
        """
        Callback function to calculate the distance between two nodes in the Traveling Salesman Problem.

        :param from_index: index of the starting node.
        :param to_index: index of the destination node.
        :return: distance between the two nodes.
        """

        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self._manager.IndexToNode(from_index)
        to_node = self._manager.IndexToNode(to_index)

        # Retrieve the distance value from the precomputed distance matrix
        distance_ = self._dist_matrix[from_node][to_node]

        return distance_

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return human-readable string representation for TravellingSalesmanProblem object.

        :return: string representation for the object.
        """

        return f"TravellingSalesmanProblem[Items: {len(self)}]"

    def __repr__(self) -> str:
        """
        Return string representation for TravellingSalesmanProblem object.

        :return: string representation for the object.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Get the number of items involved in the Traveling Salesman Problem.

        :return: number of items for the Traveling Salesman Problem.
        """

        return len(self._data)

    # ORDERING

    def solve_tsp(self):
        """
        Solve the Traveling Salesman Problem (TSP) to find the optimal order of visiting items.

        This method uses the OR-Tools library to solve the TSP problem and computes the optimal order
        in which to visit the items, minimizing the total distance traveled.
        """

        # Return if already computed
        if self.order_computed:
            log(info="Order already computed. Use `order` to retrieve it. ")
            return

        # Router solver
        log(info="Solving TSP. ")

        solution = self._routing.SolveWithParameters(self._search_parameters)

        order = []
        index = self._routing.Start(0)

        # Perform routing
        i = 0
        while not self._routing.IsEnd(index):
            i += 1
            order.append(self._manager.IndexToNode(index))
            index = solution.Value(self._routing.NextVar(index))

        # Save solutions
        self._order = np.array(order)

        # Enable computed flag
        self._order_computed = True

    @property
    def order(self) -> np.ndarray:
        """
        Get the optimal order of visiting items as computed by solving the Traveling Salesman Problem (TSP).

        :return: optimal order of visiting items, represented as an array of item indices.
        """
        # Raise an exception if solution order was not computed
        if not self.order_computed:
            raise Exception("Order was not computed yet. Use `solve_tsp()` first. ")

        return self._order

    @property
    def order_computed(self) -> bool:
        """
        Check whether the optimal order of visiting items has been computed after solving the Traveling Salesman Problem (TSP).

        :return: `True` if the order has been computed, `False` otherwise.
        """

        return self._order_computed

    @property
    def sorted_items(self) -> np.ndarray:
        """
        Return the items sorted in the optimal order based on the solution to the Traveling Salesman Problem (TSP).

        :return: NumPy array containing the items sorted in the optimal order.
        """

        return self._data[self.solve_tsp]


class DocIdReassignment:
    """
    This class represents the process of reordering document IDs based on clustering and TSP.

    Attributes:
        _data_name (str): A name used to refer to a specific configuration of the dataset.
        _collection_clusters (RCV1Clusters): An instance of RCV1Clusters representing
                                             the clustering of the document collection.
        _centroids_order (np.ndarray): An array containing the order of centroids based on the TSP solution.
        _clusters_order (Dict[int, np.ndarray]): A dictionary where keys are cluster labels, and values are arrays
                                                 containing the order of documents within each cluster
                                                 based on the TSP solution.
        _order_computed (bool): A flag indicating whether the document ID reordering has been computed.
    """

    # CONSTRUCTOR

    def __init__(self, cluster: RCV1Clusters, data_name: str):
        """
        Initialize an instance of DocIdReassignment.

        :param cluster: clustered RCV1Clusters.
        :param data_name: name used to refer to a specific configuration of the dataset.
        """

        # Input attributes
        self._data_name: str = data_name

        # Create cluster and compute centroids
        self._collection_clusters: RCV1Clusters = cluster
        self._collection_clusters.compute_centroids()

        # Order solution
        self._centroids_order: np.ndarray = np.array([])
        self._clusters_order: Dict[int, np.ndarray] = dict()
        self._order_computed: bool = False

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return human-readable string representation for DocIdReassignment object.

        :return: string representation for the object.
        """

        return f"DocIdReassignment({self._data_name})[Items: {len(self._collection_clusters)}]"

    def __repr__(self) -> str:
        """
        Return string representation for DocIdReassignment object.

        :return: string representation for the object.
        """

        return str(self)

    # ORDER REASSIGNMENT

    def solve(self, try_load: bool = True):
        """
        Solve the document ID reordering problem based on clustering and TSP.

        :param try_load: If `True`, attempt to load precomputed solutions from disk if available.
        """

        if self.order_computed:
            log(info="Reassignment order was already computed. "
                     "Use `centroids_order` and `cluster_order` to retrieve them. ")
            return

        if try_load:
            if self._order_precomputed:
                log(info=f"Retrieving order from disk at {self._order_fp}. ")
                self._load_order()
                return
            else:
                log(info=f"Information not present at {self._order_fp}. ")

        # 1. TSP over centroids
        log(info="Solving TSP over centroids ")
        centroids = self._collection_clusters.centroids
        tsp = TravellingSalesmanProblem(data=centroids)
        tsp.solve_tsp()
        self._centroids_order = tsp.order

        # 2. Solving cluster internal order
        log(info="Solving cluster internal order")
        clusters = self._collection_clusters.clusters

        self._clusters_order = {
            label: self._centroid_order_distance(label=label)
            for label in tqdm(clusters.keys())
        }

        # Enable computed flag
        self._order_computed = True

    def _centroid_order_distance(self, label: int):
        """
        Compute the order of documents within a cluster based on their distance to the cluster centroid.

        :param label: cluster label for which the document order is computed.
        :return: array containing the order of documents within the cluster based on their distance to the centroid.
        """

        # Get centroids and cluster

        label_idx = list(self._collection_clusters.clusters.keys()).index(label)

        cluster = self._collection_clusters[label]
        centroid = csr_matrix(self._collection_clusters.centroids[label_idx])

        # Compute euclidean distances
        distances = pairwise_distances(centroid, cluster)[0]

        # Sorting by distance to the centroid
        order = np.argsort(distances)

        return order

    @property
    def order_computed(self) -> bool:
        """
        Check if the document ID reordering has been computed.

        :return: `True` if the reassignment order has been computed, `False` otherwise.
        """

        return self._order_computed

    @property
    def centroids_order(self) -> np.ndarray:
        """
        Get the order of centroids based on the TSP solution.

        :return: NumPy array containing the order of centroids.
        """

        if not self.order_computed:
            raise Exception("Reassignment order was not computed yet. Use `solve` to compute")

        return self._centroids_order

    @property
    def cluster_order(self) -> Dict[int, np.ndarray]:
        """
        Get the order of documents within each cluster based on the TSP solution.

        :return: dictionary where keys are cluster labels and values are NumPy arrays
                 containing the order of documents within each cluster.
        """

        if not self.order_computed:
            raise Exception("Reassignment order was not computed yet. Use `solve` to compute")

        return self._clusters_order

    # SAVE / LOAD

    @property
    def _order_fp(self) -> str:
        """
        Get the file path for storing reassignment order.

        This property returns the file path where the computed reassignment order will be saved or loaded from.
        The path is based on the dataset name.

        :return: file path for storing reassignment order.
        """

        sample_dir = get_collection_dir(collection_name=self._data_name)
        return path.join(sample_dir, "reassignment_order.json")

    @property
    def _order_precomputed(self) -> bool:
        """
        Check if reassignment order wad precomputed on disk.

        :return: `True` if reassignment order is precomputed on disk, `False` otherwise.
        """

        return path.exists(self._order_fp)

    def save_order(self):
        """
        Save reassignment order to disk as a JSON file.
        """

        log(info=f"Saving centroids and cluster order to {self._order_fp} ")

        # Directory creation
        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        # Conversion for JSON serialization
        centroids_sol = [int(i) for i in self.centroids_order]
        clusters_sol = {int(k): [int(i) for i in v] for k, v in self.cluster_order.items()}

        # Save
        solutions = {
            "centroids_sol": centroids_sol,
            "clusters_sol": clusters_sol
        }

        store_json(path_=self._order_fp, obj=solutions)

    def _load_order(self):
        """
        Load centroids from a previously saved JSON file on disk.
        """

        loaded = load_json(path_=self._order_fp)

        # NumPy array conversion
        self._centroids_order = np.array(loaded["centroids_sol"])
        self._clusters_order = {int(k): np.array(v) for k, v in loaded["clusters_sol"].items()}

        # Enable computed flag
        self._order_computed = True

    def reassign_doc_id(self) -> RCV1Collection:
        """
        Reassign document IDs based on the computed ordering.

        :return: new instance of RCV1Collection with reordered document IDs.
        """

        # Extract number of clusters
        n_cluster = self._collection_clusters.n_cluster

        # Sort clusters internally
        clusters_reordered = [
            self._collection_clusters[label][self._clusters_order[label]]
            for label, idx in zip(self._collection_clusters.clusters.keys(), range(n_cluster))
        ]

        # Sort clusters by centroids
        centroids_reordered = [clusters_reordered[i] for i in self._centroids_order]

        # Stack clusters in a single matrix
        data_reordered = vstack(centroids_reordered)
        return RCV1Collection(data=data_reordered)


class OneStepReassignment:
    """
    Performs all operation for Doc-Id reassignment with one-step clustering
    """

    def __init__(self):

        self._dataloader = RCV1Loader()

        self._compression: float = 0.
        self._compression_computed: bool = False

    @staticmethod
    def _permutation_reassignment(collection: RCV1Collection, config: DataConfig) -> RCV1Collection:

        log(info="  -  Evaluating clustering...")
        kmeans = KMeansClustering(collection=collection, data_name=config.name, k=config.n_cluster)
        kmeans.fit()
        if SAVE:
            kmeans.save_labeling()

        log(info="  -  Computing centroids...")
        collection_clusters = kmeans.clusters
        collection_clusters.compute_centroids()
        if SAVE:
            collection_clusters.save_centroids()

        log(info="  -  Reassign doc-id...")
        reassignment_computation = DocIdReassignment(
            cluster=collection_clusters,
            data_name=config.name
        )
        reassignment_computation.solve()
        if SAVE:
            reassignment_computation.save_order()

        return reassignment_computation.reassign_doc_id()

    def compute_compression(self, config: DataConfig):

        if self._compression_computed:
            log(info="Compression computed yet. Use `compression` to retrieve it")
            return

        collection = self._dataloader.load()

        log(info="  -  Computing d-gap...")
        dgap = DGapComputation(collection=collection, data_name=config.name)
        dgap.compute_d_gaps()
        if SAVE:
            dgap.save_d_gaps()

        # Performing reassignment
        collection_reassigned = self._permutation_reassignment(collection=collection, config=config)

        log(info="  -  Computing d-gap after reassignment...")
        dgap_reass = DGapComputationReassigned(collection=collection_reassigned, data_name=config.name)
        dgap_reass.compute_d_gaps()
        if SAVE:
            dgap_reass.save_d_gaps()

        log(info="  -  Performing inference...")
        inference = DGapInference(d_gap_original=dgap, d_gap_reassigned=dgap_reass, data_name=config.name)
        inference.plot_avg_d_gap()

        self._compression = inference.avg_compression
        self._compression_computed = True

    @property
    def compression(self) -> float:

        if not self._compression_computed:
            raise Exception("Compression not computed yet. Use `compute_compression`.")

        return self._compression


class TwoStepReassignment(OneStepReassignment):
    """
    Performs all operation for Doc-Id reassignment with one-step clustering
    """

    def __init__(self):

        super().__init__()

    def compute_compression(self, config: DataConfig):

        if self._compression_computed:
            log(info="Compression computed yet. Use `compression` to retrieve it")
            return

        collection = self._dataloader.load()

        log(info="  -  Computing d-gap...")
        dgap = DGapComputation(collection=collection, data_name=config.name)
        dgap.compute_d_gaps()
        if SAVE:
            dgap.save_d_gaps()

        log(info="  -  Computing k-means...")
        kmeans = KMeansClustering(collection=collection, data_name=config.name, k=config.n_cluster)
        kmeans.fit()
        if SAVE:
            kmeans.save_labeling()

        log(info="  -  Computing centroids...")
        collection_clusters = kmeans.clusters
        collection_clusters.compute_centroids()
        if SAVE:
            collection_clusters.save_centroids()

        log(info="  -  Solving TSP...")
        reassignment_computation = DocIdReassignment(
            cluster=collection_clusters,
            data_name=config.name
        )
        reassignment_computation.solve()
        if SAVE:
            reassignment_computation.save_order()

        log(info="  - Reordering clusters...")

        reordered_clusters  = []

        for i in range(collection_clusters.n_cluster):

            print(f"  - Evaluating cluster {i}")

            N = 100

            config_sub = DataConfig(name=f"{config.name}-c{i}", n_cluster=N)

            cluster = collection_clusters[i]

            if cluster.shape[0] >= N:
                RCV1_cluster = RCV1Collection(data=cluster)
                reordered_cluster = self._permutation_reassignment(collection=RCV1_cluster, config=config_sub).data
            else:
                centroid = csr_matrix(collection_clusters.centroids[i])
                distances = pairwise_distances(centroid, cluster)[0]
                order = np.argsort(distances)
                reordered_cluster = cluster[order]

            reordered_clusters.append(reordered_cluster)

        centroids_reordered = [reordered_clusters[i] for i in reassignment_computation.centroids_order]
        data_reordered = vstack(centroids_reordered)
        collection_reassigned = RCV1Collection(data=data_reordered)

        log(info="  -  Computing d-gap after reassignment...")
        dgap_reass = DGapComputationReassigned(collection=collection_reassigned, data_name=config.name)
        dgap_reass.compute_d_gaps()
        if SAVE:
            dgap_reass.save_d_gaps()

        log(info="  -  Performing inference...")
        inference = DGapInference(d_gap_original=dgap, d_gap_reassigned=dgap_reass, data_name=config.name)
        inference.plot_avg_d_gap()

        self._compression = inference.avg_compression
        self._compression_computed = True