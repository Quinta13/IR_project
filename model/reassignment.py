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
from scipy.sparse import vstack
from scipy.spatial import distance
from tqdm import tqdm

from io_ import get_collection_dir, log, make_dir, store_json, load_json
from model.clustering import RCV1Clusters
from model.rcv1 import RCV1Collection


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


class DocIdReassignmentComputation:
    """
    This class represents the process of reordering document IDs based on clustering and TSP.

    Attributes:
        _data_name (str): A name used to refer to a specific configuration of the dataset.
        _collection_clusters (RCV1Clusters): An instance of RCV1Clusters representing
                                             the clustering of the document collection.
        _medoids_order (np.ndarray): An array containing the order of medoids based on the TSP solution.
        _clusters_order (Dict[int, np.ndarray]): A dictionary where keys are cluster labels, and values are arrays
                                                 containing the order of documents within each cluster
                                                 based on the TSP solution.
        _order_computed (bool): A flag indicating whether the document ID reordering has been computed.
    """

    # CONSTRUCTOR

    def __init__(self, cluster: RCV1Clusters, data_name: str):
        """
        Initialize an instance of DocIdReassignmentComputation.

        :param cluster: clustered RCV1Clusters.
        :param data_name: name used to refer to a specific configuration of the dataset.
        """

        # Input attributes
        self._data_name: str = data_name

        # Create cluster and compute medoids
        self._collection_clusters: RCV1Clusters = cluster
        self._collection_clusters.compute_medoids()

        # Order solution
        self._medoids_order: np.ndarray = np.array([])
        self._clusters_order: Dict[int, np.ndarray] = dict()
        self._order_computed: bool = False

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return human-readable string representation for DocIdReassignmentComputation object.

        :return: string representation for the object.
        """

        return f"DocIdReassignmentComputation({self._data_name})[]"

    def __repr__(self) -> str:
        """
        Return string representation for DocIdReassignmentComputation object.

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
                     "Use `medoids_order` and `cluster_order` to retrieve them. ")
            return

        if try_load:
            if self._order_precomputed:
                log(info=f"Retrieving order from disk at {self._order_fp}. ")
                self._load_order()
                return
            else:
                log(info=f"Information not present at {self._order_fp}. ")

        # 1. TSP over medoids
        log(info="Solving TSP over medoids ")
        medoids = self._collection_clusters.medoids
        tsp = TravellingSalesmanProblem(data=medoids)
        tsp.solve_tsp()
        self._medoids_order = tsp.order

        # 2. Solving cluster internal order
        log(info="Solving cluster internal order")
        clusters = self._collection_clusters.clusters

        self._clusters_order = {
            label: self._medoid_order_distance(label=label)
            for label, mat in tqdm(clusters.items())
        }

        # Enable computed flag
        self._order_computed = True

    def _medoid_order_distance(self, label: int):
        """
        Compute the order of documents within a cluster based on their distance to the cluster medoid.

        :param label: cluster label for which the document order is computed.
        :return: array containing the order of documents within the cluster based on their distance to the medoid.
        """

        # Get medoids and cluster
        cluster = self._collection_clusters[label]
        medoid = self._collection_clusters.medoids[label]

        # Compute euclidean distances
        distances = np.array([distance.euclidean(point, medoid) for point in cluster])

        # Sorting by distance to the medoid
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
    def medoids_order(self) -> np.ndarray:
        """
        Get the order of medoids based on the TSP solution.

        :return: NumPy array containing the order of medoids.
        """

        if not self.order_computed:
            raise Exception("Reassignment order was not computed yet. Use `solve` to compute")

        return self._medoids_order

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

        log(info=f"Saving medoids and cluster order to {self._order_fp} ")

        # Directory creation
        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        # Conversion for JSON serialization
        medoids_sol = [int(i) for i in self.medoids_order]
        clusters_sol = {int(k): [int(i) for i in v] for k, v in self.cluster_order.items()}

        # Save
        solutions = {
            "medoids_sol": medoids_sol,
            "clusters_sol": clusters_sol
        }

        store_json(path_=self._order_fp, obj=solutions)

    def _load_order(self):
        """
        Load medoids from a previously saved JSON file on disk.
        """

        loaded = load_json(path_=self._order_fp)

        # NumPy array conversion
        self._medoids_order = np.array(loaded["medoids_sol"])
        self._clusters_order = {int(k): np.array(v) for k, v in loaded["clusters_sol"].items()}

        # Enable computed flag
        self._order_computed = True


class DocIdReassignment:
    """
    This class represents the reassignment of document IDs based on clustering and ordering.

    Attributes:
        _collection_cluster (RCV1Clusters): An instance of RCV1Clusters representing
                                            the clustered collection of documents.
        _medoids_order (np.ndarray): An ordered array of cluster indices representing the order
                                     in which medoids should be visited.
        _clusters_order (Dict[int, np.ndarray]): A dictionary where keys are cluster indices and values are
                                                 ordered arrays representing the order in which documents within
                                                 each cluster should be visited.
    """

    # CONSTRUCTOR

    def __init__(self, collection: RCV1Collection, labeling: np.ndarray,
                 medoids_order: np.ndarray, clusters_order: Dict[int, np.ndarray], data_name: str):
        """
        Initialize an instance of DocIdReassignment.

        :param collection: instance of RCV1Collection representing the collection of documents.
        :param labeling: array representing the cluster labels assigned to each document.
        :param medoids_order: ordered array of cluster indices representing the order in which medoids should be visited.
        :param clusters_order: dictionary where keys are cluster indices and values are ordered arrays
            representing the order in which documents within each cluster should be visited.
        :param data_name: name used to refer to a specific configuration of the dataset.
        """

        self._collection_cluster = RCV1Clusters(data=collection.data, labeling=labeling, data_name=data_name)
        self._medoids_order: np.ndarray = medoids_order
        self._clusters_order: Dict[int, np.ndarray] = clusters_order

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return human-readable string representation for DocIdReassignment object.

        :return: string representation for the object.
        """

        return f"DocIdReassignment[Items: {len(self._collection_cluster)}]"

    def __repr__(self) -> str:
        """
        Return string representation for DocIdReassignment object.

        :return: string representation for the object.
        """

        return str(self)

    # REASSIGNMENT

    def reassign_doc_id(self) -> RCV1Collection:
        """
        Reassign document IDs based on the computed ordering.

        :return: new instance of RCV1Collection with reordered document IDs.
        """

        # Extract number of clusters
        n_cluster = self._collection_cluster.n_cluster

        # Sort clusters internally
        clusters_reordered = [
            self._collection_cluster[i][self._clusters_order[i]]
            for i in range(n_cluster)
        ]

        # Sort clusters by medoids
        medoids_reordered = [clusters_reordered[i] for i in self._medoids_order]

        # Stack clusters in a single matrix
        data_reordered = vstack(medoids_reordered)
        return RCV1Collection(data=data_reordered)
