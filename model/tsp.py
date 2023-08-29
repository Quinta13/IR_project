"""
Travelling Salesman Problem
---------------------------

"""
import math

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver.pywrapcp import RoutingIndexManager, RoutingModel, DefaultRoutingSearchParameters
from scipy.spatial import distance_matrix

from io_ import log


class TravellingSalesmanProblem:
    """ This class ... TODO """

    # CONSTRUCTOR

    def __init__(self, mat: np.ndarray):
        """
        Initialize classes for solving TSP problem

        :param mat: items in matrix format
        """

        # Save matrix
        self._mat: np.ndarray = mat

        # Compute distance matrix
        self._d_mat = self._get_euclidean_distance(mat=self._mat)

        # Ortools classes
        self._manager = RoutingIndexManager(len(self._d_mat), 1, 2)
        self._routing = RoutingModel(self._manager)

        transit_callback_index = self._routing.RegisterTransitCallback(self._distance_callback)
        self._routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        self._search_parameters = DefaultRoutingSearchParameters()
        self._search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        self._solutions: np.ndarray = np.array([])

    @staticmethod
    def _get_euclidean_distance(mat: np.ndarray) -> np.ndarray:
        """
        Returns euclidean distance.

        :param mat: matrix of items.
        :return: euclideian distance square matrix.
        """

        distances = []
        for from_counter, from_node in enumerate(mat):
            distances2 = []
            for to_counter, to_node in enumerate(mat):
                if from_counter == to_counter:
                    distances2.append(0)
                else:
                    # Euclidean distance
                    distances2.append(int(
                        math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
                    ))
            distances.append(distances2)
        return np.array(distances)

    def _distance_callback(self, from_index: int, to_index: int):
        """Returns the distance between the two nodes."""

        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self._manager.IndexToNode(from_index)
        to_node = self._manager.IndexToNode(to_index)
        return self._d_mat[from_node][to_node]

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return string representation for TravellingSalesmanProblem object.

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
        Return number of items involved in the TSP problem.

        :return: items for TSP problem.
        """

        return len(self._mat)

    # SOLUTION

    @property
    def solutions(self) -> np.ndarray:
        """
        Return solution

        :return: solution
        """

        if len(self._solutions) != 0:
            return self._solutions

        log(info="Evaluating solution")

        # Router solver
        solution = self._routing.SolveWithParameters(self._search_parameters)

        order = []
        index = self._routing.Start(0)

        # Perform routing
        while not self._routing.IsEnd(index):
            order.append(self._manager.IndexToNode(index))
            index = solution.Value(self._routing.NextVar(index))

        # Save solutions
        self._solutions = np.array(order)

        return self._solutions

    @property
    def sorted_items(self) -> np.ndarray:
        """
        Return sorted items.

        :return: sorted items.
        """

        return self._mat[self.solutions]
