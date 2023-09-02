from dataclasses import dataclass

from model.clustering import KMeansClustering
from model.d_gap import DGapComputationReassigned, DGapInference, DGapComputation
from model.rcv1 import RCV1Loader
from model.reassignment import DocIdReassignment, DocIdReassignmentComputation


@dataclass
class DataConfig:
    """
    A class for storing data configuration settings.

    Attributes:
        name (str): Name of the dataset configuration.
        docs (int): Number of documents to consider. -1 for all documents.
        terms (int): Number of terms to consider. -1 for all terms.
        eps (float): Approximation error for random projection embedding.
        n_cluster (int): Number of clusters to create when evaluating a clustering model.
    """

    # Attributes definition
    name: str
    docs: int
    terms: int
    eps: float
    n_cluster: int

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the DataConfig object.

        :return: string representation for the object.
        """

        docs = self.docs if self.docs != -1 else "all"
        terms = self.terms if self.terms != -1 else "all"

        return f"{self.name} [Docs: {docs}; Terms: {terms}; Approximation error: {self.eps}; N Clusters: {self.n_cluster}]"

    def __repr__(self) -> str:
        """
        Return the string representation of the DataConfig object.

        :return: string representation of the object.
        """
        return str(self)


class Solver:
    """
    TODO
    """

    def __init__(self, config: DataConfig):
        self._config: DataConfig = config

        self._avg_compression: float = 0.

    @property
    def avg_compression(self):
        return self._avg_compression

    def solve(self):

        print("Loading matrix... ")
        loader: RCV1Loader = RCV1Loader()
        collection = loader.load(n_docs=self._config.docs, n_terms=self._config.terms)

        print("Computing d-gap... ")
        dgap: DGapComputation = DGapComputation(collection=collection, data_name=self._config.name)
        dgap.compute_d_gaps()
        dgap.save_d_gaps()

        print("Computing embedding... ")
        embedded = collection.embed(eps=self._config.eps)

        print("Performing clustering... ")
        kmeans = KMeansClustering(mat=embedded, data_name=self._config.name, k=self._config.n_cluster)
        kmeans.fit()
        kmeans.save_labeling()
        labeling = kmeans.labeling

        print("Computing medoids... ")
        collection_clusters = kmeans.clusters
        collection_clusters.compute_medoids()
        collection_clusters.save_medoids()

        print("Computing TSP... ")
        reassignment_computation = DocIdReassignmentComputation(
            cluster=collection_clusters,
            data_name=self._config.name
        )
        reassignment_computation.solve()
        reassignment_computation.save_order()

        print("Reassigning... ")
        docs_reassignment = DocIdReassignment(
            collection=collection,
            labeling=labeling,
            medoids_order=reassignment_computation.medoids_order,
            clusters_order=reassignment_computation.cluster_order,
            data_name=self._config.name
        )

        collection_reassigned = docs_reassignment.reassign_doc_id()

        print("Computing reassigned d-gap... ")
        dgap_reassigned = DGapComputationReassigned(collection=collection_reassigned, data_name=self._config.name)
        dgap_reassigned.compute_d_gaps()
        dgap_reassigned.save_d_gaps()

        inference = DGapInference(d_gap_original=dgap, d_gap_reassigned=dgap_reassigned, data_name=self._config.name)
        inference.plot_avg_d_gap()
        self._avg_compression = inference.avg_compression
