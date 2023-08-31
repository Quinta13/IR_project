from dataclasses import dataclass


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

        :return: string representation for DataConfig object.
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
