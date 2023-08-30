"""

Utils
-------------------------
"""

from dataclasses import dataclass
from os import path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_rcv1
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from tqdm import tqdm

from io_ import get_dataset_dir, get_data_fp, store_sparse_matrix, make_dir, load_sparse_matrix, get_sample_dir, \
    store_json, load_json
from io_ import log


@dataclass
class DataConfig:
    """
    This class contains settings for a certain configuration, in particular
     - `name` is the dataset name for the specific configuration; a specific directory will be dedicated to store
              files associated with this specific configuration.
    - `docs` it specifies to use only the first `docs` (rows) of the original collection.
    - `terms` it specifies to use only the first `terms` (columns) of the original collection.
    - `eps` it specifics the approximation error to use when projecting to a new vector space.
    """
    name: str
    docs: int
    terms: int
    eps: float

    def __str__(self) -> str:

        docs = self.docs if self.docs != -1 else "all"
        terms = self.terms if self.terms != -1 else "all"

        return f"{self.name} [Docs: {docs}; Terms: {terms}; Approximation error: {self.eps}]"

    def __repr__(self) -> str:

        return str(self)


class DataCollection:
    """
    This class represent a collection of data (tf-idf matrix) and allows operation like compute the gap
    """

    # CONSTRUCTOR

    def __init__(self, config: DataConfig, mat: csr_matrix):
        """
        Initialize DocumentsCollection instance.
        """

        self._mat: csr_matrix = mat
        self._config = config

        self._gaps: Dict = dict()  # if empty, information need to be computed

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the DocumentsCollection object.

        :returns: string representation of the object.
        """

        return f"DataCollection({self._config.name})[Docs: {self.n_docs}; Terms: {self.n_terms}; Nonzero: {self.nonzero}; Info avaialable : {self._info_available}]"

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

        return self.n_docs

    def __getitem__(self, doc_id: int) -> np.ndarray:
        """
        Return tf-idf for given item.

        :param doc_id: doc-id identifier for a document.
        :return: document vector.
        """

        return self._mat[doc_id]

    # INFORMATION'S

    @property
    def data(self) -> csr_matrix:
        """
        Data

        :return: data.
        """

        return self._mat

    @property
    def n_docs(self) -> int:
        """
        Returns number of documents in the collection.

        :return: number of documents.
        """

        docs, _ = self._mat.shape

        return docs

    @property
    def n_terms(self) -> int:
        """
        Returns number of terms in the collection.

        :return: number of terms.
        """

        _, terms = self._mat.shape

        return terms

    @property
    def nonzero(self) -> int:
        """
        Returns number of non-zero elements

        :return: number of non-zero.
        """

        return self._mat.nnz

    # GAP

    @property
    def gaps(self) -> Dict[str, np.ndarray[int | float]]:
        """
        It computes total gap of the postings derived from the matrix.

        :return: total d-gap.
        """

        # Return if result is computed yet
        if self._gaps:
            return self._gaps

        if not self._info_available:

            # Need to compute gaps
            log(info="Computing gaps per term")

            # Fields indices and indptr for faster access
            indices = self._mat.indices
            indptr = self._mat.indptr

            # Accumulators per term
            tot_d_gap = np.zeros(self.n_terms, dtype=int)  # gap accumulator per term
            tot_gaps = np.zeros(self.n_terms, dtype=int)  # number of gaps per term
            max_gap = np.zeros(self.n_terms, dtype=int)  # maximum gap per term
            term_last_doc_id = np.zeros(self.n_terms, dtype=int)  # last doc-id seen per term, used to compute d-gap

            # Iterate through document (rows)
            for doc_id in tqdm(range(self.n_docs)):

                # Get the indices and data for the current document
                doc_indices = indices[indptr[doc_id]:indptr[doc_id + 1]]

                # Iterate through the terms in the current document
                for term_id in doc_indices:
                    # Compute the d-gap
                    d_gap = doc_id - term_last_doc_id[term_id]

                    # Update information
                    tot_d_gap[term_id] += d_gap
                    tot_gaps[term_id] += 1
                    max_gap[term_id] = max(max_gap[term_id], d_gap)

                    # Update the last doc-id for the current term
                    term_last_doc_id[term_id] = doc_id

            self._gaps = {
                "tot_d_gap": tot_d_gap,
                "avg_d_gap": np.around(tot_d_gap / tot_gaps, decimals=3),  # round to 3 decimals
                "max_d_gap": max_gap
            }

        else:

            log(info=f"Information was saved locally at {self._info_fp}. Loading from disk. ")
            info = self._load_info()

            self._gaps = {
                "tot_d_gap": info["tot_d_gap_pterm"],
                "avg_d_gap": info["avg_d_gap_pterm"],
                "max_d_gap": info["max_d_gap_pterm"],
            }

        return self._gaps

    @property
    def tot_d_gap(self) -> int:
        """
        Return total d-gap over all terms.

        :return: total d-gap.
        """

        return sum(self.gaps["tot_d_gap"])

    @property
    def avg_d_gap(self) -> int:
        """
        Return average d-gap over all terms.

        :return: average d-gap.
        """

        return np.mean(self.gaps["avg_d_gap"])

    @property
    def max_d_gap(self) -> int:
        """
        Return maximum d-gap over all terms.

        :return: maximum d-gap.
        """

        return np.max(self.gaps["max_d_gap"])

    # PLOT

    def _plot(self, key: str, title: str):
        """
        Plot gap information trend.

        :param key: key for gaps dictionary.
        :param title: plot title name.
        """

        # Get data
        data = self.gaps[key]

        # Define number of bins
        num_bins = 900

        # Number of points per bin
        points_per_bin = len(data) // num_bins

        # Calculate bin averagesshift
        binned_data = data[:num_bins * points_per_bin].reshape(-1, points_per_bin)
        averages = np.mean(binned_data, axis=1)

        # X values
        x_vals = np.array(range(num_bins))

        # Plot the averages
        plt.plot(x_vals, averages)

        # Add labels and a title
        plt.xlabel('term-id')
        plt.ylabel('d-gap')
        plt.title(title)

        # Show the plot
        plt.show()

    def plot_tot_d_gap(self):
        """
        Plot histogram of total d-gap per term.
        """

        self._plot(key="tot_d_gap", title="Total d-gap per term")

    def plot_avg_d_gap(self):
        """
        Plot histogram of average d-gap per term.
        """

        self._plot(key="avg_d_gap", title="Average d-gap per term")

    def plot_max_d_gap(self):
        """
        Plot histogram of maximum d-gap per term.
        """

        self._plot(key="max_d_gap", title="Maximum d-gap per term")

    # DIMENSIONALITY REDUCTION

    def embed(self) -> np.ndarray:
        """
        Embed sparse tf-idf matrix into a dense vector space using Johnson-Lindenstrauss lemma

        :return: vector projected in lower dimensional space.
        """

        # Compute target number of components
        n_components = johnson_lindenstrauss_min_dim(n_samples=self._mat.shape[0], eps=self._config.eps)

        # Create a Sparse Random Projection object
        transformer = SparseRandomProjection(n_components=n_components)

        # Fit and transform the original array
        reduced = transformer.fit_transform(self._mat).toarray()

        return reduced

    # SAVE / LOAD

    @property
    def _info_fp(self) -> str:
        """
        :return: path to info file
        """

        sample_dir = get_sample_dir(sample_name=self._config.name)
        return path.join(sample_dir, "info.json")

    @property
    def _info_available(self) -> bool:
        """
        Check if information is already available on disk
        """

        return path.exists(path=self._info_fp)

    def save_info(self, name: str):
        """
        Save information to proper directory.

        :param name: dataset name
        """

        log(info="Saving collection information ")

        sample_dir = get_sample_dir(sample_name=name)
        make_dir(path_=sample_dir)

        info_fp = self._info_fp

        # Create JSON
        info: Dict[str, int | float] = {
            "n_docs": int(self.n_docs),
            "n_terms": int(self.n_terms),
            "tot_d_gap": int(self.tot_d_gap),
            "avg_d_gap": float(self.avg_d_gap),
            "max_d_gap": int(self.max_d_gap),
            "tot_d_gap_pterm": [int(d_gap) for d_gap in self.gaps["tot_d_gap"]],
            "avg_d_gap_pterm": [float(d_gap) for d_gap in self.gaps["avg_d_gap"]],
            "max_d_gap_pterm": [int(d_gap) for d_gap in self.gaps["max_d_gap"]]
        }

        store_json(path_=info_fp, obj=info)

    def _load_info(self) -> Dict[str, int | float]:
        """

        """

        info = load_json(path_=self._info_fp)

        for key in ["tot_d_gap_pterm", "avg_d_gap_pterm", "max_d_gap_pterm"]:
            info[key] = np.array(info[key])

        return info


class RCV1Downloader:
    """
    This class provides an interface to download the RCV1 dataset
     and extract it to a designated directory.

    Attributes:
     - dataset_dir str: path to dataset directory.
     - data_fp str: path to data file path.
    """

    # CONSTRUCTOR

    def __init__(self):
        """
        Initialize the RCV1Downloader instance.

        The `dataset_dir` attribute is set to the path of the directory
         where the dataset will be stored.
        """

        self._dataset_dir: str = get_dataset_dir()
        self._data_fp: str = get_data_fp()

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the RCV1Downloader object.

        :returns: string representation of the object.
        """

        return f"RCV1Downloader [File: {self._data_fp}; Downloaded: {self.is_downloaded}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the RCV1Downloader object.

        :returns: string representation of the object.
        """

        return str(self)

    # DOWNLOAD

    @property
    def is_downloaded(self) -> bool:
        """
        Check if the RCV1Downloader dataset is already downloaded.

        :return: True if the dataset is downloaded, False otherwise.
        """

        return path.exists(self._data_fp)

    def download(self):
        """
        Download and extract the RCV1Downloader dataset if not already downloaded.
        """

        # Check if already downloaded
        if self.is_downloaded:
            log(info=f"Dataset is already downloaded at {self._data_fp}")
            return

        # Create directory
        make_dir(path_=self._dataset_dir)

        # Fetch dataset
        log(info="Fetching dataset. ")
        rcv1 = fetch_rcv1()
        data = rcv1["data"]

        # Save file
        log(info="Saving dataset. ")
        store_sparse_matrix(path_=self._data_fp, mat=data)


class RCV1Loader:
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, config: DataConfig):
        """
        Initialize the RCV1Loader instance.

        The `dataset_dir` attribute is set to the path of the directory
         where the dataset will be stored.
        """

        self._data_fp: str = get_data_fp()
        self._config: DataConfig = config

        if not path.exists(self._data_fp):
            raise Exception(f"File {self._data_fp} is not present, use RCV1Downloader to download it.")

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the RCV1Downloader object.

        :returns: string representation of the object.
        """

        return f"RCV1Loader({self._config.name}) [File: {self._data_fp}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the RCV1Downloader object.

        :returns: string representation of the object.
        """

        return str(self)

    # LOAD

    def load(self, sort_docs: bool = False, sort_terms: bool = False) -> DataCollection:
        """
        Load RCV1 dataset.

        :param sort_docs: if `True` all documents are sorted by decreasing number of distinct terms.
        :param sort_terms: if `True` all documents are sorted by decreasing number frequency in terms.
        :return: loaded sparse matrix
        """

        # Load matrix
        log(info="Loading matrix. ")
        mat = load_sparse_matrix(path_=self._data_fp)

        # Reduce matrix to given number of documents and terms
        tot_docs, tot_terms= mat.shape

        docs = tot_docs if self._config.docs == -1 else self._config.docs
        terms = tot_terms if self._config.terms == -1 else self._config.terms

        mat = mat[:docs, :terms]

        # Remove non-informative columns
        log(info="Removing non informative terms. ")
        cols_zero = np.where(mat.getnnz(axis=0) == 0)[0]
        mask = np.logical_not(np.isin(np.arange(mat.shape[1]), cols_zero))
        mat = mat[:, mask]

        # Sort documents (rows) by number of distinct terms
        if sort_docs:
            log(info="Sorting documents by terms count. ")
            sorted_row_indices = np.argsort(mat.getnnz(axis=1), axis=0)[::-1]
            mat = mat[sorted_row_indices]

        # Sort terms (columns) by their frequency in the corpus
        if sort_terms:
            log(info="Sorting terms by their frequency. ")
            sorted_cols_indices = np.argsort(mat.getnnz(axis=0), axis=0)[::-1]
            mat = mat[:, sorted_cols_indices]

        return DataCollection(mat=mat, config=self._config)
