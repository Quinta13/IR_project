"""

Utils
-------------------------
"""

from os import path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_rcv1
from tqdm import tqdm

from io_ import get_dataset_dir, get_data_fp, store_sparse_matrix, make_dir, load_sparse_matrix
from io_ import log


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


class DocumentsCollection:
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self, mat: csr_matrix):
        """
        Initialize DocumentsCollection instance.
        """

        self._mat: csr_matrix = mat
        self._gaps: Dict = dict()  # if empty, information need to be computed

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the DocumentsCollection object.

        :returns: string representation of the object.
        """

        return f"DocumentsCollection[Docs: {self.n_docs}; Terms: {self.n_terms}; Nonzero: {self.nonzero}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the DocumentsCollection object.

        :returns: string representation of the object.
        """

        return str(self)

    # INFORMATION'S

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
            "tot_gaps": tot_gaps,
            "avg_d_gap": tot_d_gap / tot_gaps,
            "max_d_gap": max_gap
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

        # Calculate bin averages
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


class RCV1Loader:
    """
    This class ... TODO
    """

    # CONSTRUCTOR

    def __init__(self):
        """
        Initialize the RCV1Loader instance.

        The `dataset_dir` attribute is set to the path of the directory
         where the dataset will be stored.
        """

        self._data_fp: str = get_data_fp()

        if not path.exists(self._data_fp):
            raise Exception(f"File {self._data_fp} is not present, use RCV1Downloader to download it.")

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the RCV1Downloader object.

        :returns: string representation of the object.
        """

        return f"RCV1Loader [File: {self._data_fp}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the RCV1Downloader object.

        :returns: string representation of the object.
        """

        return str(self)

    # LOAD

    def load(self, docs: int = -1, terms: int = -1,
             sort_docs: bool = False, sort_terms: bool = False) -> DocumentsCollection:
        """
        Load RCV1 dataset.

        :param docs: number of documents (rows) to load, if not given all documents are loaded.
        :param terms: number of terms (columns) to load, if not given all terms are loaded.
        :param sort_docs: if `True` all documents are sorted by decreasing number of distinct terms.
        :param sort_terms: if `True` all documents are sorted by decreasing number frequency in terms.
        :return: loaded sparse matrix
        """

        # Load matrix
        log(info="Loading matrix. ")
        mat = load_sparse_matrix(path_=self._data_fp)

        # Remove non-informative columns
        log(info="Removing non informative terms. ")
        cols_zero = np.where(mat.getnnz(axis=0) == 0)[0]
        mask = np.logical_not(np.isin(np.arange(mat.shape[1]), cols_zero))
        mat = mat[:, mask]

        # Reduce matrix to given number of documents and terms
        tot_docs, tot_terms = mat.shape
        if docs == -1:
            docs = tot_docs
        if terms == -1:
            terms = tot_terms
        mat = mat[:docs, :terms]

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

        return DocumentsCollection(mat=mat)
