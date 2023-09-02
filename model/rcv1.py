"""
RCV1 Dataset
------------

This file contains classes related to the RCV1 dataset, including data downloading, loading, and preprocessing.

Classes:
    - RCV1Collection: Represents a collection of data with tf-idf matrices.
    - RCV1Downloader: Provides functionality to download and extract the RCV1 dataset.
    - RCV1Loader: Provides functionality to load and preprocess the RCV1 dataset.

Each class offers specific functionality for handling different aspects of the RCV1 dataset, from representing data to downloading and preprocessing it.
"""


from os import path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_rcv1
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection

from io_ import get_dataset_dir, get_data_fp, store_sparse_matrix, make_dir, load_sparse_matrix

from io_ import log


class RCV1Collection:
    """
    This class represent a collection of data (tf-idf matrix).

    Attributes:
        _data (csr_matrix): a CSR matrix representing the tf-idf data.
    """

    # CONSTRUCTOR

    def __init__(self, data: csr_matrix):
        """
        Initialize RCV1Collection instance.

        :param data: A CSR matrix representing the tf-idf data.
        """

        self._data: csr_matrix = data

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the DocumentsCollection object.

        :returns: string representation of the object.
        """

        return f"DataCollection [Docs: {self.n_docs}; Terms: {self.n_terms}; Nonzero: {self.nonzero}]"

    def __repr__(self) -> str:
        """
        Return the string representation of the DocumentsCollection object.

        :return: string representation of the object.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Return number of items (documents) in the collection.

        :return: number items within the collection.
        """

        return self.n_docs

    def __getitem__(self, doc_id: int) -> np.ndarray:
        """
        Return tf-idf vector for given item.

        :param doc_id: document identifier for a document.
        :return: document vector.
        """

        return self._data[doc_id]

    # INFORMATION

    @property
    def data(self) -> csr_matrix:
        """
        Data property.

        :return: the underlying data as a CSR matrix.
        """

        return self._data

    @property
    def n_docs(self) -> int:
        """
        Returns the number of documents in the collection.

        :return: number of documents.
        """

        docs, _ = self._data.shape
        return docs

    @property
    def n_terms(self) -> int:
        """
        Returns the number of terms in the collection.

        :return: number of terms.
        """

        _, terms = self._data.shape
        return terms

    @property
    def nonzero(self) -> int:
        """
        Returns number of non-zero elements in the data.

        :return: number of non-zero elements.
        """

        return self._data.nnz

    # DIMENSIONALITY REDUCTION

    def embed(self, eps: float) -> np.ndarray:
        """
        Embed sparse tf-idf matrix into a dense vector space using Johnson-Lindenstrauss lemma.

        :param eps: maximum distortion rate in the range (0, 1).
        :return: vector projected in lower dimensional space.
        """

        n_components = johnson_lindenstrauss_min_dim(n_samples=self._data.shape[0], eps=eps)
        transformer = SparseRandomProjection(n_components=n_components)
        reduced = transformer.fit_transform(self._data).toarray()
        return reduced


class RCV1Downloader:
    """
    This class provides an interface to download the RCV1 dataset and extract it to a designated directory.

    Attributes:
        _dataset_dir (str): path to dataset directory.
        _data_fp (str): path to data file path.
    """

    # CONSTRUCTOR

    def __init__(self):
        """
        Return a human-readable string representation of the DocumentsCollection object.

        :returns: string representation of the object.
        """

        self._dataset_dir: str = get_dataset_dir()
        self._data_fp: str = get_data_fp()

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the RCV1Downloader object.

        :returns: string representation of the object.
        """

        return f"RCV1Downloader [File: {self._data_fp}; Downloaded: {self.is_downloaded}]"

    def __repr__(self) -> str:
        """
        Return the string representation of the RCV1Downloader object.

        :return: string representation of the object.
        """

        return str(self)

    # DOWNLOAD

    @property
    def is_downloaded(self) -> bool:
        """
        Check if the RCV1Downloader dataset is already downloaded by looking at the target path.

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
    This class provides functionality to load and preprocess the RCV1 dataset.

    Attributes:
        _data_fp (str): Path to the data file path.
    """

    # CONSTRUCTOR

    def __init__(self):
        """
        Initialize the RCV1Loader instance.

        The `_data_fp` attribute is set to the path of the directory where the dataset will be stored.
        """

        self._data_fp: str = get_data_fp()
        self._data = load_sparse_matrix(path_=self._data_fp)

        if not path.exists(self._data_fp):
            raise Exception(f"File {self._data_fp} is not present. Use RCV1Downloader to download it.")

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the RCV1Loader object.

        :returns: string representation of the object.
        """

        return f"RCV1Loader [File: {self._data_fp}]"

    def __repr__(self) -> str:
        """
        Return the string representation of the RCV1Downloader object.

        :return: string representation of the object.
        """

        return str(self)

    # LOAD

    def load(self, n_docs: int = -1, n_terms: int = -1,
             sort_docs: bool = False, sort_terms: bool = True) -> RCV1Collection:
        """
        Load and preprocess the RCV1 dataset.

        :param n_docs: Number of documents to load. -1 for all documents.
        :param n_terms: Number of terms to load. -1 for all terms.
        :param sort_docs: If `True`, sort documents by the number of distinct terms.
        :param sort_terms: If `True`, sort terms by their frequency.
        :return: An instance of RCV1Collection containing the loaded data.
        """

        data = self._data.copy()

        # Reduce matrix to given number of documents and terms
        tot_docs, tot_terms = data.shape
        docs = tot_docs if n_docs == -1 else n_docs
        terms = tot_terms if n_terms == -1 else n_terms
        data = data[:docs, :terms]

        # Remove non-informative columns
        cols_zero = np.where(data.getnnz(axis=0) == 0)[0]
        mask = np.logical_not(np.isin(np.arange(data.shape[1]), cols_zero))
        data = data[:, mask]

        # Sort documents (rows) by number of distinct terms
        if sort_docs:
            sorted_row_indices = np.argsort(data.getnnz(axis=1), axis=0)[::-1]
            data = data[sorted_row_indices]

        # Sort terms (columns) by their frequency in the corpus
        if sort_terms:
            sorted_cols_indices = np.argsort(data.getnnz(axis=0), axis=0)[::-1]
            data = data[:, sorted_cols_indices]

        return RCV1Collection(data=data)
