"""
D-Gap Statistics and Inference
-------------------------------

This file contains classes for computing and inferring the goodness of reordering a document collection
based on d-gap statistics. D-gap statistics measure the gap between occurrences of terms in a document
collection, and the classes provided here help assess the effectiveness of document reordering.

Classes:
    - DGapComputation: Computes d-gap statistics for an original document collection.
    - DGapComputationReassigned: Computes d-gap statistics for a reassigned (reordered) document collection.
    - DGapInference: Evaluates the goodness of reordering based on d-gap statistics.

These classes can be used to analyze and optimize document orderings for various applications,
 such as improving search engine efficiency or document retrieval systems.
"""

import math
from os import path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from io_ import log, get_collection_dir, make_dir, store_json, load_json
from model.rcv1 import RCV1Collection


class DGapComputation:
    """
    Computes d-Gaps for a given collection of data represented as a tf-idf matrix.

    Attributes:
        _PLOT_PREFIX (str): Prefix for plot titles.
        _collection (RCV1Collection): The collection of data in tf-idf matrix format.
        _data_name (str): A name to reference a specific configuration of the dataset.
        _avg_d_gap_pterm (np.ndarray): An array storing the average d-gap per term.
        _d_gap_computed (bool): A flag indicating whether d-Gap statistics have been computed.
   """

    _PLOT_PREFIX = "Original doc-id order"

    def __init__(self, collection: RCV1Collection, data_name: str):
        """
        Initializes a DGapComputation instance.

        :param collection: The collection of data in tf-idf matrix format.
        :param data_name: A name used to refer to a specific configuration of the dataset.
                                If results were previously computed and saved, they are loaded from disk
                                instead of recomputed to save computation.
        """

        # Input attributes
        self._collection: RCV1Collection = collection
        self._data_name: data_name = data_name

        # Computation
        self._avg_d_gap_pterm: np.ndarray = np.array([])
        self._d_gap_computed: bool = False

    def __str__(self):
        """
        Return a human-readable string representation of the DGapComputation object.

        :return: string representation for the object.
        """

        return f"DGapComputation({self._data_name})[Docs: {self._collection.n_docs}; Terms: {self._collection.n_terms}; Computed: {self._d_gap_computed}]"

    def __repr__(self):
        """
        Return a string representation of the DGapComputation object.

        :return: string representation for the object.
        """

        return str(self)

    # GAP

    @staticmethod
    def d_gap_to_bitsize(gap: int) -> int:
        """
        Converts a d-gap value to its corresponding bit size.

        :param gap: The d-gap value to be converted to a bit size.
        :return: The bit size required to represent the given d-gap value.
        """

        # The initial value of the postings has gap equals to zero
        if gap == 0:
            return 0
        return math.ceil(math.log(gap) + 1 / 7) * 8  # VB representation

    def compute_d_gaps(self, try_load: bool = True):
        """
        Computes the d-gap average bit-size for each term for the given tf-idf corpus.

        The d-gap represents the average number of bits required to encode
         the gap between occurrences of each term in the documents.

        :param try_load: whether to attempt loading precomputed d_gaps from disk.
        """

        # Check if d_gaps were already computed
        if self.d_gap_computed:
            log(info="Average d-gaps already computed. Use `gaps_stats` to retrieve them. ")
            return

        # Try to load precomputed d_gaps if flag is enabled
        if try_load:
            if self._stats_precomputed:
                log(info=f"Retrieving average d-gaps from disk at {self._avg_dgap_fp}. ")
                self._load_d_gaps()
                return
            else:
                log(info=f"Information not present at {self._avg_dgap_fp}. ")

        # Need to compute gaps
        log(info="Computing average d-gap per term. ")

        # Fields indices and indptr for faster access
        indices = self._collection.data.indices
        indptr = self._collection.data.indptr
        n_terms, n_docs = self._collection.n_terms, self._collection.n_docs

        # Accumulators per term
        tot_d_gap = np.zeros(n_terms, dtype=int)  # gap accumulator per term
        tot_gaps = np.zeros(n_terms, dtype=int)  # number of gaps per term
        term_last_doc_id = np.zeros(n_terms, dtype=int)  # last doc-id seen per term, used to compute d-gap

        # Iterate through document (rows)
        for doc_id in tqdm(range(n_docs)):

            # Get the indices and data for the current document
            doc_indices = indices[indptr[doc_id]:indptr[doc_id + 1]]

            # Iterate through the terms (column) in the current document
            for term_id in doc_indices:
                # Compute the d-gap
                gap = doc_id - term_last_doc_id[term_id]
                d_gap = self.d_gap_to_bitsize(gap=gap)

                # Update information
                tot_d_gap[term_id] += d_gap
                tot_gaps[term_id] += 1

                # Update the last doc-id for the current term
                term_last_doc_id[term_id] = doc_id

        # Compute average
        self._avg_d_gap_pterm = np.around(tot_d_gap / tot_gaps, decimals=5)  # round to 3 decimals

        # Enable computed flag
        self._d_gap_computed = True

    @property
    def avg_d_gap_pterm(self) -> np.ndarray:
        """
        Returns the average d-gap per term in the tf-idf corpus.

        :return: NumPy array containing the average d-gap per term for each term in the corpus.
        """

        # Raise an exception if d_gaps were not computed
        if not self.d_gap_computed:
            raise Exception("d-gaps average were not computed yet. Use `compute_d_gaps()` first. ")

        return self._avg_d_gap_pterm

    @property
    def avg_d_gap(self) -> float:
        """.
        Returns the average d-gap value for all terms in the tf-idf corpus.

        :return: average d-gap value for all terms in the corpus.
        """

        return np.mean(self.avg_d_gap_pterm)

    @property
    def d_gap_computed(self) -> bool:
        """
        Indicates whether the d-gap statistics have been computed for the tf-idf corpus.

        :return: `True` if the d-gap statistics have been computed, `False` otherwise.
        """

        return self._d_gap_computed

    # PLOT

    def _plot(self, data: np.ndarray, title: str):
        """
        Plot the trend of data information.

        This method generates a plot to visualize the trend of gap information, typically d-gap values,
        across a dataset. It is used to visualize how d-gap values change across terms or other dimensions.

        :param data: array of data points representing the gap information to be plotted.
        :param title: title to be displayed on the plot.
        """

        # Define number of bins
        num_bins = 1000

        # Number of points per bin
        points_per_bin = len(data) // num_bins

        # Calculate bins average
        binned_data = data[:num_bins * points_per_bin].reshape(-1, points_per_bin)
        averages = np.mean(binned_data, axis=1)

        # X values
        x_vals = np.array(range(num_bins))

        # Plot averages
        plt.plot(x_vals, averages)

        # Title and labels
        plt.xlabel('term-id')
        plt.ylabel('d-gap')
        plt.title(f"{self._PLOT_PREFIX} - {title}")

        # Show
        plt.show()

    def plot_avg_d_gap(self):
        """
        Plot histogram of average d-gap per term.
        """

        self._plot(data=self._avg_d_gap_pterm, title="Average d-gap per term")

    # SAVE / LOAD

    @property
    def _avg_dgap_fp(self) -> str:
        """
        Get the file path for storing d-gap statistics.

        This property returns the file path where the computed d-gap statistics will be saved or loaded from.
        The path is based on the dataset name.

        :return: file path for storing d-gap statistics.
        """

        # File path format for d-gap statistics
        sample_dir = get_collection_dir(collection_name=self._data_name)
        return path.join(sample_dir, "avg_d_gap.json")

    @property
    def _stats_precomputed(self) -> bool:
        """
        Check if d-gap statistics information is precomputed on disk.

        :return: `True` if the d-gap statistics information is precomputed on disk, `False` otherwise.
        """

        return path.exists(path=self._avg_dgap_fp)

    def save_d_gaps(self):
        """
        Save d-gaps statistics to disk as a JSON file.
        """

        log(info=f"Saving average d-gap per term to {self._avg_dgap_fp}. ")

        # Creating directory
        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        # Create JSON
        avg_d_gap_pterm = [int(avg_d_gap) for avg_d_gap in self._avg_d_gap_pterm]

        store_json(path_=self._avg_dgap_fp, obj=avg_d_gap_pterm)

    def _load_d_gaps(self):
        """
        Load d-gaps statistics from a previously saved JSON file on disk.
        """

        # Loading
        avg_d_gap_pterm = load_json(path_=self._avg_dgap_fp)

        # NumPy array conversion
        self._avg_d_gap_pterm = np.array(avg_d_gap_pterm)

        # Computed flag enabling
        self._d_gap_computed = True


class DGapComputationReassigned(DGapComputation):
    """
    d-Gap computation for reordered doc-id according to clustering and tsp.

    This class extends the DGapComputation class to compute d-gaps for
    documents that have been reordered according to a clustering and TSP (Traveling Salesperson Problem) approach.

    Attributes:
        _PLOT_PREFIX (str): Prefix for plot titles.
        _collection (RCV1Collection): The collection of data in tf-idf matrix format.
        _data_name (str): A name to reference a specific configuration of the dataset.
        _avg_d_gap_pterm (np.ndarray): An array storing the average d-gap per term.
        _d_gap_computed (bool): A flag indicating whether d-Gap statistics have been computed.
    """

    _PLOT_PREFIX = "Reassigned doc-id order"

    def __init__(self, collection: RCV1Collection, data_name: str):
        """
        Initializes a DGapComputationReassigned instance.

        :param collection (RCV1Collection): The collection of data in tf-idf matrix format.
        :param data_name (str): A name used to refer to a specific configuration of the dataset.
                                If results were previously computed and saved, they are loaded from disk
                                instead of recomputed to save computation.
        """

        super().__init__(collection=collection, data_name=data_name)

    @property
    def _avg_dgap_fp(self) -> str:
        """
        Get the file path to the reassigned d-gap statistics information.

        :return: Path to the JSON file containing reassigned d-gap statistics.
        """

        sample_dir = get_collection_dir(collection_name=self._data_name)
        return path.join(sample_dir, "avg_d_gap_reassigned.json")


class DGapInference:
    """
    Class for evaluating the goodness of reordering based on d-gap average per term.

    Attributes:
        _d_gap_original (DGapComputation): d-gap statistics with original doc-id order.
        _d_gap_reassigned (DGapComputationReassigned): d-gap statistics with reassigned doc-ids.
    """

    # CONSTRUCTOR

    def __init__(self, d_gap_original: DGapComputation, d_gap_reassigned: DGapComputationReassigned, data_name: str):
        """
        Initialize a DGapInference instance.

        :param d_gap_original: d-gap statistics with original doc-id order.
        :param d_gap_reassigned: d-gap statistics with reassigned doc-ids.
        :param data_name: name specific of certain data configuration.
        """

        # Input attributed
        self._d_gap_original: DGapComputation = d_gap_original
        self._d_gap_reassigned: DGapComputationReassigned = d_gap_reassigned
        self._data_name = data_name

        # Compute d-gaps
        self._d_gap_original.compute_d_gaps()
        self._d_gap_reassigned.compute_d_gaps()

    # REPRESENTATION

    def __str__(self):
        """
        Return a human-readable string representation of the DGapInference object.

        :return: string representation for the object.
        """

        return f"DGapInference[Terms: {len(self._d_gap_original.avg_d_gap_pterm)}]"

    def __repr__(self):
        """
        Return a string representation of the DGapInference object.

        :return: string representation for the object.
        """

        return str(self)

    # INFERENCE

    @property
    def compression_pterm(self):
        """
        Calculate the compression percentage for each term based on d-gap improvement.

        :return: array containing the compression percentage for each term.
        """

        gaps_original = self._d_gap_original.avg_d_gap_pterm
        gaps_reassigned = self._d_gap_reassigned.avg_d_gap_pterm

        # Compute d-gap improvement for each term in percentage
        improvement = (gaps_original - gaps_reassigned + 0.001) / gaps_original * 100

        improvement[improvement < 0] = 0

        return improvement

    @property
    def avg_compression(self):
        """
        Calculate the average compression percentage across all terms.

        :return: average compression percentage.
        """

        return np.mean(self.compression_pterm)

    def plot_avg_d_gap(self):
        """
        Plot the trend of data information.
        """

        for data, label in zip(
                [self._d_gap_original.avg_d_gap_pterm, self._d_gap_reassigned.avg_d_gap_pterm],
                ["Original", "Reassigned"]
        ):

            # Define number of bins
            num_bins = 1000

            # Number of points per bin
            points_per_bin = len(data) // num_bins

            # Calculate bins average
            binned_data = data[:num_bins * points_per_bin].reshape(-1, points_per_bin)
            averages = np.mean(binned_data, axis=1)

            # X values
            x_vals = np.array(range(num_bins))

            # Plot averages
            plt.plot(x_vals, averages, label=label)

        # Title and labels
        plt.xlabel('term-id')
        plt.ylabel('d-gap')
        plt.title(f"Average d-gap comparison")
        plt.legend()

        # Save
        out = path.join(get_collection_dir(collection_name=self._data_name), "avg_d_gap.png")
        plt.savefig(out)

        # Show
        plt.show()
