"""
d-Gaps
--------------------

"""

import math
from os import path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from io_ import log, get_collection_dir, make_dir, store_json, load_json
from model.dataset import RCV1Collection


class DGapComputation:
    """
    TODO
    """

    def __init__(self, collection: RCV1Collection, data_name: str):
        """
        TODO

        data_name is used to refer to a specific configuration of the dataset, if results were previoulsly
        compute and saved, then they are loaded from disk instead of computed again, this is to save computation

        :param collection:
        :param data_name:
        """

        self._collection: RCV1Collection = collection
        self._data_name: data_name = data_name

        self._gaps_stats: Dict = dict()
        self._stats_computed: bool = False

    def __str__(self):
        return f"DGapComputation[Docs: {self._collection.n_docs}; Terms: {self._collection.n_terms}; Computed: {self._stats_computed}]"

    def __repr__(self):
        return str(self)

    # GAP

    @staticmethod
    def d_gap_to_bitsize(gap: int) -> int:
        if gap == 0:
            return 0
        return math.ceil(math.log(gap) + 1 / 7) * 8

    def compute_gaps_stats(self, try_load: bool = True):

        if self._stats_computed:
            log(info="Stats were already computed. Use `gaps_stats` to get them. ")
            return

        if try_load:
            if self._stats_available:
                log(info=f"Retrieving stats from disk at {self._stats_fp}. ")
                self._load_stats()
                return
            else:
                log(info=f"Information not present at {self._stats_fp}. ")

        # Need to compute gaps
        log(info="Computing gaps ")

        # Fields indices and indptr for faster access
        indices = self._collection.data.indices
        indptr = self._collection.data.indptr
        n_terms, n_docs = self._collection.n_terms, self._collection.n_docs

        # Accumulators per term
        tot_d_gap = np.zeros(n_terms, dtype=int)  # gap accumulator per term
        tot_gaps = np.zeros(n_terms, dtype=int)  # number of gaps per term
        max_d_gap = np.zeros(n_terms, dtype=int)  # maximum gap per term
        term_last_doc_id = np.zeros(n_terms, dtype=int)  # last doc-id seen per term, used to compute d-gap

        # Iterate through document (rows)
        for doc_id in tqdm(range(n_docs)):

            # Get the indices and data for the current document
            doc_indices = indices[indptr[doc_id]:indptr[doc_id + 1]]

            # Iterate through the terms in the current document
            for term_id in doc_indices:
                # Compute the d-gap
                gap = doc_id - term_last_doc_id[term_id]
                d_gap = self.d_gap_to_bitsize(gap=gap)

                # Update information
                tot_d_gap[term_id] += d_gap
                tot_gaps[term_id] += 1
                max_d_gap[term_id] = max(max_d_gap[term_id], d_gap)

                # Update the last doc-id for the current term
                term_last_doc_id[term_id] = doc_id

        avg_d_gap = np.around(tot_d_gap / tot_gaps, decimals=3)  # round to 3 decimals

        self._gaps_stats = {
            "tot_d_gap_pterm": tot_d_gap,
            "avg_d_gap_pterm": avg_d_gap,
            "max_d_gap_pterm": max_d_gap,
            "tot_d_gap": sum(tot_d_gap),
            "avg_d_gap": round(np.mean(avg_d_gap), 3),  # round to 3 decimals
            "max_d_gap": np.max(max_d_gap)
        }

        self._stats_computed = True


    @property
    def gaps_stats(self) -> Dict[str, np.ndarray[int | float] | int | float]:
        """
        It computes total gap of the postings derived from the matrix.

        :return: total d-gap.
        """

        if not self._stats_computed:
            raise Exception("Gaps stats were not computed yet. Use `compute_gaps_stats`. ")

        return self._gaps_stats

    # PLOT

    def _plot(self, key: str, title: str):
        """
        Plot gap information trend.

        :param key: key for gaps dictionary.
        :param title: plot title name.
        """

        # Get data
        data = self._gaps_stats[key]

        # Define number of bins
        num_bins = 900

        # Number of points per bin
        points_per_bin = len(data) // num_bins

        # Calculate bin averagesshift
        print(type(data))
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

        self._plot(key="tot_d_gap_pterm", title="Total d-gap per term")

    def plot_avg_d_gap(self):
        """
        Plot histogram of average d-gap per term.
        """

        self._plot(key="avg_d_gap_pterm", title="Average d-gap per term")

    def plot_max_d_gap(self):
        """
        Plot histogram of maximum d-gap per term.
        """

        self._plot(key="max_d_gap_pterm", title="Maximum d-gap per term")

    # SAVE / LOAD

    @property
    def _stats_fp(self) -> str:
        """
        :return: path to info file
        """

        sample_dir = get_collection_dir(collection_name=self._data_name)
        return path.join(sample_dir, "d-gap_stats.json")

    @property
    def _stats_available(self) -> bool:
        """
        Check if information is already available on disk
        """

        return path.exists(path=self._stats_fp)

    def save_stats(self):
        """
        Save information to proper directory.
        """

        log(info="Saving gaps statistics. ")

        sample_dir = get_collection_dir(collection_name=self._data_name)
        make_dir(path_=sample_dir)

        info_fp = self._stats_fp

        # Create JSON
        gaps_info = self.gaps_stats
        gaps_info["tot_d_gap"] = int(gaps_info["tot_d_gap"])
        gaps_info["avg_d_gap"] = float(gaps_info["avg_d_gap"])
        gaps_info["max_d_gap"] = int(gaps_info["max_d_gap"])
        gaps_info["tot_d_gap_pterm"] = [int(d_gap) for d_gap in gaps_info["tot_d_gap_pterm"]]
        gaps_info["avg_d_gap_pterm"] = [float(d_gap) for d_gap in gaps_info["avg_d_gap_pterm"]]
        gaps_info["max_d_gap_pterm"] = [int(d_gap) for d_gap in gaps_info["max_d_gap_pterm"]]

        store_json(path_=info_fp, obj=gaps_info)

    def _load_stats(self):
        """

        """

        gaps_info = load_json(path_=self._stats_fp)

        for key in ["tot_d_gap_pterm", "avg_d_gap_pterm", "max_d_gap_pterm"]:
            gaps_info[key] = np.array(gaps_info[key])

        self._gaps_stats = gaps_info
        self._stats_computed = True
