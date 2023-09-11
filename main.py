from os import path
from typing import List

from matplotlib import pyplot as plt

from io_ import get_dataset_dir, store_json, load_json
from model.rcv1 import DataConfig
from model.reassignment import OneStepReassignment, TwoStepReassignment


def two_step(n_cluster1: int, n_cluster2: int) -> float:
    """
    Perform two-step clustering
    :param n_cluster1: first level clusters
    :param n_cluster2: second level clusters
    :return: compression
    """

    config = DataConfig(name=f"rcv1-{n_cluster1}-v2", n_cluster=n_cluster1)
    reassignment = TwoStepReassignment(config=config, k2=n_cluster2)

    reassignment.reassign()

    inference = reassignment.inference

    inference.plot_avg_d_gap()

    return inference.avg_compression


def one_step(ks: List[int]):
    """
    Perform one-step clustering
    :param ks: list of clustering to evaluate
    """

    # Compression computation

    compression_dir = dict()

    print(f"Evaluating clusters: {ks}")

    for k in ks:

        print(f"Evaluating number of clusters: {k}")

        config = DataConfig(name=f"rcv1-{k}", n_cluster=k)
        reassignment = OneStepReassignment(config=config)
        reassignment.reassign()

        compression_dir[str(k)] = float(reassignment.inference.avg_compression)

    out_fp = path.join(get_dataset_dir(), "results.json")

    store_json(path_=out_fp, obj=compression_dir)

    # Plot

    compression_dir = load_json(out_fp)

    x = [int(key) for key in compression_dir.keys()]
    y = list(compression_dir.values())

    plt.plot(x, y, marker='o', linestyle='-', markersize=4, label='Data Points')
    plt.xticks([x[i] for i in range(0, len(x), 2)])

    plt.title("Postings compression")
    plt.xlabel('Number of clusters')
    plt.ylabel('Compression')

    img_fp = path.join(get_dataset_dir(), "compression.png")
    plt.savefig(img_fp)


if __name__ == "__main__":

    one_step(ks=[i * 10 for i in range(10, 31)])

    # compression = two_step(n_cluster1=170, n_cluster2=100)
    # print(compression)