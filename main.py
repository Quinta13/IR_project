from os import path

from matplotlib import pyplot as plt

from io_ import get_dataset_dir, store_json, load_json
from model.rcv1 import DataConfig
from model.reassignment import OneStepReassignment, TwoStepReassignment


def two_step():
    reassignment = TwoStepReassignment()
    config = DataConfig(name=f"rcv1-170-v2", n_cluster=170)

    reassignment.compute_compression(config=config)

    print(reassignment.compression)


def one_step():
    # Compression computation

    reassignment = OneStepReassignment()

    compression_dir = dict()

    ks = [i * 10 for i in range(10, 31)]
    print(f"Evaluating clusters: {ks}")

    for k in ks:
        print(f"Evaluating number of clusters: {k}")

        config = DataConfig(name=f"rcv1-{k}", n_cluster=k)
        reassignment.compute_compression(config=config)

        compression_dir[str(k)] = float(reassignment.compression)

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
    one_step()
