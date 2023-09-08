from os import path

from matplotlib import pyplot as plt

from io_ import store_json, get_dataset_dir
from model.clustering import KMeansClustering
from model.d_gap import DGapComputation, DGapComputationReassigned, DGapInference
from model.rcv1 import DataConfig, RCV1Loader
from model.reassignment import DocIdReassignment


def perform_reassignment(config: DataConfig, loader: RCV1Loader):

    collection = loader.load()

    print("  -  Computing d-gap...")
    dgap = DGapComputation(collection=collection, data_name=config.name)
    dgap.compute_d_gaps()
    dgap.save_d_gaps()

    print("  -  Evaluating clustering...")
    kmeans = KMeansClustering(collection=collection, data_name=config.name, k=config.n_cluster)
    kmeans.fit()
    kmeans.save_labeling()

    print("  -  Computing centroids...")
    collection_clusters = kmeans.clusters
    collection_clusters.compute_centroids()
    collection_clusters.save_centroids()

    print("  -  Reassign doc-id...")
    reassignment_computation = DocIdReassignment(
        cluster=collection_clusters,
        data_name=config.name
    )
    reassignment_computation.solve()
    reassignment_computation.save_order()
    collection_reassigned = reassignment_computation.reassign_doc_id()

    print("  -  Computing d-gap after reassignment...")
    dgap_reass = DGapComputationReassigned(collection=collection_reassigned, data_name=config.name)
    dgap_reass.compute_d_gaps()
    dgap_reass.save_d_gaps()

    print("  -  Performing inference...")
    inference = DGapInference(d_gap_original=dgap, d_gap_reassigned=dgap_reass, data_name=config.name)
    inference.plot_avg_d_gap()
    return inference.avg_compression


if __name__ == "__main__":

    compression_dir = dict()

    ks = [i * 10 for i in range(10, 31)]
    print(f"Evaluating clusters: {ks}")

    print("Loading data...")
    loader_ = RCV1Loader()

    for k in ks:

        print(f"Evaluating number of clusters: {k}")

        config_ = DataConfig(name=f"rcv1-{k}", n_cluster=k)
        compression = perform_reassignment(config=config_, loader=loader_)

        compression_dir[str(k)] = float(compression)

    out_fp = path.join(get_dataset_dir(), "results.json")

    store_json(path_=out_fp, obj=compression_dir)

    # Plot

    x = [int(key) for key in compression_dir.keys()]
    y = list(compression_dir.values())

    plt.scatter(x, y)

    plt.xlabel('Number of clusters')
    plt.ylabel('Compression')

    img_fp = path.join(get_dataset_dir(), "compression.png")
    plt.savefig(img_fp)

