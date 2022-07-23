import gzip
import os
import pickle
import sys
from time import time

import pandas as pd
from tqdm import tqdm

from constatns import get_paths
from ex3_sib_vs_kmeans_setup import N_RUNS, SETUPS, DATASETS, ALGORITHMS, EMBEDDINGS, HIDDEN_PARAMS, EX_NAME
from ex3_sib_vs_kmeans_utils import populate_kwargs, str_kwargs

DATASETS_FULL_PATH, VECTORS_FULL_PATH, SUMMARY_FULL_PATH, \
    DATASETS_METADATA_FULL_PATH, SETUPS_METADATA_FULL_PATH, \
    EMBEDDINGS_META_FULL_PATH, PREDICTIONS_FULL_PATH, \
    METRICS_FULL_PATH, AGGREGATED_FULL_PATH, \
    TABLES_FULL_PATH, FIGURES_FULL_PATH, HARDWARE_PATH = get_paths(EX_NAME)


# cluster all datasets with all setups
def cluster():
    df_embeddings = pd.read_csv(EMBEDDINGS_META_FULL_PATH)

    results = []

    for dataset_type, dataset_desc in DATASETS.items():
        dataset_name, dataset_fetcher = dataset_desc
        dataset_file_name = dataset_type + '.pkl.gz'
        dataset_full_path = os.path.join(DATASETS_FULL_PATH, dataset_file_name)
        print("Reading dataset: %s from pickle" % dataset_name)
        with gzip.open(dataset_full_path, 'rb') as fp:
            dataset = pickle.load(fp)

        # run all setups
        for setup in SETUPS:
            algorithm_name, embedding_type = setup

            if isinstance(algorithm_name, tuple):
                algorithm_full_name = " + ".join([ALGORITHMS[name][0] for name in list(algorithm_name)])
            else:
                algorithm_full_name = ALGORITHMS[algorithm_name][0]

            print("\tRunning %s" % algorithm_full_name, flush=True)

            embedding_desc = EMBEDDINGS[embedding_type]
            vectorizer_name, vectorizer_factory, vectorizer_params = embedding_desc
            vectorizer_kwargs_list = populate_kwargs({}, vectorizer_params)
            for vectorizer_kwargs in vectorizer_kwargs_list:
                print("\t\tOn top of %s(%s)" % (vectorizer_name, vectorizer_kwargs), flush=True)

                vectors_file_name = embedding_type + '_' + str_kwargs(vectorizer_kwargs) + '.pkl'
                vectors_full_path = os.path.join(VECTORS_FULL_PATH, dataset_type, vectors_file_name)
                with open(vectors_full_path, 'rb') as fp:
                    vectors = pickle.load(fp)

                vectorizer_time = df_embeddings.query(
                    'dataset==\'%s\' & embedding==\'%s\' & vectors_file_name==\'%s\'' %
                    (dataset_name, embedding_type, vectors_file_name))['vectorizer_time'].item()

                clustering_algorithm_name = algorithm_name[1] if isinstance(algorithm_name, tuple) else algorithm_name
                algorithm_display_name, algorithm_factory, algorithm_params = ALGORITHMS[clustering_algorithm_name]

                algorithm_kwargs_list = populate_kwargs({
                    'n_clusters': dataset.n_clusters}, algorithm_params)

                for algorithm_kwargs in algorithm_kwargs_list:
                    with tqdm(range(N_RUNS), file=sys.stdout) as t:
                        for i in t:
                            algorithm_display_params = algorithm_kwargs.copy()
                            for param in HIDDEN_PARAMS:
                                algorithm_display_params.pop(param, None)
                            t.set_description("\t\t\t%s(%s)" % (algorithm_display_name, str(algorithm_display_params)))
                            t0 = time()
                            algorithm_instance = algorithm_factory(**algorithm_kwargs)
                            algorithm_instance.fit(vectors)
                            algorithm_time = time() - t0
                            results.append({
                                'dataset': dataset_name,
                                'n_samples': dataset.n_samples,
                                'n_clusters': dataset.n_clusters,
                                'algorithm': algorithm_full_name,
                                'algorithm_params': str(algorithm_display_params),
                                'vectorizer_name': vectorizer_name,
                                'vectorizer_params': vectorizer_kwargs,
                                'vectorizer_time': vectorizer_time,
                                'run': i,
                                'algorithm_time': algorithm_time,
                                'total_time': algorithm_time + vectorizer_time,
                                'predicted_labels': algorithm_instance.labels_.tolist(),
                                'gold_labels': dataset.target.tolist()
                            })

        pd.DataFrame(results).to_csv(PREDICTIONS_FULL_PATH)


def main():
    cluster()


if __name__ == '__main__':
    main()
