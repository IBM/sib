import gzip
import json
import os
import pickle
from time import time

import pandas as pd

from constatns import get_paths
from ex3_sib_vs_kmeans_setup import SETUPS, DATASETS, ALGORITHMS, EMBEDDINGS, DATASET_MAX_SIZE, EX_NAME
from ex3_sib_vs_kmeans_utils import populate_kwargs, str_kwargs, get_system_desc
from vectorizers import GloveVectorizer


DATASETS_FULL_PATH, VECTORS_FULL_PATH, SUMMARY_FULL_PATH, \
    DATASETS_METADATA_FULL_PATH, SETUPS_METADATA_FULL_PATH, \
    EMBEDDINGS_META_FULL_PATH, PREDICTIONS_FULL_PATH, \
    METRICS_FULL_PATH, AGGREGATED_FULL_PATH, \
    TABLES_FULL_PATH, FIGURES_FULL_PATH, HARDWARE_PATH = get_paths(EX_NAME)


def prepare():
    os.makedirs(os.path.join(DATASETS_FULL_PATH), exist_ok=True)
    os.makedirs(os.path.join(VECTORS_FULL_PATH), exist_ok=True)
    os.makedirs(os.path.join(SUMMARY_FULL_PATH), exist_ok=True)
    os.makedirs(os.path.join(TABLES_FULL_PATH), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_FULL_PATH), exist_ok=True)
    GloveVectorizer.download_model()


def record_setups():
    data = []

    # go over all setups
    for setup in SETUPS:
        algorithm_name, embedding_type = setup
        algorithm_display_name = ALGORITHMS[algorithm_name][0]
        embedding_display_name, _, _ = EMBEDDINGS[embedding_type]
        data.append({
            'algorithm': algorithm_display_name,
            'embedding': embedding_display_name
        })

    # store the metadata of the setups
    pd.DataFrame(data).to_csv(SETUPS_METADATA_FULL_PATH)


def read_datasets():
    data = []

    # read all datasets and save into pickle
    for dataset_type, dataset_desc in DATASETS.items():
        dataset_name, dataset_fetcher = dataset_desc
        dataset_file_name = dataset_type + '.pkl.gz'
        dataset_full_path = os.path.join(DATASETS_FULL_PATH, dataset_file_name)
        if not os.path.exists(dataset_full_path):
            print("Reading dataset: %s..." % dataset_name, end=" ", flush=True)
            t0 = time()
            dataset = dataset_fetcher(DATASET_MAX_SIZE)
            print("done in %.3f sec." % (time() - t0), end=" ", flush=True)
            print("%d samples, %d classes" % (dataset.n_samples, dataset.n_clusters))
            with gzip.open(dataset_full_path, 'wb') as fp:
                pickle.dump(dataset, fp)
            data.append({
                'dataset': dataset_name,
                'n_samples': dataset.n_samples,
                'n_samples_org': dataset.n_samples_org,
                'word_count_mean': dataset.word_count_mean,
                'word_count_std': dataset.word_count_std,
                'word_count_median': dataset.word_count_median,
                'n_clusters': dataset.n_clusters,
                'topics': dataset.target_names
            })

    # store the metadata of the datasets
    pd.DataFrame(data).to_csv(DATASETS_METADATA_FULL_PATH)


def vectorize():

    # vectorize all datasets for all setups
    if not os.path.exists(EMBEDDINGS_META_FULL_PATH):
        embedding_results = []

        for dataset_type, dataset_desc in DATASETS.items():
            dataset_name, dataset_fetcher = dataset_desc
            dataset_file_name = dataset_type + '.pkl.gz'
            dataset_full_path = os.path.join(DATASETS_FULL_PATH, dataset_file_name)
            print("Reading dataset: %s from pickle" % dataset_name)
            with gzip.open(dataset_full_path, 'rb') as fp:
                dataset = pickle.load(fp)
                texts = dataset.data

            for embedding_type, embedding_desc in EMBEDDINGS.items():
                vectorizer_name, vectorizer_factory, vectorizer_params = embedding_desc
                vectorizer_kwargs_list = populate_kwargs({}, vectorizer_params)
                print(f'\tEmbedding: {embedding_type}')
                for vectorizer_kwargs in vectorizer_kwargs_list:
                    vectors_file_name = embedding_type + '_' + str_kwargs(vectorizer_kwargs) + '.pkl'
                    os.makedirs(os.path.join(VECTORS_FULL_PATH, dataset_type), exist_ok=True)
                    vectors_full_path = os.path.join(VECTORS_FULL_PATH, dataset_type, vectors_file_name)
                    print(f"\t\tWith: {str(vectorizer_kwargs) if len(vectorizer_kwargs)>0 else 'defaults'}...", end=' ')
                    t0 = time()
                    vectorizer_instance = vectorizer_factory(**vectorizer_kwargs)
                    vectors = vectorizer_instance.fit_transform(texts)
                    vectorizer_time = time() - t0
                    embedding_results.append({
                        'dataset': dataset_name,
                        'embedding': embedding_type,
                        'vectorizer_name': vectorizer_name,
                        'vectorizer_params': vectorizer_kwargs,
                        'vectorizer_time': vectorizer_time,
                        'vectors_file_name': vectors_file_name,
                    })
                    print(f'done in: {vectorizer_time:.3f} secs.')
                    with open(vectors_full_path, 'wb') as fp:
                        pickle.dump(vectors, fp)

        # store the metadata of the embeddings
        pd.DataFrame(embedding_results).to_csv(EMBEDDINGS_META_FULL_PATH)


def record_hardware():
    hardware_desc = get_system_desc()
    with open(HARDWARE_PATH, 'wt') as f:
        json.dump(hardware_desc, f, indent=4)


def main():
    prepare()
    record_setups()
    read_datasets()
    vectorize()
    record_hardware()


if __name__ == '__main__':
    main()
