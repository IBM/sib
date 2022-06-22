# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from sklearn.datasets import fetch_20newsgroups


def fetch_20ng(subset):
    dataset = fetch_20newsgroups(subset=subset, categories=None,
                                 shuffle=True, random_state=256)
    texts = dataset.data
    gold_labels_array = dataset.target
    unique_labels = np.unique(gold_labels_array)
    n_clusters = unique_labels.shape[0]
    topics = dataset.target_names
    n_samples = len(texts)
    return texts, gold_labels_array, n_clusters, topics, n_samples


