# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import platform

import numpy as np
import psutil
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


def get_system_desc():
    return {
        'machine': platform.machine(),
        'version': platform.version(),
        'platform': platform.platform(),
        'uname': platform.uname(),
        'system': platform.system(),
        'processor': platform.processor(),
        'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3)))+" GB",
        'cpus_physical': psutil.cpu_count(logical=False),
        'cpus_logical': psutil.cpu_count(logical=True)
    }
