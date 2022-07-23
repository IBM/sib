# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from time import time

from sib import SIB, clustering_utils
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

from constatns import get_paths
from eval_datasets import fetch_20ng

EX_NAME = 'ex2'
DATASET_NAME = '20 News Groups'
DATASET_FETCHER = fetch_20ng
MAX_SIZE = None
N_INIT = 4

DATASETS_FULL_PATH, VECTORS_FULL_PATH, SUMMARY_FULL_PATH, \
    _, _, _, _, _, _, _, _, _ = get_paths(EX_NAME)

# read the datasets - train for clustering, test for classification
train_dataset = DATASET_FETCHER(None, 'train')
test_dataset = DATASET_FETCHER(None, 'test')

# prepare a vectorizer and vectorize the train dataset
vectorizer = CountVectorizer(max_features=10000)
train_vectors = vectorizer.fit_transform(train_dataset.data)

# cluster the data
t0 = time()
algorithm = SIB(n_clusters=train_dataset.n_clusters)
algorithm.fit(train_vectors)
clustering_time = time() - t0
train_labels = algorithm.labels_

# report some metrics
ami = metrics.adjusted_mutual_info_score(train_dataset.target, algorithm.labels_)
ari = metrics.adjusted_rand_score(train_dataset.target, algorithm.labels_)
print("Clustering quality: AMI: %.3f, ARI: %.3f" % (ami, ari))

# reuse the same vectorizer to vectorize the data for classification
test_vectors = vectorizer.transform(test_dataset.data)

# classify the test data to the existing clusters
t0 = time()
test_labels = algorithm.predict(test_vectors)
classification_time = time() - t0

# re-align the cluster ids with the class ids
test_labels = clustering_utils.reindex_labels(train_dataset.target, train_labels, test_labels)

# report the accuracy of classification
print("Classification accuracy: %.2f%%" % (accuracy_score(test_dataset.target,
                                                          test_labels) * 100))

# save a report
os.makedirs(SUMMARY_FULL_PATH, exist_ok=True)
with open(os.path.join(SUMMARY_FULL_PATH, "report.txt"), "wt") as f:
    f.write("Clustering:\n")
    f.write("Size: %d samples\n" % train_vectors.shape[0])
    f.write("Time: %.3f seconds\n" % clustering_time)
    f.write("Measures:\n")
    f.write("\tAdjusted Mutual Information: %.3f\n"
            "\tAdjusted Rand Index: %.3f\n" % (ami, ari))
    f.write("\n\n")
    f.write("Classification:\n")
    f.write("Size: %d samples\n" % test_vectors.shape[0])
    f.write("Time: %.3f seconds\n" % classification_time)
    f.write("Measures:\n")
    f.write(classification_report(test_dataset.target, test_labels,
                                  target_names=test_dataset.target_names,
                                  zero_division=0))
    f.write("\n")
