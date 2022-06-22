# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

from sib import SIB
import example_utils


# we will treat the 20 news groups dataset as a benchmark. it contains
# messages about 20 topics that we want to cluster to 20 clusters.

# by default, this test is meant for evaluating the clustering
# quality and it runs with 10 random initializations. however,
# if we are interested in evaluating speed, we will use a single
# initialization.
import heatmap

speed_test_mode = False

# step 0 - create an output directory if it does not exist
output_path = os.path.join("output", "ex2")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# step 1 - read the dataset
texts, gold_labels_array, n_clusters, topics, n_samples = example_utils.fetch_20ng('all')
print("Clustering dataset contains %d texts from %d topics" % (n_samples, n_clusters))
print()

# if you wish to apply any sort of text preprocessing like stemming /
# lemmatization, filtering of stop words, etc. do it right here.
# for example:
# texts = remove_stop_words(texts)
# texts = lemmatize(texts)

for algorithm_name in ['kmeans', 'sib']:
    print("Running with %s:" % algorithm_name)

    # step 2 - represent the clustering data using bag of words vectors.
    # adjust the defaults of the vectorizer as needed. normally limiting
    # the vocabulary to the 10k most frequent unigrams is a good choice.
    # it is possible to use ngrams of various sizes and also to control
    # the minium and maximum number of times that a word should appear
    # to be considered as a candidate for the vocabulary.
    if algorithm_name is 'kmeans':
        vectorizer = TfidfVectorizer(max_features=10000)
    elif algorithm_name is 'sib':
        vectorizer = CountVectorizer(max_features=10000)
    else:
        raise ValueError("Unsupported algorithm %s" % algorithm_name)

    vectors = vectorizer.fit_transform(texts)
    print("Created vectors for %d texts" % vectors.shape[0])

    # step 3 - cluster the data
    # these settings should be adjusted based on the real-world use-case.
    # the default for k-means in sklearn is n_init=10, max_iter=300;
    # for sib it is enough to use max_iter=15.
    # here we use max_iter=15 for both to be able to compare run-time
    # we set kmeans algorithm to 'full' since it gives better run-time
    n_init = 1 if speed_test_mode else 10
    if algorithm_name is 'kmeans':
        algorithm = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=-1, max_iter=15, algorithm='full')
    elif algorithm_name is 'sib':
        algorithm = SIB(n_clusters=n_clusters, n_init=n_init, n_jobs=-1, max_iter=15)
    else:
        raise ValueError("Unsupported algorithm %s" % algorithm_name)

    clustering_start_t = time()
    algorithm.fit(vectors)
    clustering_end_t = time()

    predictions_array = algorithm.labels_

    # save a heatmap
    heatmap.create_heatmap(gold_labels_array, predictions_array,
                           topics, algorithm_name + ' clustering heatmap',
                           os.path.join(output_path, algorithm_name + '_heatmap'))

    # measure the clustering quality and save a report
    ami = metrics.adjusted_mutual_info_score(gold_labels_array, predictions_array)
    ari = metrics.adjusted_rand_score(gold_labels_array, predictions_array)
    print("Clustering time: %.3f secs." % (clustering_end_t - clustering_start_t))
    print("Clustering measures: AMI: %.3f, ARI: %.3f" % (ami, ari))
    with open(os.path.join(output_path, algorithm_name + "_report.txt"), "wt") as f:
        f.write("Clustering:\n")
        f.write(str(algorithm) + "\n")
        f.write("Size: %d vectors\n" % vectors.shape[0])
        f.write("Time: %.3f seconds\n" % (clustering_end_t - clustering_start_t))
        f.write("Measures:\n")
        f.write("\tAdjusted Mutual Information: %.3f\n\tAdjusted Rand Index: %.3f\n" % (ami, ari))
        f.write("\n\n")
    print()
