# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score

from sib import SIB, clustering_utils
import example_utils

# we will treat the 20 news groups dataset as a benchmark. it contains
# messages about 20 topics that we want to cluster to 20 clusters.

# step 0 - create an output directory if it does not exist
output_path = os.path.join("output", "ex3")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# step 1 - read the dataset
clustering_texts, clustering_gold_labels_array, \
    clustering_n_clusters, _, clustering_n_samples = example_utils.fetch_20ng('train')
print("Clustering dataset contains %d texts from %d topics" % (clustering_n_samples, clustering_n_clusters))

classification_texts, classification_gold_labels_array, \
    classification_n_clusters, classification_topics, \
    classification_n_samples = example_utils.fetch_20ng('test')
print("Classification dataset contains %d texts from %d topics" % (classification_n_samples, classification_n_clusters))

print()

# make sure we have 20 topics in each part of the split
assert clustering_n_clusters == classification_n_clusters == 20

for algorithm_name in ['kmeans', 'sib']:
    print("Running with %s:" % algorithm_name)

    # step 2 - represent the clustering data using bow
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

    bow_model = vectorizer.fit(clustering_texts)
    clustering_vectors = vectorizer.transform(clustering_texts)
    print("Created vectors for %d samples for clustering" % clustering_vectors.shape[0])

    # step 3 - cluster the clustering data
    # these settings should be adjusted based on the real-world use-case.
    # the default for k-means in sklearn is n_init=10, max_iter=300;
    # for sib it is enough to use max_iter=15
    # here we use max_iter=15 for both to be able to compare run-time
    # we set kmeans algorithm to 'full' since it gives better run-time
    if algorithm_name is 'kmeans':
        algorithm = KMeans(n_clusters=clustering_n_clusters, n_init=10, n_jobs=-1, max_iter=15, algorithm='full')
    elif algorithm_name is 'sib':
        algorithm = SIB(n_clusters=clustering_n_clusters, n_init=10, n_jobs=-1, max_iter=15)
    else:
        raise ValueError("Unsupported algorithm %s" % algorithm_name)

    clustering_start_t = time()
    clustering_model = algorithm.fit(clustering_vectors)
    clustering_end_t = time()
    print("Clustering time: %.3f secs." % (clustering_end_t - clustering_start_t))

    clustering_predictions_array = algorithm.labels_
    ami = metrics.adjusted_mutual_info_score(clustering_gold_labels_array, clustering_predictions_array)
    ari = metrics.adjusted_rand_score(clustering_gold_labels_array, clustering_predictions_array)
    print("Clustering measures: AMI: %.3f, ARI: %.3f" % (ami, ari))
    with open(os.path.join(output_path, algorithm_name + "_report.txt"), "wt") as f:
        f.write("Clustering:\n")
        f.write(str(algorithm) + "\n")
        f.write("Size: %d vectors\n" % clustering_vectors.shape[0])
        f.write("Time: %.3f seconds\n" % (clustering_end_t - clustering_start_t))
        f.write("Measures:\n")
        f.write("\tAdjusted Mutual Information: %.3f\n\tAdjusted Rand Index: %.3f\n" % (ami, ari))
        f.write("\n\n")

    # step 4 - now assume that we got some new examples and we want to classify
    # them to the clusters we have. we represent them using the vocabulary we
    # induced for the clustering stage, and then classify them to the clusters.
    classification_vectors = vectorizer.transform(classification_texts)
    print("Created vectors for %d samples for classification" % classification_vectors.shape[0])
    classification_start_t = time()
    classification_predictions_array = algorithm.predict(classification_vectors)
    classification_end_t = time()
    print("Classification time: %.3f secs." % (classification_end_t - classification_start_t))

    # save a heatmap
    example_utils.create_heatmap(classification_gold_labels_array, classification_predictions_array,
                                 classification_topics, algorithm_name + ' classification heatmap',
                                 os.path.join(output_path, algorithm_name + '_heatmap'))

    # the clustering algorithm uses its own enumeration of clusters; hence the
    # cluster ids that it returns are not aligned with the class ids in the 
    # ground-truth of the dataset. when we evaluate clustering, this is not a 
    # problem because the measures we used (ami, ari, etc.) can tolerate this.
    # however, in order to create a report on the accuracy of classification,
    # we need to align the cluster ids and the class ids. this is what the code
    # below is responsible for. the alignment is done based on the clustering 
    # result as follows:
    # the clustering result is a partition of the samples into k (20) clusters.
    # for every predicted label (cluster id), we look at the samples assigned to it,
    # and check which is the most dominant gold label in this population. for example,
    # for the samples in cluster id = 3, we may find that the most common gold label 
    # (class id) is 8. subsequently, we rewrite the classification_predictions_array
    # using the aligned class ids.
    # now we can use sklearn's standard classification metrics to report the 
    # quality of the classification compared to the ground-truth.
    # alignment = clustering_utils.get_alignment(clustering_gold_labels_array, clustering_predictions_array)
    # new_predicted_array = np.zeros_like(classification_predictions_array)
    # for predicted_label, gold_label in alignment.items():
    #     predicted_label_indices = np.where(classification_predictions_array == predicted_label)
    #     new_predicted_array[predicted_label_indices] = gold_label
    # classification_predictions_array = new_predicted_array
    classification_predictions_array = clustering_utils.reindex_labels(clustering_gold_labels_array,
                                                                       clustering_predictions_array,
                                                                       classification_predictions_array)

    print("Classification accuracy: %.2f%%" % (accuracy_score(classification_gold_labels_array,
                                                              classification_predictions_array) * 100))

    # add a full report to the output file
    with open(os.path.join(output_path, algorithm_name + "_report.txt"), "at") as f:
        f.write("Classification:\n")
        f.write("Size: %d vectors\n" % classification_vectors.shape[0])
        f.write("Time: %.3f seconds\n" % (classification_end_t - classification_start_t))
        f.write("Measures:\n")
        f.write(classification_report(classification_gold_labels_array,
                                      classification_predictions_array,
                                      target_names=classification_topics,
                                      zero_division=0))
        f.write("\n")

    print()
