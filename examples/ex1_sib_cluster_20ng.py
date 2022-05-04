# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import pickle
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sib import SIB, clustering_utils
import example_utils


# This test is meant for evaluating the clustering quality and
# it runs with 4 random initializations. For evaluating speed,
# we use a single initialization.
speed_test_mode = False

# step 0 - create an output directory if it does not exist yet
output_path = os.path.join("output", "ex1")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# step 1 - read the dataset
texts, gold_labels, n_clusters, topics, n_samples = example_utils.fetch_20ng('all')
print("Dataset contains %d texts from %d topics" % (n_samples, n_clusters))

# step 2 - represent the clustering data using bow of the 5K most frequent
# unigrams in the dataset. cache the vectors for faster runs afterwards.
vectors_path = os.path.join(output_path, 'vectors.pkl')
if os.path.exists(vectors_path):
    with open(vectors_path, "rb") as fp:
        vectors, terms = pickle.load(fp)
else:
    print("Vectorizing texts...")
    vectorizing_start_t = time()
    vectorizer = CountVectorizer(tokenizer=example_utils.custom_tokenizer,
                                 max_df=0.5, min_df=10, max_features=5000)
    vectors = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names()
    vectorizing_end_t = time()
    print("Vectorizing time: %.3f secs." % (vectorizing_end_t - vectorizing_start_t))
    with open(vectors_path, "wb") as fp:
        pickle.dump((vectors, terms), fp)

print("Vector size: %d" % vectors.shape[1])
print('-------')


# step 3 - create an instance of sIB and run the clustering
# n_init = the number of random initializations to perform
# max_ter = the maximal number of iterations in each initialization
# n_jobs = the maximal number of initializations to run in parallel
print("Clustering texts...")
clustering_start_t = time()
n_init = 1 if speed_test_mode else 4
sib = SIB(n_clusters=n_clusters, random_state=128, n_init=n_init,
          n_jobs=-1, max_iter=15, verbose=True)
sib.fit(vectors)
clustering_end_t = time()
print("Clustering time: %.3f secs." % (clustering_end_t - clustering_start_t))
print('-------')


# step 4 - standard clustering evaluation metrics
homogeneity = metrics.homogeneity_score(gold_labels, sib.labels_)
completeness = metrics.completeness_score(gold_labels, sib.labels_)
v_measure = metrics.v_measure_score(gold_labels, sib.labels_)
ami = metrics.adjusted_mutual_info_score(gold_labels, sib.labels_)
ari = metrics.adjusted_rand_score(gold_labels, sib.labels_)
print("Clustering evaluation metrics:")
print("Homogeneity: %0.3f" % homogeneity)
print("Completeness: %0.3f" % completeness)
print("V-measure: %0.3f" % v_measure)
print("Adjusted Mutual-Information: %.3f" % ami)
print("Adjusted Rand-Index: %.3f" % ari)
print('-------')


# step 5 - perform a p-value analysis to identify characteristic terms in every cluster
# the analysis is done on the document-level. For every cluster, for every term, we
# check whether the number of docs in this cluster in which the term appears term appears
# is unusually large compared to the number of docs in which the term appears in the whole
# corpus. By 'unusually large' we mean that the likelihood of such an event is smaller than
# the threshold (0.01 by default) based on a hyper-geometric test. From the terms that pass
# the p-value test, we report the top k (15 by default) in ascending order.
print("Performing p-value analysis")
p_value_analysis_start_t = time()
clusters = clustering_utils.get_clusters(sib.labels_)
cluster_key_terms = clustering_utils.get_key_terms(vectors, clusters,
                                                   p_value_threshold=0.01, top_k=15)
cluster_key_texts = clustering_utils.get_ket_texts(vectors, clusters, cluster_key_terms, 2)
p_value_analysis_end_t = time()
print("P-value analysis time: %.3f secs." % (p_value_analysis_end_t - p_value_analysis_start_t))

print(cluster_key_texts)

# step 6 - align the generated clusters to the original classes
# this is done only for having a more informative report
label_enrichment = clustering_utils.get_enriched_labels(gold_labels, sib.labels_,
                                                        threshold=0.15)


# report class enrichment and key terms
print('Class enrichment and key-terms per cluster:')
for label, key_terms in cluster_key_terms.items():
    print("%02d: [%s]" % (label, ", ".join("%s (%.2f%%)" % (topics[enriched_class], enrichment_value * 100)
          for (enriched_class, enrichment_value) in label_enrichment[label])))
    print("\t[%s]" % ", ".join(terms[term] for term in key_terms))
print('-------')


# report class precision/recall/f1
reindexed_labels = clustering_utils.reindex_labels(gold_labels, sib.labels_, sib.labels_)
report = metrics.classification_report(gold_labels, reindexed_labels,
                                       target_names=topics, zero_division=0)
print("Full classification report:")
print(report)

# generate a heatmap and save it as svg
heatmap_path = os.path.join(output_path, 'sib_heatmap')
example_utils.create_heatmap(gold_labels, sib.labels_,
                             topics, 'sIB clustering heatmap',
                             heatmap_path, use_svg=True)
heatmap_path += '.svg'

# save a report
report_path = os.path.join(output_path, "sib_report_1.txt")
with open(report_path, "wt") as f:
    f.write(str(sib) + "\n")
    f.write("Size: %d vectors\n" % vectors.shape[0])
    f.write("Time: %.3f seconds\n" % (clustering_end_t - clustering_start_t))
    f.write("\n")
    f.write("Clustering evaluation metrics:\n")
    f.write("\tHomogeneity: %0.3f\n" % homogeneity)
    f.write("\tCompleteness: %0.3f\n" % completeness)
    f.write("\tV-measure: %0.3f\n" % v_measure)
    f.write("\tAdjusted Mutual Information: %.3f\n" % ami)
    f.write("\tAdjusted Rand Index: %.3f\n" % ari)
    f.write("\n")
    f.write("Class enrichment and key-terms per cluster:\n")
    for label, key_terms in cluster_key_terms.items():
        f.write("\t%02d: [%s]\n" % (label, ", ".join("%s (%.2f%%)" % (topics[enriched_class], enrichment_value * 100)
                                                     for (enriched_class, enrichment_value) in label_enrichment[label])))
        f.write("\t\t[%s]\n" % ", ".join(terms[term] for term in key_terms))
    f.write("\n")
    f.write("Classification report:\n")
    f.write(report + "\n")

print("Recorded results:")
print("\tHeatmap saved at: %s" % heatmap_path)
print("\tFull report saved at: %s" % report_path)
print()
print('Done.')
