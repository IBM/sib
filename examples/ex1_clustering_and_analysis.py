# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from time import time

from sib import SIB, clustering_utils
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

import custom_tokenizer
import heatmap
from constatns import get_paths
from eval_datasets import fetch_20ng

EX_NAME = 'ex1'
DATASET_NAME = '20 News Groups'
DATASET_FETCHER = fetch_20ng
N_INIT = 10

DATASETS_FULL_PATH, _, SUMMARY_FULL_PATH, \
    _, _, _, _, _, _, _, _, _ = get_paths(EX_NAME)

# dataset reading
print("Reading dataset: %s..." % DATASET_NAME, end=" ", flush=True)
t0 = time()
dataset = DATASET_FETCHER()
print("done in %.3f sec." % (time() - t0), end=" ", flush=True)
print("%d samples, %d classes" % (dataset.n_samples, dataset.n_clusters))

# vectorization stage
print("Preprocessing and vectorizing texts...", end=" ", flush=True)
vectorizing_start_t = time()
vectorizer = CountVectorizer(tokenizer=custom_tokenizer.custom_tokenizer,
                             max_df=0.5, min_df=10, max_features=5000)
vectors = vectorizer.fit_transform(dataset.data)
terms = vectorizer.get_feature_names_out()
vectorizing_end_t = time()
print("done in %.3f secs." % (vectorizing_end_t - vectorizing_start_t))

# clustering stage
print("Clustering vectors...", end=" ", flush=True)
clustering_start_t = time()
sib = SIB(n_clusters=dataset.n_clusters, random_state=128, n_init=N_INIT)
sib.fit(vectors)
clustering_end_t = time()
print("done in %.3f secs." % (clustering_end_t - clustering_start_t))

# clustering evaluation
homogeneity = metrics.homogeneity_score(dataset.target, sib.labels_)
completeness = metrics.completeness_score(dataset.target, sib.labels_)
v_measure = metrics.v_measure_score(dataset.target, sib.labels_)
ami = metrics.adjusted_mutual_info_score(dataset.target, sib.labels_)
ari = metrics.adjusted_rand_score(dataset.target, sib.labels_)
print("Clustering evaluation metrics:")
print("Homogeneity: %0.3f" % homogeneity)
print("Completeness: %0.3f" % completeness)
print("V-measure: %0.3f" % v_measure)
print("Adjusted Mutual-Information: %.3f" % ami)
print("Adjusted Rand-Index: %.3f" % ari)
print('-------')


# perform a p-value analysis to identify characteristic terms in every cluster
# the analysis is done on the document-level. For every cluster, for every term, we
# check whether the number of docs in this cluster in which the term appears
# is unusually large compared to the number of docs in which the term appears in the whole
# corpus. By 'unusually large' we mean that the likelihood of such an event is smaller than
# the threshold (e.g. 0.01) based on a hyper-geometric test. From the terms that pass
# the p-value test, we report the top k (e.g. 15) in ascending order.
print("Performing cluster analysis...", end=" ", flush=True)
cluster_analysis_start_t = time()
clusters = clustering_utils.get_clusters(sib.labels_)
cluster_key_terms = clustering_utils.get_key_terms(vectors, clusters,
                                                   p_value_threshold=0.01, top_k=15)

# align the generated clusters to the original classes
# this is done only for having a more informative report
label_enrichment = clustering_utils.get_enriched_labels(dataset.target, sib.labels_,
                                                        threshold=0.15)

cluster_analysis_end_t = time()
print("done in %.3f secs." % (cluster_analysis_end_t - cluster_analysis_start_t))

# report class enrichment and key terms
print('Class enrichment and key-terms per cluster:')
for label, key_terms in cluster_key_terms.items():
    print("%02d: [%s]" % (label, ", ".join("%s (%.2f%%)" % (dataset.target_names[enriched_class],
                                                            enrichment_value * 100)
          for (enriched_class, enrichment_value) in label_enrichment[label])))
    print("\t[%s]" % ", ".join(terms[term] for term in key_terms))

print('-------')

# report class precision/recall/f1
reindexed_labels = clustering_utils.reindex_labels(dataset.target, sib.labels_, sib.labels_)
report = metrics.classification_report(dataset.target, reindexed_labels,
                                       target_names=dataset.target_names, zero_division=0)
print("Full classification report:")
print(report)

# for saving the heatmap and summary
os.makedirs(SUMMARY_FULL_PATH, exist_ok=True)

# generate a heatmap and save it as svg
heatmap_path = os.path.join(SUMMARY_FULL_PATH, 'sib_heatmap')
heatmap.create_heatmap(dataset.target, sib.labels_,
                       dataset.target_names, 'sIB clustering heatmap',
                       heatmap_path, use_svg=True)
heatmap_path += '.svg'

# save a report
report_path = os.path.join(SUMMARY_FULL_PATH, "report.txt")
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
        f.write("\t%02d: [%s]\n" % (label, ", ".join("%s (%.2f%%)" % (dataset.target_names[enriched_class],
                                                                      enrichment_value * 100)
                                                     for (enriched_class, enrichment_value) in
                                                     label_enrichment[label])))
        f.write("\t\t[%s]\n" % ", ".join(terms[term] for term in key_terms))
    f.write("\n")
    f.write("Classification report:\n")
    f.write(report + "\n")

print("Recorded results:")
print("\tHeatmap saved at: %s" % heatmap_path)
print("\tFull report saved at: %s" % report_path)
print()
print('Done.')
