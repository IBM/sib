# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import re
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics.cluster import contingency_matrix
from sklearn.datasets import fetch_20newsgroups
from scipy.stats import hypergeom
from scipy.sparse import csr_matrix
from nltk.stem.snowball import SnowballStemmer


def create_heatmap(gold_array, predicted_array, names, title, file_name,
                   threshold=0.05, use_png=False, use_svg=True):
    gold_labels = np.unique(gold_array)
    gold_labels_len = len(gold_labels)
    heatmap_matrix = contingency_matrix(gold_array, predicted_array)
    reordered_array = np.zeros_like(predicted_array)
    for i in range(gold_labels_len):
        gold_label, predicted_label = np.unravel_index(np.argmax(heatmap_matrix, axis=None), heatmap_matrix.shape)
        heatmap_matrix[gold_label, :] = -1
        heatmap_matrix[:, predicted_label] = -1
        predicted_indices = np.where(predicted_array == predicted_label)[0]
        np.put(reordered_array, predicted_indices, [gold_label], mode='wrap')
    heatmap_matrix = contingency_matrix(gold_array, reordered_array)
    sums_vector = heatmap_matrix.sum(axis=1)
    heatmap_matrix = np.divide(heatmap_matrix, sums_vector[:, np.newaxis])
    heatmap_matrix = np.around(heatmap_matrix, decimals=2)
    mask = np.isclose(heatmap_matrix, np.zeros_like(heatmap_matrix), atol=threshold)
    plt.ioff()
    plt.figure(figsize=(gold_labels_len * 0.667, gold_labels_len * 0.667))
    ax = sns.heatmap(heatmap_matrix,
                     cmap="BuGn",
                     yticklabels=names,
                     fmt=".2f",
                     mask=mask,
                     annot=True, cbar=False)
    ax.set_title(title)
    ax.figure.tight_layout()
    if use_png:
        plt.savefig(file_name + ".png", format='png', dpi=300)
    if use_svg:
        plt.savefig(file_name + ".svg", format='svg')
    plt.close()


def reindex_labels(gold_array, predicted_array, target_array):
    label_distribution = get_label_distribution(gold_array, predicted_array)
    label_alignment = label_distribution.argmax(axis=1)
    new_target_array = np.zeros_like(target_array)
    for predicted_label, gold_label in enumerate(label_alignment):
        indices = np.where(target_array == predicted_label)
        new_target_array[indices] = gold_label
    return new_target_array


def get_label_distribution(gold_array, predicted_array):
    distribution = []
    n_gold_labels = len(np.unique(gold_array))
    for predicted_label in np.unique(predicted_array):
        gold_array_labels = gold_array[np.where(predicted_array == predicted_label)]
        distribution.append(np.bincount(gold_array_labels, minlength=n_gold_labels))
    return np.vstack(distribution)


def get_enriched_labels(gold_array, predicted_array, threshold):
    label_distribution = get_label_distribution(gold_array, predicted_array)
    label_enrichment = label_distribution / label_distribution.sum(axis=1, keepdims=True)
    label_order = label_enrichment.argsort()[:, ::-1]
    label_size = np.bincount(np.where(label_enrichment > threshold)[0])
    alignment = {}
    for predicted_label, size in enumerate(label_size):
        label_ids = label_order[predicted_label, range(size)]
        label_values = label_enrichment[predicted_label, label_ids]
        alignment[predicted_label] = list(zip(label_ids, label_values))
    return alignment


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


def p_value(M, n, N, k):
    # M = population size                               (the total number of words in the corpus)
    # n = number of success states in the population    (how many times a particular word w1 appears in the corpus)
    # N = number of draws                               (the size of a particular cluster c1)
    # k = number of observed successes                  (number of instances of w1 in c1)
    hyp = hypergeom(M, n, N)
    return 1 - hyp.cdf(k - 1)  # P(X>=k) = 1 - P(X <= k-1)


def calc_p_value_term_level(vectors, labels):
    n_clusters = np.unique(labels).shape[0]
    n_term_instances = vectors.sum(axis=0).A.ravel()
    n_terms_instances = n_term_instances.sum()
    p_values = []
    for cluster_id in range(n_clusters):
        sample_ids = [sample_id for sample_id, label in enumerate(labels)
                      if label == cluster_id]
        cluster_vectors = vectors[sample_ids, :]
        n_cluster_term_instances = cluster_vectors.sum(axis=0).A.ravel()
        n_cluster_terms_instances = n_cluster_term_instances.sum()
        p_value_cluster_terms = p_value(n_terms_instances,
                                        n_term_instances,
                                        n_cluster_terms_instances,
                                        n_cluster_term_instances)
        p_values.append(p_value_cluster_terms)
    return np.vstack(p_values)


def calc_p_value_doc_level(vectors, labels):
    n_clusters = np.unique(labels).shape[0]
    n_docs = vectors.shape[0]
    n_docs_term = vectors.sum(axis=0).A.ravel()
    p_values = []
    for cluster_id in range(n_clusters):
        sample_ids = [sample_id for sample_id, label in enumerate(labels)
                      if label == cluster_id]
        cluster_vectors = vectors[sample_ids, :]
        n_cluster_docs = cluster_vectors.shape[0]
        n_cluster_docs_term = cluster_vectors.sum(axis=0).A.ravel()
        p_value_cluster_terms = p_value(n_docs,
                                        n_docs_term,
                                        n_cluster_docs,
                                        n_cluster_docs_term)
        p_values.append(p_value_cluster_terms)
    return np.vstack(p_values)


def get_key_terms(vectors, labels, p_value_threshold, top_k):
    # perform document-level analysis. count every token only once per document.
    binary_vectors = csr_matrix((np.ones_like(vectors.data),
                                 vectors.indices, vectors.indptr), shape=vectors.shape)
    p_values = calc_p_value_doc_level(binary_vectors, labels)

    # apply Bonferoni Correction
    n_clusters = np.unique(labels).shape[0]
    bonferoni_factor = n_clusters * vectors.shape[1]
    p_values = p_values * bonferoni_factor

    # find a mapping between a cluster id and its top rated terms based on p-value
    result = {}
    order = p_values.argsort()
    p_values_mask = np.where(p_values < p_value_threshold, True, False)
    for cluster_id in range(p_values.shape[0]):
        top_k_terms = order[cluster_id, :top_k]
        result[cluster_id] = [term_id for term_id, valid in
                              zip(top_k_terms, p_values_mask[cluster_id, top_k_terms]) if valid]
    return result


STOP_WORDS = {'a', 'about', 'all', 'also', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'being', 'been',
              'between', 'both', 'by', "can", "could", "did", "do", "does", "doing", "during", "each", "for", "from",
              "few", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "hers", "herself",
              "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "in", "into", "is", "it",
              "its", "itself", "let", "let's", "lets", "me", "more", "most", "my", "must", "myself", "of", "off", "on",
              "once", "one", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
              "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
              "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
              "they're", "they've", "this", "those", "to", "too", "until", "us", "very", "was", "we", "we'd", "we'll",
              "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
              "who's", "whom", "why", "why's", "will", "with", "would", "you", "you'd", "you'll", "you're", "you've",
              "your", "yours", "yourself", "yourselves",

              # negation
              "not", "no", "nor", "neither"

              # verbs
              "am", "ain't",
              "is", "isn't",
              "are", "aren't",
              "was", "wasn't",
              "were", "weren't",
              "will", "won't", "will've",
              "can", "can't", "cannot",
              "could", "couldn't", "could've",
              "should", "shouldn't", "should've",
              "would", "wouldn't", "would've",
              "might", "mightn't", "might've",
              "must", "mustn't", "must've",
              "may",
              "do", "don't",
              "does", "doesn't",
              "did", "didn't",
              "have", "haven't",
              "has", "hasn't",
              "had", "hadn't",

              # personal pronouns
              "i", "i'm", "i'd", "i've", "i'd", "i'll", "i'd",
              "you", "you're", "you'd", "you've", "you'd", "you'll", "you'd",
              "he", "he's", "he'd", "he'll",
              "she", "she's", "she'd", "she'll",
              "it", "it's", "it'd", "it'll",
              "we", "we're", "we'd", "we've", "we'll",
              "they", "they're", "they'd", "they've", "they'll",
              "somebody", "somebody's", "somebody'd", "somebody'll",
              "someone", "someone's", "someone'd", "someone'll",
              "something", "something's", "something'd", "something'll",

              # interrogative verbs
              "who", "who's", "who're", "who'd", "who've", "who'll",
              "what", "what's", "what're", "what'd", "what've", "what'll",
              "when", "when's", "when're", "when'd", "when've", "when'll",
              "where", "where's", "where're", "where'd", "where've", "where'll",
              "why", "why's", "why're", "why'd", "why've", "why'll",
              "how", "how's", "how're", "how'd", "how've", "how'll",
              "which", "which's", "which're", "which'd", "which've", "which'll",

              # demonstratives
              "this", "this's", "this'd", "this'll",
              "these", "these're", "these'd", "these'll",
              "that", "that's", "that'd", "that'll",
              "those", "those're", "those'd", "those'll",
              "here", "here's", "here'd", "here'll",
              "there", "there's", "there're", "there'd", "there'll",

              # other common contractions
              "gimme", "lemme", "cause", "'cuz", "imma", "gonna", "wanna", "gotta",
              "hafta", "woulda", "coulda", "shoulda", "howdy", "let's", "y'all",

              # general
              "if", "but", "however", "just", "only", "thing", "way", "whole",
              "after", "before", "think", "either",
              }

STOP_WORDS_RE = re.compile(r"\b(" + "|".join(STOP_WORDS) + r")\b")
INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9]")
STEMMER = SnowballStemmer("english", ignore_stopwords=True)


def custom_tokenizer(text):
    text = text.lower()
    text = STOP_WORDS_RE.sub(" ", text)                         # remove full stop words from raw text
    text = INVALID_CHARS_RE.sub(" ", text)                      # remove invalid chars
    words = text.split()                                        # split by white space
    words = [word for word in words if word not in STOP_WORDS]  # remove stop words (after removal of invalid chars)
    words = [word for word in words if len(word) > 1]           # keep only terms that are longer than 1 character
    words = [STEMMER.stem(word) for word in words]              # stem every term
    return words
