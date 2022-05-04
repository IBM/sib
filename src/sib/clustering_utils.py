import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom


def p_value(M, n, N, k):
    # M = population size                               (the total number of words in the corpus)
    # n = number of success states in the population    (how many times a particular word w1 appears in the corpus)
    # N = number of draws                               (the size of a particular cluster c1)
    # k = number of observed successes                  (number of instances of w1 in c1)
    hyp = hypergeom(M, n, N)
    return 1 - hyp.cdf(k - 1)  # P(X>=k) = 1 - P(X <= k-1)


def calc_p_value_term_level(vectors, clusters):
    n_term_instances = vectors.sum(axis=0).A.ravel()
    n_terms_instances = n_term_instances.sum()
    p_values = []
    for cluster_id, sample_ids in clusters.items():
        cluster_vectors = vectors[sample_ids, :]
        n_cluster_term_instances = cluster_vectors.sum(axis=0).A.ravel()
        n_cluster_terms_instances = n_cluster_term_instances.sum()
        p_value_cluster_terms = p_value(n_terms_instances,
                                        n_term_instances,
                                        n_cluster_terms_instances,
                                        n_cluster_term_instances)
        p_values.append(p_value_cluster_terms)
    return np.vstack(p_values)


def calc_p_value_doc_level(vectors, clusters):
    n_docs = vectors.shape[0]
    n_docs_term = vectors.sum(axis=0).A.ravel()
    p_values = []
    for cluster_id, sample_ids in clusters.items():
        cluster_vectors = vectors[sample_ids, :]
        n_cluster_docs = cluster_vectors.shape[0]
        n_cluster_docs_term = cluster_vectors.sum(axis=0).A.ravel()
        p_value_cluster_terms = p_value(n_docs,
                                        n_docs_term,
                                        n_cluster_docs,
                                        n_cluster_docs_term)
        p_values.append(p_value_cluster_terms)
    return np.vstack(p_values)


def get_clusters(labels):
    sorted_idx = np.argsort(labels, kind="stable")
    counts = np.bincount(labels)
    split_idx = np.split(sorted_idx, np.cumsum(counts[:-1]))
    return {n: indices.tolist() for n, indices in enumerate(split_idx)}


def get_binary_vectors(vectors):
    if isinstance(vectors, csr_matrix):
        if not all(vectors.data == 1):
            binary_vectors = csr_matrix((np.ones_like(vectors.data),
                                         vectors.indices, vectors.indptr), shape=vectors.shape)
        else:
            binary_vectors = vectors
    elif isinstance(vectors, np.ndarray):
        if not all(vectors == 1):
            binary_vectors = np.where(vectors != 0, 1, 0)
        else:
            binary_vectors = vectors
    else:
        raise ValueError("Unexpected vectors type")
    return binary_vectors


def get_key_terms(vectors, clusters, p_value_threshold, top_k):
    # perform document-level analysis. count every token only once per document.
    binary_vectors = get_binary_vectors(vectors)
    p_values = calc_p_value_doc_level(binary_vectors, clusters)

    # apply Bonferoni Correction
    bonferoni_factor = len(clusters) * vectors.shape[1]
    p_values *= bonferoni_factor
    order = p_values.argsort()
    p_values_mask = np.where(p_values < p_value_threshold, True, False)

    # find a mapping between a cluster id and its top-rated terms based on p-value
    result = {}
    for cluster_id in range(p_values.shape[0]):
        top_k_terms = order[cluster_id, :top_k]
        result[cluster_id] = [term_id for term_id, valid in
                              zip(top_k_terms, p_values_mask[cluster_id, top_k_terms]) if valid]
    return result


def get_key_texts(vectors, clusters, key_terms, top_k):
    binary_vectors = get_binary_vectors(vectors)
    result = {}
    for cluster_id, cluster_key_terms in key_terms.items():
        vectors_slice = binary_vectors[clusters[cluster_id], :][:, cluster_key_terms].toarray().sum(axis=1)
        ordered_texts_ids = np.argsort(vectors_slice)[-top_k:][::-1]
        result[cluster_id] = [clusters[cluster_id][x] for x in ordered_texts_ids]
    return result


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
