import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import contingency_matrix


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
