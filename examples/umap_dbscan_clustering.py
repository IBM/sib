from umap.umap_ import UMAP
from sklearn.cluster import DBSCAN


class UMapDBScan:

    def __init__(self, n_clusters, random_state):
        self.reducer = UMAP(n_components=5)
        self.dbscan = DBSCAN(eps=0.4, min_samples=2)
        self.labels_ = None

    def fit(self, x):
        reduced = self.reducer.fit_transform(x)
        self.dbscan.fit(reduced)
        self.labels_ = self.dbscan.labels_
