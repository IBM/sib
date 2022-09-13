import umap
from sklearn.cluster import DBSCAN


class UMapDBScan:

    def __init__(self):
        self.reducer = umap.UMAP(n_components=5)
        self.dbscan = DBSCAN(eps=0.4, min_samples=2)
        self._labels = None

    def fit(self, x):
        reduced = self.reducer.fit_transform(x)
        self.dbscan.fit(reduced)
        self._labels = self.dbscan.labels_.max()
