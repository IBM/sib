import random

from sib import SIB
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from eval_datasets import fetch_20ng, fetch_bbc_news, fetch_dbpedia, fetch_ag_news, fetch_yahoo_answers
from ex_workflow import ExampleWorkflow
from umap_dbscan_clustering import UMapDBScan
from vectorizers import SBertVectorizer, SBertHQVectorizer

# random seed for deterministic results
random.seed(1024)

# example name
EX_NAME = 'ex5'

# determine the number of runs per algorithm
N_RUNS = 2

# the datasets to use: {short name: (full name, retriever)}
DATASETS = {
    # '20ng':     ('20 News Groups', fetch_20ng),
    # 'dbpedia':  ('DBPedia',        fetch_dbpedia),
    'bbc_news': ('BBC News',       fetch_bbc_news),
    # 'ag_news':  ('AG NEWS',        fetch_ag_news),
    # 'yahoo':    ('Yahoo! Answers', fetch_yahoo_answers)
}

# the text vectorizers to use: {short name: (name, factory, optional params)}
EMBEDDINGS = {
    'tf':    ('TF',     CountVectorizer, {'max_features': [10000], 'stop_words': ['english']}),
    'sbert': ('S-Bert', SBertVectorizer, {}),
    'sbert_hq': ('S-Bert HQ', SBertHQVectorizer, {}),
}

# the algorithms: {short name: (display name, factory, optional params)}
ALGORITHMS = {
    'sib':    ('sIB',     SIB,          {}),
    'kmeans': ('K-Means', KMeans,       {}),
    'dbscan': ('DBScan',  UMapDBScan,   {})
}

# the setups to run: (vectorizer name, vectors manipulator name, algorithms list)
SETUPS = [
    ('sib',     'tf'),
    ('kmeans',  'sbert'),
    ('kmeans',  'sbert_hq'),
    ('dbscan',  'sbert'),
    ('dbscan',  'sbert_hq'),
]

DATASET_MAX_SIZE = None

ALGORITHM_VIEW_ORDER = ['kmeans', 'dbscan', 'sib']
EMBEDDING_VIEW_ORDER = ['tf', 'sbert', 'sbert_hq']

workflow = ExampleWorkflow(ex_name=EX_NAME, datasets=DATASETS, embeddings=EMBEDDINGS,
                           algorithms=ALGORITHMS, setups=SETUPS, dataset_max_size=DATASET_MAX_SIZE, n_runs=N_RUNS,
                           algorithm_view_order=ALGORITHM_VIEW_ORDER, embedding_view_order=EMBEDDING_VIEW_ORDER,
                           seed=1024)

workflow.prepare()
workflow.cluster()
workflow.evaluate()
