import random

from sib import SIB
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from eval_datasets import fetch_20ng, fetch_bbc_news, fetch_dbpedia, fetch_ag_news, fetch_yahoo_answers
from vectorizers import GloveVectorizer, SBertVectorizer

# random seed for deterministic results
random.seed(1024)

# determine the number of runs per algorithm and the number of random inits per run
N_RUNS = 10
N_INIT = 10

# the datasets to use: {short name: (full name, retriever)}
DATASETS = {
    '20ng':     ('20 News Groups', fetch_20ng),
    'dbpedia':  ('DBPedia',        fetch_dbpedia),
    'bbc_news': ('BBC News',       fetch_bbc_news),
    'ag_news':  ('AG NEWS',        fetch_ag_news),
    'yahoo':    ('Yahoo! Answers', fetch_yahoo_answers)
}

# the text vectorizers to use: {short name: (name, factory, optional params)}
EMBEDDINGS = {
    'tf':    ('TF',     CountVectorizer, {'max_features': [10000], 'stop_words': ['english']}),
    'tfidf': ('TF/IDF', TfidfVectorizer, {'max_features': [10000], 'stop_words': ['english']}),
    'glove': ('GloVe',  GloveVectorizer, {}),
    'sbert': ('S-Bert', SBertVectorizer, {}),
}

# the algorithms: {short name: (display name, factory, optional params)}
ALGORITHMS = {
    'sib':    ('sIB',     SIB,     {}),
    'kmeans': ('K-Means', KMeans,  {}),
}

# the setups to run: (vectorizer name, vectors manipulator name, algorithms list)
SETUPS = [
    ('sib',              'tf'),
    ('kmeans',           'tf'),
    ('kmeans',           'tfidf'),
    ('kmeans',           'glove'),
    ('kmeans',           'sbert'),
]

DATASET_MAX_SIZE = None

HIDDEN_PARAMS = ['n_clusters']

EX_NAME = 'ex3'

ALGORITHM_VIEW_ORDER = ['kmeans', 'sib']
EMBEDDING_VIEW_ORDER = ['tf', 'tfidf', 'glove', 'sbert']
