import random
import sys

from sib import SIB
from sklearn.feature_extraction.text import CountVectorizer

from eval_datasets import fetch_20ng, fetch_bbc_news, fetch_dbpedia, fetch_ag_news, fetch_yahoo_answers
from ex_workflow import ExampleWorkflow

# random seed for deterministic results
random.seed(1024)

# example name
EX_NAME = 'ex4'

# determine the number of runs per algorithm
N_RUNS = 2

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
}

# the algorithms: {short name: (display name, factory, optional params)}
ALGORITHMS = {
    'sib':    ('sIB <VERSION>',     SIB,     {}),
}

# the setups to run: (vectorizer name, vectors manipulator name, algorithms list)
SETUPS = [
    ('sib',              'tf'),
]

DATASET_MAX_SIZE = None

ALGORITHM_VIEW_ORDER = ['sib']
EMBEDDING_VIEW_ORDER = ['tf']

workflow = ExampleWorkflow(ex_name=EX_NAME, datasets=DATASETS, embeddings=EMBEDDINGS,
                           algorithms=ALGORITHMS, setups=SETUPS, dataset_max_size=DATASET_MAX_SIZE, n_runs=N_RUNS,
                           algorithm_view_order=ALGORITHM_VIEW_ORDER, embedding_view_order=EMBEDDING_VIEW_ORDER,
                           seed=1024)

mode = sys.argv[1]

if mode == 'prepare':
    workflow.prepare()
elif mode == 'cluster':
    workflow.cluster()
elif mode == 'evaluate':
    workflow.evaluate()
