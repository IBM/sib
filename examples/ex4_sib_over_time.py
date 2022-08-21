import random
import sys

from sib import SIB
from sklearn.feature_extraction.text import CountVectorizer

from eval_datasets import fetch_20ng, fetch_bbc_news, fetch_dbpedia, fetch_ag_news, fetch_yahoo_answers
from ex_workflow import ExampleWorkflow

# random seed for deterministic results
random.seed(1024)

workflow = ExampleWorkflow(
    ex_name='ex4',
    datasets={
        '20ng':     ('20 News Groups', fetch_20ng),
        'dbpedia':  ('DBPedia',        fetch_dbpedia),
        'bbc_news': ('BBC News',       fetch_bbc_news),
        'ag_news':  ('AG NEWS',        fetch_ag_news),
        'yahoo':    ('Yahoo! Answers', fetch_yahoo_answers)
    },
    embeddings={
        'tf':    ('TF',     CountVectorizer, {'max_features': [10000], 'stop_words': ['english']}),
    },
    algorithms={
        'sib':    ('sIB $VERSION',     SIB,     {}),
    },
    setups=[
        ('sib',              'tf'),
    ],
    dataset_max_size=None,
    n_runs=10,
    algorithm_view_order=['sib'],
    embedding_view_order=['tf'],
    seed=1024)

mode = sys.argv[1]

if mode == 'prepare':
    workflow.prepare()
elif mode == 'cluster':
    workflow.cluster()
elif mode == 'evaluate':
    workflow.evaluate()
