import os
import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from scipy.sparse import csr_matrix
from sib import SIB


base_dir = os.path.dirname(os.path.abspath(__file__))

setups = {
    'baseline':      {'n_cls': 20, 'n_jobs': 1, 'n_init': 1, 'max_iter': 8,
                      'tol': 0.05, 'uniform': True, 'sparse': True, 'opt': 'B'},
    'n_init':        {'n_cls': 20, 'n_jobs': 1, 'n_init': 8, 'max_iter': 8,
                      'tol': 0.05, 'uniform': True, 'sparse': True, 'opt': 'C'},
    'n_jobs':        {'n_cls': 20, 'n_jobs': -1, 'n_init': 8, 'max_iter': 8,
                      'tol': 0.05, 'uniform': True, 'sparse': True, 'opt': 'C'},
    'max_iter':      {'n_cls': 20, 'n_jobs': 1, 'n_init': 1, 'max_iter': 3,
                      'tol': 0.02, 'uniform': True, 'sparse': True, 'opt': 'C'},
    'tol':           {'n_cls': 20, 'n_jobs': 1, 'n_init': 1, 'max_iter': 8,
                      'tol': 0.10, 'uniform': True, 'sparse': True, 'opt': 'B'},
    'uniform_prior': {'n_cls': 20, 'n_jobs': 1, 'n_init': 1, 'max_iter': 8,
                      'tol': 0.05, 'uniform': False, 'sparse': True, 'opt': 'C'},
    'float':         {'n_cls': 20, 'n_jobs': 1, 'n_init': 1, 'max_iter': 8,
                      'tol': 0.05, 'uniform': False, 'sparse': True, 'opt': 'C', 'type': 'float'},
    'dense':          {'n_cls': 20, 'n_jobs': 1, 'n_init': 1, 'max_iter': 8,
                       'tol': 0.05, 'uniform': True, 'sparse': False, 'opt': 'B'},
}

equal_refs = [['baseline', 'dense'], ['n_init', 'n_jobs'], ['uniform_prior', 'float']]

ami_values = {
    'baseline': 0.479308645259772,
    'n_init': 0.4839664852243509,           # higher than baseline because uses more inits sequentially
    'n_jobs': 0.4839664852243509,           # should be identical to n_init
    'max_iter': 0.4054343178732652,         # less than baseline because less iterations
    'tol': 0.42316606323860384,             # less than baseline because higher tol
    'uniform_prior': 0.4833358896883293,    # normally higher than baseline
    'float': 0.4833358896883293,            # should be identical to uniform_prior
    'dense': 0.479308645259772              # should be identical to baseline
}

vectors_path = os.path.join(base_dir, 'resources', 'vectors')

gold_labels_path = os.path.join(base_dir, 'resources', 'gold_labels')

random_state = 527802

max_features = 2500


def get_names():
    return setups.keys()


def create(setup_name, optimizer=None):
    setup = setups[setup_name]
    n_clusters = setup['n_cls']
    n_jobs = setup['n_jobs']
    n_init = setup['n_init']
    max_iter = setup['max_iter']
    tol = setup['tol']
    uniform_prior = setup['uniform']
    optimizer = setup['opt'] if optimizer is None else optimizer
    return SIB(n_clusters=n_clusters, n_jobs=n_jobs, n_init=n_init,
               max_iter=max_iter, tol=tol, uniform_prior=uniform_prior,
               random_state=random_state, optimizer_type=optimizer, verbose=True)


def is_sparse(setup_name):
    setup = setups[setup_name]
    return setup['sparse']


def is_float(setup_name):
    setup = setups[setup_name]
    return 'type' in setup and setup['type'] == 'float'


def get_path(setup_name):
    return os.path.join(base_dir, 'resources', setup_name + '_ref')


def exists(setup_name):
    return os.path.exists(get_path(setup_name))


def save(setup_name, labels, costs, score):
    setup_path = get_path(setup_name)
    os.makedirs(setup_path)
    np.savez_compressed(os.path.join(setup_path, "data.npz"),
                        labels=labels, costs=costs, score=score)


def load(setup_name):
    setup_path = get_path(setup_name)
    data = np.load(os.path.join(setup_path, "data.npz"))
    return data['labels'], data['costs'], data['score']


def generate(setup_name, vectors):
    if not exists(setup_name):
        print("Generating reference for: %s" % setup_name)
        if not is_sparse(setup_name):
            vectors = vectors.toarray()
        if is_float(setup_name):
            vectors = vectors.astype(np.float)
        sib = create(setup_name, optimizer='B')
        sib.fit(vectors)
        save(setup_name, sib.labels_, sib.costs_, sib.score_)
    else:
        print("Reference already exists for " + setup_name)


def verify(setup_name, vectors):
    if exists(setup_name):
        ref_labels, ref_costs, ref_score = load(setup_name)
        if not is_sparse(setup_name):
            vectors = vectors.toarray()
        sib = create(setup_name)
        sib.fit(vectors)
        assert np.allclose(ref_labels, sib.labels_)
        assert np.allclose(ref_costs, sib.costs_)
        assert np.allclose(ref_score, sib.score_)


def vectorize_20ng():
    dataset = fetch_20newsgroups(subset='train', categories=None,
                                 shuffle=True, random_state=256)
    vectorizer = CountVectorizer(max_features=max_features)
    return vectorizer.fit_transform(dataset.data), dataset.target


def load_vectors():
    data = np.load(os.path.join(vectors_path, "data.npz"))
    return csr_matrix((data['data'], data['indices'], data['indptr']))


@pytest.fixture
def supply_vectors():
    return load_vectors()


def generate_references():
    vectors, gold_labels = vectorize_20ng()
    for setup_name in setups.keys():
        generate(setup_name, vectors)
    if not os.path.exists(vectors_path):
        os.makedirs(vectors_path)
        print("Saving vectors to: %s" % vectors_path)
        np.savez_compressed(os.path.join(vectors_path, "data.npz"),
                            data=vectors.data, indptr=vectors.indptr, indices=vectors.indices)
    else:
        print("Vectors dump is already generated")

    if not os.path.exists(gold_labels_path):
        os.makedirs(gold_labels_path)
        print("Saving gold labels to: %s" % gold_labels_path)
        np.savez_compressed(os.path.join(gold_labels_path, "gold_labels.npz"), labels=gold_labels)
    else:
        print("Vectors dump is already generated")

    # verification that setups that are expected to give equal result really do so
    for setup_list in equal_refs:
        if len(setup_list) > 0:
            first_labels, first_costs, first_score = load(setup_list[0])
            for setup_id, setup_name in enumerate(setup_list):
                if setup_id > 0:
                    current_labels, current_costs, current_score = load(setup_name)
                    assert np.allclose(first_labels, current_labels)
                    assert np.allclose(first_costs, current_costs)
                    assert np.allclose(first_score, current_score)
    print("Verified equal result for marked tests")

    for setup, ref_ami in ami_values.items():
        labels, _, _ = load(setup)
        ami = metrics.adjusted_mutual_info_score(gold_labels, labels)
        assert np.isclose(ami, ref_ami)
    print("Verified ami scores")


@pytest.mark.parametrize("setup_name", list(setups.keys()))
def test_setups(supply_vectors, setup_name):
    vectors = supply_vectors
    verify(setup_name, vectors)
