import random
from typing import Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch

from datasets import load_dataset, Split


def fetch_20ng(max_size=None, subset='all') -> Bunch:
    dataset = fetch_20newsgroups(subset=subset, categories=None,
                                 shuffle=True, random_state=256)
    return prepare_dataset(dataset, max_size)


def fetch_dbpedia(max_size=None) -> Bunch:
    return fetch_hf_dataset('dbpedia_14', max_size,  text_column='content')


def fetch_ag_news(max_size=None) -> Bunch:
    return fetch_hf_dataset('ag_news', max_size)


def fetch_bbc_news(max_size=None) -> Bunch:
    return fetch_hf_dataset('SetFit/bbc-news', max_size)


def fetch_yahoo_answers(max_size=None) -> Bunch:
    return fetch_hf_dataset('yahoo_answers_topics', max_size, label_column='topic',
                            text_column=['question_title', 'question_content', 'best_answer'])


def fetch_hf_dataset(name, max_size, label_column: str = 'label', text_column: Union[str, list] = 'text') -> Bunch:
    hf_dataset = load_dataset(name, split=Split.ALL)
    df = pd.DataFrame(hf_dataset)

    dataset = Bunch()
    dataset.target = df[label_column].to_numpy()
    if isinstance(text_column, list):
        dataset.data = df[text_column].apply(lambda x: ' '.join(x.values), axis=1).to_list()
    else:
        dataset.data = df[text_column].to_list()
    if 'label_text' in df.columns:
        dataset.target_names = df.groupby('label').agg(
            label_text=('label_text', lambda x: list(set(x))[0])).sort_index()['label_text'].to_list()
    else:
        dataset.target_names = hf_dataset.features[label_column].names
    return prepare_dataset(dataset, max_size)


def prepare_dataset(dataset: Bunch, max_size: Union[int, None]) -> Bunch:
    # add size information
    dataset.n_clusters = np.unique(dataset.target).shape[0]
    dataset.n_samples = len(dataset.target)
    dataset.n_samples_org = dataset.n_samples

    # reduce the dataset if it's too large
    if max_size and dataset.n_samples > max_size:
        print(f"Using a sample of {max_size} out of {dataset.n_samples}")
        data, target = zip(*random.sample(list(
            zip(dataset.data, dataset.target)), max_size))
        target = np.array(target)
        new_dataset = Bunch()
        new_dataset.data = data
        new_dataset.target = target
        new_dataset.target_names = dataset.target_names
        new_dataset.n_clusters = np.unique(new_dataset.target).shape[0]
        new_dataset.n_samples = len(new_dataset.target)
        new_dataset.n_samples_org = dataset.n_samples
        dataset = new_dataset

    word_count = np.array([len(sample.split()) for sample in dataset.data])
    dataset.word_count_mean = np.mean(word_count)
    dataset.word_count_std = np.std(word_count)
    dataset.word_count_median = np.median(word_count)

    # offset the class ids to 0 if it needs to
    dataset.target -= np.min(dataset.target)

    return dataset


if __name__ == '__main__':
    fetch_ag_news(40000)
