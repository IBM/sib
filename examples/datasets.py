import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


def fetch_20ng():
    cat = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'rec.motorcycles',
    ]
    cat = None
    return fetch_20newsgroups(subset='all', categories=cat,
                              shuffle=True, random_state=256)


def fetch_dbpedia():
    path = os.path.join('datasets', 'dbpedia')
    train_path = os.path.join(path, 'train.zip')
    classes_path = os.path.join(path, 'classes.txt')
    df_train = pd.read_csv(train_path, names=['class_id', 'title', 'text'])
    df_classes = pd.read_csv(classes_path, names=['class_name'])
    data = Bunch()
    data.target = np.array(df_train['class_id'].tolist())
    data.data = df_train['text'].tolist()
    data.target_names = df_classes['class_name'].tolist()
    return data


def fetch_agnews():
    path = os.path.join('datasets', 'ag_news')
    train_path = os.path.join(path, 'train.zip')
    classes_path = os.path.join(path, 'classes.txt')
    df_train = pd.read_csv(train_path, names=['class_id', 'title', 'text'])
    df_classes = pd.read_csv(classes_path, names=['class_name'])
    data = Bunch()
    data.target = np.array(df_train['class_id'].tolist())
    data.data = [text.replace('\\', ' ') for text in df_train['text'].tolist()]
    data.target_names = df_classes['class_name'].tolist()
    return data


def fetch_bbc():
    path = os.path.join('datasets', 'bbc')
    data = Bunch()
    target_nanes = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    texts = []
    class_ids = []
    for target_nane_id, target_nane in enumerate(target_nanes):
        target_nane_path = os.path.join(path, target_nane)
        for doc_file in os.listdir(target_nane_path):
            with open(os.path.join(target_nane_path, doc_file), 'rt') as file:
                texts.append(file.read().replace('\n', ' '))
                class_ids.append(target_nane_id)
    data.target = np.array(class_ids)
    data.data = texts
    data.target_names = target_nanes
    return data
