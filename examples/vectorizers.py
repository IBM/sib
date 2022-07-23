import sys
import os
import re
import zipfile

import numpy as np
from tqdm import tqdm
import wget
from sklearn.feature_extraction import text as sklearn_text

from sentence_transformers import SentenceTransformer


class SBertVectorizer:

    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def fit_transform(self, texts):
        return self.encoder.encode(texts)


GLOVE_MODEL = 'glove.840B.300d'
GLOVE_ZIP_FILE = GLOVE_MODEL + '.zip'
GLOVE_TXT_FILE = GLOVE_MODEL + '.txt'
GLOVE_VOCAB_FILE = GLOVE_MODEL + '.vocab'
GLOVE_VECTORS_FILE = GLOVE_MODEL + '.npy'

GLOVE_URL = 'https://huggingface.co/stanfordnlp/glove/resolve/main/' + GLOVE_ZIP_FILE

GLOVE_DIR = os.path.join('download', 'glove')


class GloveVectorizer:
    """
    Based on https://blog.ekbana.com/loading-glove-pre-trained-word-embedding-model-from-text-file-faster-5d3e8f2b8455
    """
    def __init__(self):
        vocab_full_path = os.path.join(GLOVE_DIR, GLOVE_VOCAB_FILE)
        vectors_full_path = os.path.join(GLOVE_DIR, GLOVE_VECTORS_FILE)
        self.embeddings = self.load_embeddings(vocab_full_path, vectors_full_path)
        self.empty_vector = np.zeros_like(next(iter(self.embeddings.items()))[1])
        self.stop_words_re = re.compile(r"\b(" + "|".join(sklearn_text.ENGLISH_STOP_WORDS) + r")\b")
        self.invalid_chars_re = re.compile(r"[^A-Za-z0-9]")

    @staticmethod
    def download_model():
        if not os.path.exists(GLOVE_DIR):
            os.makedirs(GLOVE_DIR, exist_ok=True)

            def bar_progress(current, total, _):
                progress_message = "Downloading %s: %d%% [%d / %d] bytes" % (GLOVE_MODEL, current / total * 100,
                                                                             current, total)
                # Don't use print() as it will print in new line every time.
                sys.stdout.write("\r" + progress_message)
                sys.stdout.flush()

            # download
            wget.download(GLOVE_URL, out=GLOVE_DIR, bar=bar_progress)

            # unzip
            print("Unzipping model file")
            zip_full_path = os.path.join(GLOVE_DIR, GLOVE_ZIP_FILE)
            with zipfile.ZipFile(zip_full_path, 'r') as zip_ref:
                zip_ref.extractall(GLOVE_DIR)
            os.remove(zip_full_path)

            # convert to binary
            txt_full_path = os.path.join(GLOVE_DIR, GLOVE_TXT_FILE)
            vocab_full_path = os.path.join(GLOVE_DIR, GLOVE_VOCAB_FILE)
            vectors_full_path = os.path.join(GLOVE_DIR, GLOVE_VECTORS_FILE)
            GloveVectorizer.convert_to_binary(txt_full_path, vocab_full_path, vectors_full_path)
            os.remove(txt_full_path)

    @staticmethod
    def convert_to_binary(txt_file_path, vocab_file_path, bin_file_path):
        with open(txt_file_path, 'rt', encoding='utf-8') as f:
            wv = []
            with open(vocab_file_path, "wt", encoding='utf-8') as vocab_write:
                count = 0
                for line in tqdm(f, desc='Converting to binary format'):
                    space_index = line.index(' ')
                    vocab_write.write(line[0:space_index] + '\n')
                    wv.append([float(val) for val in line[space_index:].split()])
                count += 1
        np.save(bin_file_path, np.array(wv))

    @staticmethod
    def load_embeddings(vocab_file_path, bin_file_path):
        with open(vocab_file_path, 'rt', encoding='utf-8') as f_in:
            index2word = [line.strip() for line in f_in]
        wv = np.load(bin_file_path)
        word_embedding_map = {}
        for i, w in enumerate(index2word):
            word_embedding_map[w] = wv[i]
        return word_embedding_map

    def fit_transform(self, texts):
        vectors = []
        for text in texts:
            text = text.lower()
            text = self.stop_words_re.sub(" ", text)     # remove full stop words from raw text
            text = self.invalid_chars_re.sub(" ", text)  # remove invalid chars
            words = text.split()                         # split by white space
            word_vectors = [self.embeddings.get(word, self.empty_vector) for word in words]
            if len(word_vectors) == 0:
                vectors.append(self.empty_vector)
            else:
                vectors.append(np.sum(word_vectors, axis=0) / len(word_vectors))
        return np.array(vectors)
