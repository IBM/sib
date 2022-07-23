import re

from nltk import SnowballStemmer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS


STOP_WORDS = ENGLISH_STOP_WORDS
STOP_WORDS_RE = re.compile(r"\b(" + "|".join([re.escape(x) for x in STOP_WORDS]) + r")\b")
INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9]")
STEMMER = SnowballStemmer("english", ignore_stopwords=True)


def custom_tokenizer(text):
    text = text.lower()
    text = STOP_WORDS_RE.sub(" ", text)                         # remove full stop words from raw text
    text = INVALID_CHARS_RE.sub(" ", text)                      # remove invalid chars
    words = text.split()                                        # split by white space
    words = [word for word in words if word not in STOP_WORDS]  # remove stop words (after removal of invalid chars)
    words = [word for word in words if len(word) > 1]           # keep only terms that are longer than 1 character
    words = [STEMMER.stem(word) for word in words]              # stem every term
    return words
