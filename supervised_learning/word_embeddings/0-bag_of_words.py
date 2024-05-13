#!/usr/bin/env python3
"""
    Bag Of Words
"""
import numpy as np
from gensim import corpora
from gensim.utils import simple_preprocess


def bag_of_words(sentences, vocab=None):
    """
        creates a bag of words embedding matrix

    :param sentences: list of sentences to analyse
    :param vocab: list of vocabulary words to use for the analysis
        if None: all words within sentences should be used

    :return: embeddings, features
        embeddings: ndarray, shape(s,f) containing embeddings
            s: number of sentences in sentences
            f: number of features analysed
        features: list of the features used for embeddings
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences should be a list.")

    # Creation list of features
    sentences_tokenized = [simple_preprocess(sentence) for sentence in sentences]
    dict_word = corpora.Dictionary(sentences_tokenized)

    if vocab is None:
        list_word = []
        for k, _ in dict_word.token2id.items():
            list_word.append(k)
        features = sorted(list_word)
    else:
        features = vocab

    # construct incorporation matrix
    embeddings = np.zeros((len(sentences_tokenized), len(features)), dtype=int)

    for i, sentence in enumerate(sentences_tokenized):
        for j, word in enumerate(features):
            if word in sentence:
                embeddings[i, j] = 1

    return embeddings, features
