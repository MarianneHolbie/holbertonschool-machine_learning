#!/usr/bin/env python3
"""
    Bag Of Words
"""
import numpy as np
import re


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

    # extract features : list of words
    list_words = []
    for sentence in sentences:
        words = re.findall(r'\w+', sentence.lower())
        list_words.extend(words)

    if vocab is None:
        vocab = sorted(set(list_words))

    # construct incorporation matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    features = vocab

    for i, sentence in enumerate(sentences):
        words = re.findall(r'\w+', sentence.lower())
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, features
