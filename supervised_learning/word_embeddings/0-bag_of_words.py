#!/usr/bin/env python3
"""
    Bag Of Words
"""
from sklearn.feature_extraction.text import CountVectorizer


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

    vectoriz = CountVectorizer(vocabulary=vocab)
    bow = vectoriz.fit_transform(sentences)

    features = vectoriz.get_feature_names_out()

    return bow.toarray(), features




