#!/usr/bin/env python3
"""
    TF-IDF
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
            creates a TF-IDF embedding

        :param sentences: list of sentences to analyse
        :param vocab: list of vocabulary words to use for the analysis
            if None: all words within sentences should be used

        :return: embeddings, features
            embeddings: ndarray, shape(s,f) containing embeddings
                s: number of sentences in sentences
                f: number of features analysed
            features: list of the features used for embeddings
    """
    vectoriz = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectoriz.fit_transform(sentences)

    features = vectoriz.get_feature_names_out()

    return embeddings.toarray(), features
