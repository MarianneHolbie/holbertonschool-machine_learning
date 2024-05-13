#!/usr/bin/env python3
"""
    Train fastText model
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """
        creates and trains a gensim fastText model

    :param sentences: list of sentences to be trained on
    :param size: dimensionality of the embedding layer
    :param min_count: minimum count of occurrences word for use in training
    :param negative: size of negative sampling
    :param window:  maximum distance btw current and predicted word
        in a sentence
    :param cbow: True=cbow, False=skip-gram
    :param iterations: number of iterations to train over
    :param seed: seed for the random number generator
    :param workers: number of worker threads to train the model

    :return: trained model
    """
    if cbow is True:
        sg = 0
    else:
        sg = 1
    model = FastText(sentences=sentences,
                     sg=sg,
                     size=size,
                     negative=negative,
                     window=window,
                     min_count=min_count,
                     seed=seed,
                     workers=workers,
                     iter=iterations)
    return model
