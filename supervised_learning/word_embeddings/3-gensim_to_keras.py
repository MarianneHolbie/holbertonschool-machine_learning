#!/usr/bin/env python3
"""
    Extract Word2Vec
"""
from gensim.models import Word2Vec
from keras.layers import Embedding


def gensim_to_keras(model):
    """
        converts a gensim word2vec model to a keras Embedding layer

    :param model: trained gensim word2vec model
    :return: trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
