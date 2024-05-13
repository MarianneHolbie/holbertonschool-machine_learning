#!/usr/bin/env python3
"""
    TF-IDF
"""
import re
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

    if not isinstance(sentences, list):
        raise TypeError("sentences should be a list.")

    preprocessed_sentences = []
    for sentence in sentences:
        preprocessed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
        preprocessed_sentences.append(preprocessed_sentence)

    # extract features : list of words
    list_words = []
    for sentence in preprocessed_sentences:
        words = re.findall(r'\w+', sentence)
        list_words.extend(words)

    if vocab is None:
        vocab = sorted(set(list_words))

    tfidf_vect = TfidfVectorizer(vocabulary=vocab)

    tfidf_matrix = tfidf_vect.fit_transform(sentences)
    features = tfidf_vect.get_feature_names()

    return tfidf_matrix.toarray(), features
