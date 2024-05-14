#!/usr/bin/env python3
"""
    Module to calculate the n-gram BLEU score for a sentence
    without nltk
"""
from collections import Counter
import numpy as np


def generate_ngram(sentence, order):
    """
        genrate n-grams for all orders from 1 to max_order

    :param sentence: sentence to split
    :param order: number of n-gram

    :return: list of formed n-gram
    """
    ngrams = []
    for i in range(len(sentence) - order + 1):
        ngram = sentence[i: i + order]
        ngrams.append(' '.join(ngram))

    return ngrams


def modified_precision(references, sentence, order):
    """
        calculate modified ngram precision

    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :param order: size of the n-gram to use for evaluation

    :return: modified precision
    """
    sentence_ngrams = Counter(generate_ngram(sentence, order))

    if not sentence_ngrams:
        return 0

    max_counts = {}
    for reference in references:
        ref_ngrams = Counter(generate_ngram(reference, order))
        for ngram in sentence_ngrams:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    ref_ngrams[ngram])

    # intersection between hypothesis and reference's count
    clipped_counts = {
        ngram: min(count, max_counts.get(ngram, 0))
        for ngram, count in sentence_ngrams.items()
    }

    numerator = sum(clipped_counts.values())
    denominator = max(1, len(sentence) - order + 1)

    return numerator / denominator


def ngram_bleu(references, sentence, max_order):
    """
        calculates n-gram BLEU score for a sentence

    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :param max_order: size of the n-gram to use for evaluation

    :return: n-gram BLEU score
    """

    len_sentence = len(sentence)
    ngram_precisions = [modified_precision(references, sentence, order)
                        for order in range(1, max_order + 1)]

    # BP
    min_ref_length = min(len(ref) for ref in references)
    if len_sentence >= min_ref_length:
        BP = 1
    else:
        BP = np.exp(1 - min_ref_length / len_sentence)

    bleu_score = BP * np.exp(sum(np.log(p) for p in ngram_precisions)
                             / max_order)

    return bleu_score
