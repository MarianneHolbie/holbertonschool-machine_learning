#!/usr/bin/env python3
"""
    Module to calculate the n-gram BLEU score for a sentence
    without nltk
"""
from collections import Counter
import numpy as np


def generate_ngram(sentence, n):
    """
        split list of sentences in n-gram

    :param sentence: sentence to split
    :param n: number of n-gram

    :return: list of formed n-gram
    """
    if n <= 1:
        return sentence

    step = n - 1
    result = sentence[:-step]

    for i in range(len(result)):
        for j in range(step):
            result[i] += ' ' + sentence[i + 1 + j]

    return result


def modified_precision(references, sentence, max_order):
    """
        calculate modified ngram precision

    :param references: list of reference translations
    :param sentence: list containing the model proposed sentence
    :param max_order: size of the n-gram to use for evaluation

    :return: modified precision
    """
    counts = Counter(generate_ngram(sentence, max_order)) if len(sentence) >= max_order else Counter()

    max_counts = {}
    for reference in references:
        ref_counts = (Counter(generate_ngram(reference, max_order)) if len(reference) >= max_order else Counter())

        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), ref_counts[ngram])

    # intersection between hypothesis and reference's count
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    denominator = max(1, len(sentence))

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
    ngram_precisions = []

    for order in range(1, max_order + 1):
        modified_precisions = modified_precision(references, sentence, order)
        ngram_precisions.append(modified_precisions)

    # len reference
    closest_ref_len = min((abs(len(ref) - len_sentence), len(ref)) for ref in references)[1]

    # BP
    if len_sentence > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / len_sentence)

    bleu_score = BP * np.exp(sum(np.log(p) for p in ngram_precisions) / max_order)

    return bleu_score
