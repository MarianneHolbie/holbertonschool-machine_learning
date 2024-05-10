#!/usr/bin/env python3
"""
    Module to calculate the unigram BLEU score for a sentence
    without nltk
"""
from collections import Counter

import numpy as np


def uni_bleu(references, sentence):
    """
        function that calculates the unigram BLEU score
        for a sentence

    :param references: list of reference translations
                - each reference translation is a list of the words
                in the translation
    :param sentence: list containing the model proposed sentence

    :return: unigram BLEU score
    """

    len_sentence = len(sentence)
    sentence_counts = Counter(sentence)

    # most close length
    closest_ref_len = min((abs(len(ref) - len_sentence), len(ref)) for ref in references)[1]

    # calculate BP:
    if len_sentence > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / len_sentence)

    # count words in each reference sentence
    ref_counts = []
    for ref in references:
        ref_counts.append(Counter(ref))

    modified_precision = sum(min(ref_counts[word], sentence_counts[word]) for word in sentence_counts) / len_sentence

    # calculate bleu unigram score
    bleu_score = BP * modified_precision

    return bleu_score
