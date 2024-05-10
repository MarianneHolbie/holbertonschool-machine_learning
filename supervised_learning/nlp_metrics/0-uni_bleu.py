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

    len_ref = min(len(ref) for ref in references)
    len_sentence = len(sentence)

    # calculate BP:
    if len_sentence > len_ref:
        BP = 1
    else:
        BP = np.exp(1 - len_ref / len_sentence)

    # count words in each reference sentence
    ref_counts = []
    for ref in references:
        ref_counts.append(Counter(ref))
    print(ref_counts)

    # count word in sentence
    sentence_counts = Counter(sentence)
    print(sentence_counts)

    # calculate precision
    precision = 0
    for word in sentence:
        max_match = 0
        for counter in ref_counts:
            max_match = max(max_match, counter[word])

        precision += max_match / len_sentence

    # calculate bleu unigram score
    bleu_score = BP * precision

    return bleu_score
