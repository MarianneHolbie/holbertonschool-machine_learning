#!/usr/bin/env python3
"""
    Class Dataset to load and preprocess the dataset
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
        Class to loads and prepas a dataset for machine translation
    """
    def __init__(self):
        """
            class init
        """
        self.data_train, self.data_valid =\
            tfds.load('ted_hrlr_translate/pt_to_en',
                      split=['train', 'validation'],
                      as_supervised=True)
        self.tokenizer_en, self.tokenizer_pt = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
            creates sub-word tokenizers for our dataset

        :param data: tf.data.Dataset, tuple (pt,en)
            pt: tf.Tensor portuguese sentence
            en: tf.Tensor english sentence
            max vocab size : 2**15

        :return: tokenizer_pt, tokenizer_en
            respectively portuguese and english tokenizer
        """
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )
        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """
            encodes a translation into tokens

        :param pt: tf.Tensor, portuguese sentence
        :param en: tf.Tensor, english sentence
        tokenized sentences include start and end of sentence tokens
        start token index as vocab_size
        end token index as vocab_size + 1

        :return: pt_tokens, en_tokens
            pt_tokens: ndarray, portuguese tokens
            en_tokens: ndarray, english tokens
        """
        # Convert tf.Tensors to strings
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')

        # encode sentences
        pt_tokens = self.tokenizer_pt.encode(pt)
        en_tokens = self.tokenizer_en.encode(en)

        # add start and end sentences
        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens
