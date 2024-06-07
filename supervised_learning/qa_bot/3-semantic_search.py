#!/usr/bin/python3
"""semantic search"""
import os
from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np


def semantic_search(corpus_path, sentence):
    """
        performs semantic search on a corpus of documents

    :param corpus_path: path to the corpus of reference documents
    :param sentence: sentence from which to perform semantic search

    :return: reference text of the document most similar to sentence
    """
    corpus = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, 'r', encoding='utf-8') as md_file:
                corpus.append(md_file.read() + '\n')

    # load sentence transformer model
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    # set similarity native function to COSINE
    model.similarity_fn_name = SimilarityFunction.COSINE

    # encode input
    embeddings_sen = model.encode(sentence)

    similarities = [model.similarity(embeddings_sen, model.encode(doc))
                    for doc in corpus]

    best_similarity_idx = np.argmax(similarities)
    best_doc = corpus[best_similarity_idx]

    return best_doc
