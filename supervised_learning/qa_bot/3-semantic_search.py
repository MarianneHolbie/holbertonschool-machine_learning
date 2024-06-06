#!/usr/bin/env python3
"""
    Module semantic search
"""
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def semantic_search(corpus_path, sentence):
    """
        performs semantic search on a corpus of documents

    :param corpus_path: path to the corpus of reference documents
    :param sentence: sentence from which to perform semantic search

    :return: reference text of the document most similar to sentence
    """
    # load SBERT (sentence Transformer)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # tokenized inputs
    s_embedding = model.encode(sentence)

    document = []
    for filename in os.listdir(corpus_path):
        # open each document
        file_path = os.path.join(corpus_path, filename)
        print("file_path: ", file_path)
        if filename.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                document.append(f.read())

    # encode each text
    for doc in document:
        doc_embedding = model.encode(doc)

    # calculate similarity
    similarity = [cosine_similarity([s_embedding], [doc_embedding])]
    # find best result and retrieve corresponding text
    best_similarity = np.argmax(similarity)
    best_doc = document[best_similarity]

    return best_doc
