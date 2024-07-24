#!/usr/bin/env python3
"""
    List all documents in Python
"""


def list_all(mongo_collection):
    """
        list of all documents in collection

    :param mongo_collection: pymongo collection object
    :return: list of all documents in collection
    or empty if no document
    """
    return list(mongo_collection.find())
