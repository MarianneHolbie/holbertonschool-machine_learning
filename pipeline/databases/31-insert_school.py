#!/usr/bin/env python3
"""
    Insert a document in Python
"""


def insert_school(mongo_collection, **kwargs):
    """
        function that insert new doc in collection

    :param mongo_collection: pymongo collection obj
    :param kwargs: new doc params

    :return: new _id
    """
    return mongo_collection.insert_one(**kwargs).inserted_id
