#!/usr/bin/env python3
"""
     Change school topics
"""


def update_topics(mongo_collection, name, topics):
    """
        change all topics of a school doc based on the name

    :param mongo_collection: pymongo collection obj
    :param name: string, name to update
    :param topics: list of string, topics approached

    """
    mongo_collection.update_many(
        {"name": name},
        {"$set":
             {"topics": topics}
         })
