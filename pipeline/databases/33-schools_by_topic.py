#!/usr/bin/env python3
"""
      Where can I learn Python?
"""


def schools_by_topic(mongo_collection, topic):
    """
        school having specific topic

    :param mongo_collection: pymongo collection objet
    :param topic: string, topic searched

    :return: list of school having topic
    """
    return mongo_collection.find({"topic": {"$in": [topic]}})
