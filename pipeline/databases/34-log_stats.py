#!/usr/bin/env python3
"""
      Log stats
"""
from pymongo import MongoClient


def count_function(collection, method):
    """
        Method to count from nginx log

    :param collection: name of collection
    :param method: selected method

    :return: count
    """
    return collection.count_documents({"method": method})


if __name__ == "__main__":
    # connect MongoDB and load db and collection
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx

    methods_name = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    print("{} logs".format(logs.count_function({})))
    print("Methods:")
    for method in methods_name:
        print("\tmethod {}: {}".format(method, count_function(logs, method)))
    print("{} status check".format(logs.count_documents({"method": "GET",
                                                         "path": "/status"})))
