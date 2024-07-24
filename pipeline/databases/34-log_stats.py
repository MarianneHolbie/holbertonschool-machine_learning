#!/usr/bin/env python3
"""
      Log stats
"""
from pymongo import MongoClient


if __name__ == "__main__":
    # connect MongoDB and load db and collection
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx

    # number of doc in collection
    nb_doc = logs.count_documents({})
    print("{} logs".format(nb_doc))

    # number doc with different methods
    print("Methods:")
    methods_name = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in methods_name:
        count = logs.count_documents({"method": m})
        print("\tmethod {}: {}".format(m, count))

    # number doc with get and status
    filter_doc = {"method": "GET", "path": "/status"}
    path_count = logs.count_documents(filter_doc)
    print("{} status check".format(path_count))
