#!/usr/bin/env python3
"""
Script to provide stats about Nginx logs stored in MongoDB
"""

from pymongo import MongoClient


def get_nginx_stats():
    """
    Retrieves and displays statistics about Nginx logs from MongoDB
    """
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client.logs
    collection = db.nginx

    # Get total number of logs
    total_logs = collection.count_documents({})
    print("{} logs".format(total_logs))

    # Get stats for HTTP methods
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    # Get stats for GET method with path /status
    status_check = collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print("{} status check".format(status_check))


if __name__ == "__main__":
    get_nginx_stats()
