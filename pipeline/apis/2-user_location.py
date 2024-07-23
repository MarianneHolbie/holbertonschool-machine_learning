#!/usr/bin/env python3
"""
Rate me is you can!
"""
import requests
import sys
import time


if __name__ == '__main__':
    # invalid command
    if len(sys.argv) != 2:
        exit()

    url = sys.argv[1]
    # default headers
    headers = {'Accept': 'application/vnd.github+json'}

    results = requests.get(url, headers=headers)

    if results.status_code == 200:
        print(results.json()["location"])

    elif results.status_code == 404:
        print("Not found")

    # case Rate limit
    elif results.status_code == 403:
        rate_lim = int(results.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        minutes = int((rate_lim - now) / 60)
        print("Reset in {} min".format(minutes))
