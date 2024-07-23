#!/usr/bin/env python
"""
How many by rocket?
"""
import requests
from collections import defaultdict


def get_json(url):
    """ request data from url, return json"""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print("Request failed: {}".format(e))
        return None


if __name__ == "__main__":
    # get all launches
    url_launches = 'https://api.spacexdata.com/v4/launches'
    launches = get_json(url_launches)
    if launches is None:
        exit(1)

        # Get all rockets
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    rockets = get_json(rockets_url)
    if rockets is None:
        exit(1)

    # Create a dictionary to map rocket IDs to rocket names
    rocket_name_dict = {rocket['id']: rocket['name'] for rocket in rockets}

    # Count launches per rocket
    rocket_count = defaultdict(int)
    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket_name = rocket_name_dict.get(rocket_id, 'Unknown')
        rocket_count[rocket_name] += 1

    # sort rockets by number of launches in descending order
    sorted_rocket = sorted(rocket_count.items(),
                           key=lambda kv: kv[1],
                           reverse=True)

    # print number of launches per rocket
    for rocket, count in sorted_rocket:
        print("{}: {}".format(rocket, count))
