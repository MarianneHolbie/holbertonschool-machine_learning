#!/usr/bin/env python3
"""
How many by rocket?
"""
import requests



if __name__ == "__main__":
    # get all launches
    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()
    if launches is None:
        exit(1)

    rocket_dict = {}

    # Get rockets for each launches
    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(
            rocket_id)

        rocket_info = requests.get(rocket_url).json()
        rocket_name = rocket_info.get('name')

        # Count launches per rocket
        if rocket_dict.get(rocket_name) is None:
            rocket_dict[rocket_name] = 1
            continue
        rocket_dict[rocket_name] += 1

    # sort rockets by number of launches in descending order
    sorted_rocket = sorted(rocket_dict.items(),
                           key=lambda kv: kv[1],
                           reverse=True)

    # print number of launches per rocket
    for rocket, count in sorted_rocket:
        print("{}: {}".format(rocket, count))
        