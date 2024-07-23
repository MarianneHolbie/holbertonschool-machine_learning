#!/usr/bin/env python3
"""
    Where I am?
"""
import requests


def sentientPlanets():
    """
        method that returns list of names of the home planets
        of all 'sentient' species

    :return: list home planets for sentient species
    """

    url = "https://swapi-api.hbtn.io/api/species/"
    all_species = []

    # store all species
    while url:
        r = requests.get(url).json()
        all_species += r.get('results')
        url = r.get('next')
    all_planets = []
    # for each species 'sentient' get homeworld
    for species in all_species:
        if species.get('designation') == 'sentient' or \
                species.get('classification') == 'sentient':
            url = species.get('homeworld')
            # add name of planet to the list
            if url:
                planet = requests.get(url).json()
                all_planets.append(planet.get('name'))

    return all_planets
