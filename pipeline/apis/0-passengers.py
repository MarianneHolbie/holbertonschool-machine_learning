#!/usr/bin/env python3
"""
    Can I join ?
"""
import requests


def availableShips(passengerCount):
    """
        method that returns list of ships that can hold a
        given number of passengers

    :param passengerCount: given number of passengers
    :return: empty list if no ship available
            list of available ships
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'

    r = requests.get(url).json()

    available_ships = []

    while r.get("next"):
        starship = r.get("results")
        for ship in starship:
            passenger = ship.get("passengers")

            if passenger.isnumeric():
                nbr_passg = int(passenger.replace(",", ""))

                if nbr_passg >= passengerCount:
                    available_ships.append(ship.get("name"))

        r = requests.get(r.get("next")).json()

    return available_ships
