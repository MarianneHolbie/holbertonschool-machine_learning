#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
        absorbing chains
"""
import numpy as np


def absorbing(P):
    """
        determines if a markov chain is absorbing

    :param P: ndarray, shape(n,n) standard transition matrix
        P[i,j] proba transitioning from state i to state j
        n: number of states in markov chain

    :return: True if absorbing, False on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    # found abs state
    abs_state = np.where((P.sum(axis=1) == 1) & (np.diag(P) == 1))[0]
    if len(abs_state) == 0:
        return False

    # non abs state
    non_abs_state = [i for i in range(n) if i not in abs_state]

    # way to abs state or not
    for state in non_abs_state:
        # initiate abs
        abs_bool = False
        visited = set()
        queue = [state]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in abs_state:
                abs_bool = True
                break

            for next_state in range(n):
                if P[current, next_state] > 0:
                    queue.append(next_state)

        if not abs_bool:
            return False

    return True
