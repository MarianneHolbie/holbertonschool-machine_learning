#!/usr/bin/env python3
"""
    t-SNE transformation
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
        function that performs a t-SNE transformation

    :param X: ndarray, shape(n,d) dataset to be transformed
                n: number of data points
                d: number of dimensions in each datapoint
    :param ndims: new dimensional representation of X
    :param idims: intermediate dimensional representation of x after pca
    :param perplexity: perplexicity
    :param iterations: number of iterations
    :param lr: learning rate

    :return: Y: ndarray, shape(n,ndim) optimized low dimensional
            transformation of X
    """

    # check inputs
    if isinstance(ndims, float):
        raise TypeError("Array X should have type float.")
    if round(ndims) != ndims:
        raise ValueError("Number of dimensions should be an integer")

    # reduce dimensionality with PCA to idims
    X = pca(X, idims)
    n, d = X.shape

    # Compute pairwise affinities P in the original space
    P = P_affinities(X, perplexity)

    # initialization Y with random samples form normal distrib
    Y = np.random.randn(n, ndims)
    Y_update = np.zeros(Y.shape)

    # Early exaggeration for the first 100 iterations
    exaggeration_factor = 4.0
    P *= exaggeration_factor

    # Initialize momentum
    momentum = 0.5

    # Run iterations
    for i in range(iterations):

        if i == 20:
            momentum = 0.8

        # compute gradients
        dY, Q = grads(Y, P)

        Y_update = momentum * Y_update - lr * dY

        # Perform the update
        Y += Y_update

        # Recenter Y by subtracting its mean
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Print cost periodically
        if (i + 1) % 100 == 0 and i != 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        if (i + 1) == 100:
            P /= exaggeration_factor

    return Y
