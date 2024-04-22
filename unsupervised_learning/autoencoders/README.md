# Autoencoders

Autoencoders are a type of neural network that are used for unsupervised learning. They are used to learn a compressed representation of the input data, and are often used for tasks such as dimensionality reduction and data denoising. Autoencoders are used in a wide range of applications, including image recognition, speech recognition, and anomaly detection.

## Requirements

* [Python](https://www.python.org/) 3.8
* [Tensorflow](https://www.tensorflow.org/) 2.6
* [Numpy](https://numpy.org/) 1.19.2
* [pycodestyle](https://pypi.org/project/pycodestyle/) 2.6

## Tasks
| Task                                              | Description                                                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [Vanilla autoencoder](./0-vanilla.py)             | Write a function `def autoencoder(input_dims, hidden_layers, latent_dims)` that creates a "vanilla" autoencoder   |
| [Sparse autoencoder](./1-sparse.py)               | Write a function `def sparse(input_dims, hidden_layers, latent_dims, lambtha)` that creates a sparse autoencoder  |
| [Convolutional autoencoder](./2-convolutional.py) | Write a function `def autoencoder(input_dims, filters, latent_dims)` that creates a convolutional autoencoder     |
| [Variational autoencoder](./3-variational.py)     | Write a function `def autoencoder(input_dims, hidden_layers, latent_dims)` that creates a variational autoencoder |

