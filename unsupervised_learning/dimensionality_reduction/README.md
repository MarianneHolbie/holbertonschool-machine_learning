# Dimensionality Reduction

Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining a set of principal variables. It can be divided into feature selection and feature extraction.

## Feature Selection

Feature selection is the process of selecting a subset of relevant features for use in model construction. It is used to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable.

### Filter Methods
Filter methods are generally used as a preprocessing step. The selection of features is independent of any machine learning algorithm. Features are selected on the basis of their scores in various statistical tests for their correlation with the outcome variable.

### Wrapper Methods
Wrapper methods consider the selection of a set of features as a search problem. The selection of the set of features is based on the performance of the model. It follows a greedy search approach by evaluating all possible combinations of features against some evaluation criteria.

### Embedded Methods
Embedded methods learn which features best contribute to the accuracy of the model while the model is being created. The most common type of embedded feature selection methods are regularization methods.

## TASKs : Feature Extraction

Feature extraction is the process of transforming high-dimensional data into a lower-dimensional form. It is used to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable.

| Task                                              | Description                                                                                                                                                      |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Principal Component Analysis (PCA)](0-pca.py)    | The function performs Principal Component Analysis (PCA) on a dataset to reduce its dimensionality while retaining a specified fraction of the original variance |
| [Principal Component Analysis (PCA) v2](1-pca.py) | The function performs Principal Component Analysis (PCA) on a dataset to reduce its dimensionality while retaining a specified number of principal components    |
| [t-SNE](2-P_init.py)                                | The function performs t-Distributed Stochastic Neighbor Embedding (t-SNE) on a dataset to reduce its dimensionality for visualization                            |
| [Entropy](3-entropy.py)                           | The function calculates the Shannon entropy and P affinities of a dataset                                                                                        |
| [P affinities](4-P_affinities.py)                 | The function calculates the P affinities of a dataset                                                                                                            |
| [ Q affinities](5-Q_affinities.py)                | The function calculates the Q affinities of a dataset                                                                                                            |
| [Gradients](6-grads.py)                           | The function calculates the gradients of Y                                                                                                                       |
| [Cost](7-cost.py)                                 | The function calculates the cost of the t-SNE transformation                                                                                                     |
| [t-SNE v2](8-tsne.py)                             | The function performs t-Distributed Stochastic Neighbor Embedding (t-SNE) on a dataset to reduce its dimensionality for visualization                            |
