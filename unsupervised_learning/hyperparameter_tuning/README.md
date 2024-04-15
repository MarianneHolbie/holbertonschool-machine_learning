# Hyperparameter tuning

Hyperparameter tuning is the process of finding the best hyperparameters for a machine learning model. Hyperparameters are parameters that are set before the learning process begins, and they can have a significant impact on the performance of the model. Hyperparameter tuning is an important step in the machine learning process, as it can help improve the performance of the model and reduce overfitting.

## TASKS

| Task                                                     | Description                                                                                                                                                                     |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Initialize Gaussian Process](./0-gp.py)                 | Function `def initialize(X, Y):` that initializes the variables `means`, `covariances`, and `l` of a Gaussian Process.                                                          |
| [Gaussian Process Prediction](./1-gp.py)                 | Function `def predict(x_s, X_train, Y_train, cov, l=1, sigma_f=1):` that predicts the mean, standard deviation, and the covariance of a Gaussian process.                       |
| [Update Gaussian Process](./2-gp.py)                     | Function `def update(x_new, Y_new, gausian_process, l=1, sigma_f=1):` that updates a Gaussian Process.                                                                          |
| [Initialization Bayesian Optimization](./3-bayes_opt.py) | Function `def bayes_optimization(n_iterations, bounds, f, X_init, Y_init, ac_samples, l=1, sigma_f=1):` that performs Bayesian optimization on a noiseless 1D Gaussian process. |
| [Acquisition Bayesian Optimization](./4-bayes_opt.py)    | Function `def acquisition(self)`: that calculates the next best sample location` that initializes variables for Gaussian process.                                               |
| [Bayesian Optimization](./5-bayes_opt.py)                | Function `def optimize(self, iterations=100)`: that optimizes the black-box function                                                                                            |
| [Bayesian Optimization with GPyOpt](./6-bayes_opt.py)    | Script python that optimizes a machine learning model of my choice using GPyOpt                                                                                                 |