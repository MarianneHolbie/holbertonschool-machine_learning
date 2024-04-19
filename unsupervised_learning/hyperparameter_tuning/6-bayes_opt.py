#!/usr/bin/env python3
"""
    Bayesian Optimization of Neural Network Hyperparameters
    with GPyOpt on the diabete dataset from scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# import from Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# import from sklearn
from sklearn.model_selection import train_test_split

# load dataset from sklearn
from sklearn.datasets import load_diabetes

# import for optimization
import GPyOpt

if __name__ == '__main__':
    # Function to plot model history
    def plot_model_history(history, title="Model Performance"):
        """Function to plot training and validation loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        plt.show()


    # Load data
    X, y = load_diabetes(return_X_y=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define the model + first training
    print("Training initial model with default hyperparameters...")
    initial_model = keras.Sequential([
        keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    initial_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    history_initial = initial_model.fit(X_train, y_train,
                                        validation_data=(X_test, y_test),
                                        epochs=100,
                                        batch_size=32,
                                        verbose=0)

    # Store initial value of validation MSE
    initial_val_mse = np.min(history_initial.history['val_mse'])
    print("Initial model validation MSE:", initial_val_mse)
    plot_model_history(history_initial, "Initial Model Performance")

    # Space for optimization hyperparameter
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
        {'name': 'num_units', 'type': 'discrete', 'domain': (50, 100, 150)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_regularization', 'type': 'continuous', 'domain': (0.001, 0.1)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)},
    ]

    # Objective function
    def model_training(x):
        """Function to train a model with given hyperparameters and return validation MSE."""
        print(f"Evaluating: {x}")
        learning_rate = x[0][0]
        num_units = int(x[0][1])
        dropout_rate = x[0][2]
        l2 = x[0][3]
        batch_size = int(x[0][4])

        model = keras.Sequential([
            keras.layers.Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse',
                      metrics=['mse'])

        # Early stopping and checkpoint
        checkpoint_filepath = f'./tmp/model_lr_{learning_rate}_units_{num_units}-dropout_{dropout_rate}.h5'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_mse',
            mode='min',
            save_best_only=True)

        early_stopping_callback = EarlyStopping(monitor='val_mse', patience=10, verbose=1)

        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=100,
                            batch_size=batch_size,
                            callbacks=[early_stopping_callback, model_checkpoint_callback],
                            verbose=0)

        val_mse = np.min(history.history['val_mse'])
        return val_mse


    # Define a GPyOpt object
    optimizer = GPyOpt.methods.BayesianOptimization(f=model_training,
                                                    domain=bounds,
                                                    model_type='GP',
                                                    acquisition_type='EI',
                                                    maximize=False)

    # Run the optimization
    optimizer.run_optimization(max_iter=30)

    best_params = optimizer.X[np.argmin(optimizer.Y)]
    best_model = keras.Sequential([
        keras.layers.Dense(int(best_params[1]), activation='relu', kernel_regularizer=keras.regularizers.l2(best_params[3])),
        keras.layers.Dropout(best_params[2]),
        keras.layers.Dense(1)
    ])

    best_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params[0]),
                       loss='mse',
                       metrics=['mse'])

    history_best = best_model.fit(X_train, y_train,
                                  validation_data=(X_test, y_test),
                                  epochs=100,
                                  batch_size=int(best_params[4]),
                                  verbose=0,
                                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)])

    # Plot the performance of the initial model and the best model
    plot_model_history(history_initial, "Initial Model Performance")
    plot_model_history(history_best, "Best Model Performance")

    # Print optimized parameters
    print("Optimized Parameters:", optimizer.x_opt)
    print("Optimized Loss:", optimizer.fx_opt)

    # Get the number of epochs for the best model
    num_epochs_best = len(history_best.history['loss'])
    print("Number of epochs for the best model:", num_epochs_best)

    # Compare the performance of the initial model with the optimized model
    print("Comparison:")
    print("Initial Model Validation MSE:", initial_val_mse)
    print("Optimized Model Validation MSE:", optimizer.fx_opt)

    # Save report
    optimizer.save_report("bayes_opt.txt")

    # Plot convergence
    optimizer.plot_convergence()
    plt.show()
