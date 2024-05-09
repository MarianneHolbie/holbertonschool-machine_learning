#!/usr/bin/env python3
"""
    Module to forecast analysis of the Bitcoins
"""
import os
import pandas as pd
import numpy as np
import tensorflow.keras.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
import preprocess_data


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
            Make dataset with frame 24h
        :param data: dataset

        :return: dataframe create for tensorflow
        """
        df_tf = tensorflow.keras.preprocessing.timeseries_dataset_from_array(data=data,
                                                                             targets=None,
                                                                             sequence_length=self.total_window_size,
                                                                             sequence_stride=1,
                                                                             shuffle=False,
                                                                             batch_size=32)

        # apply time windowing to the tf.dataset
        df_tf = df_tf.map(self.split_window)

        return df_tf

    """
    The WindowGenerator object holds training, validation and test data.
    Properties addition for accessing them as tf.data.Datasets using the above make_dataset method. 
    Addition of a standard example batch for easy access and plotting
    """

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


def normalize_data(df_train, df_val, df_test):
    """
        normalization with train_mean and train_std
    :param df_train: data to normalize
    :param df_val: data to normalize
    :param df_test: data to normalize
    :return: normalized data
    """
    train_mean = df_train.mean(axis=0)
    train_std = df_train.std(axis=0)

    df_train_norm = (train_df - train_mean) / train_std
    df_val_norm = (df_val - train_mean) / train_std
    df_test_norm = (df_test - train_mean) / train_std
    return df_train_norm, df_val_norm, df_test_norm


def split_data(df):
    """
        function to split dataset:
            * 70 % in training set
            * 20 % in validation set
            * 10 % in testing set

    :param df: dataframe to split
    :return: train_data, val_data, test_data
    """
    # log differential
    log_df = df.apply(lambda x: np.log(x) - np.log(x.shift(1)), axis=0)
    log_df = log_df.dropna()

    n = len(log_df)
    train_data = df[:int(n * 0.7)]
    val_data = df[int(n * 0.7): int(n * 0.9)]
    test_data = df[int(n * 0.9):]

    return train_data, val_data, test_data


def plot_eval_train(history):
    """
        Plot curv
    :param: history: history from model training

    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('LSTM Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.legend()
    plt.show()


def compile_fit(model, window, patience=5, epochs=200, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min',
                                   restore_best_weights=True
                                   )

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=['mae'])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping],
                        batch_size=batch_size,
                        verbose=1)

    print(model.summary())

    return history


if __name__ == "__main__":
    # load data and preprocess
    if not os.path.isfile('preprocess_data.csv'):
        preprocess_data = preprocess_data.preprocess_data("bitstamp.csv", "coinbase.csv")
    else:
        preprocess_data = pd.read_csv("preprocess_data.csv")

    # split data and labels
    train_df, val_df, test_df = split_data(preprocess_data)

    # normalization
    train_norm, val_norm, test_norm = normalize_data(train_df, val_df, test_df)

    window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=train_norm, val_df=val_norm, test_df=test_norm,
        label_columns=['Close'])

    # model
    LSTM_model = Sequential([
        LSTM(16, return_sequences=True),
        Dropout(0.5),
        LSTM(16),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    GRU_model = Sequential([
        GRU(32, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])

    history1 = compile_fit(LSTM_model, window)
    LSTM_model.save('LSTM_model.h5')
    history2 = compile_fit(GRU_model, window)
    GRU_model.save('GRU_model.h5')

    # Performance log
    val_performance = {}
    performance = {}

    # Evaluate model on validation dataset
    val_performance['LSTM'] = LSTM_model.evaluate(window.val, verbose=0)
    val_performance['GRU'] = GRU_model.evaluate(window.val, verbose=0)

    # Evaluate model on test dataset
    performance['LSTM'] = LSTM_model.evaluate(window.test, verbose=0)
    performance['GRU'] = GRU_model.evaluate(window.test, verbose=0)

    # Print results
    print("Results on validation set :")
    print(f"LSTM : {val_performance['LSTM']}")
    print(f"GRU : {val_performance['GRU']}")

    print("\nResults on test set :")
    print(f"LSTM : {performance['LSTM']}")
    print(f"GRU : {performance['GRU']}")

    print("Plot for LSTM model :")
    plot_eval_train(history1)
    print("Plot for GRU model :")
    plot_eval_train(history2)
