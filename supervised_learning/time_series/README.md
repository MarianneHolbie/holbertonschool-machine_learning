# Forecasting Bitcoin Prices with RNNs

## Overview
Bitcoin (BTC) has attracted significant attention due to its volatile nature and potential for substantial financial gains. In this project, we aim to leverage Recurrent Neural Networks (RNNs) to predict the future value of Bitcoin based on historical data.

## Dataset
We utilize two datasets[data](..%2Fdata), coinbase and bitstamp, which contain information about Bitcoin prices over time. Each row in the dataset represents a 60-second time window and includes various metrics such as open, high, low, and close prices, as well as transaction volume.

 load the datasets : 
* [Coinbase](https://intranet.hbtn.io/rltoken/vEVzC0M9D73iMNUZqf7Tpg) (rename in coinbase.csv)
* [Bitstamp](https://intranet.hbtn.io/rltoken/JjaZZyvz3hChdFxPNbc3hA) (rename in bitstamp.csv)

## Script Description
The `forecast_btc.py` script is designed to create, train, and validate a Keras model for forecasting Bitcoin prices. Here's what the script does:

* **Data Preprocessing**: The script preprocesses the raw data, including converting timestamps to datetime objects, handling missing values, and normalizing the data for training.
* **Model Architecture**: We experiment with two RNN architectures, LSTM and GRU, to capture temporal dependencies in the data. Both models consist of multiple layers of LSTM or GRU cells, followed by dropout layers to prevent overfitting.
* **Model Training**: The models are trained using mean-squared error (MSE) as the cost function and optimized using the Adam optimizer. We monitor the model's performance on a validation set and employ early stopping to prevent overfitting.
* **Evaluation**: After training, the models are evaluated on both the validation and test sets to assess their performance. We calculate metrics such as loss and mean absolute error (MAE) to gauge how well the models generalize to unseen data.
* **Results Analysis**: Finally, we analyze the results and compare the performance of the LSTM and GRU models. We provide insights into potential areas for improvement and future research directions.

## Usage
To run the script, simply execute `forecast_btc.py` in your Python environment. Ensure that the required dependencies, including TensorFlow and pandas, are installed.

## Conclusion
While predicting Bitcoin prices remains a challenging task, our experiment demonstrates the potential of RNNs in forecasting financial time series data. Further optimization and exploration of advanced techniques could enhance the accuracy and robustness of our models.

For a detailed walkthrough and code explanation, read the full article on [Medium](https://medium.com/@marianne.arrue/unlocking-bitcoins-future-can-rnns-forecast-market-trends-like-madame-irma-s-crystal-ball-a51baf2343d0).

## Disclaimer
This project is for educational and experimental purposes only. Cryptocurrency markets are highly volatile and unpredictable, and financial decisions should not be based solely on machine learning models.

For any inquiries or feedback, please contact [5608@holbertonstudents.com].