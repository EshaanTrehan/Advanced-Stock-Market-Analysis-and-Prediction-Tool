import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import numpy as np
from prophet import Prophet
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import pickle

warnings.filterwarnings("ignore", category=ConvergenceWarning)

index_tickers = {
    'Dow Jones Industrial Average': '^DJI',
    'S&P 500': '^GSPC',
    'Nasdaq Composite': '^IXIC',
    'Financial Times Stock Exchange 100': '^FTSE',
    'Nikkei 225': '^N225',
    'Nifty 50': '^NSEI',
    'VIX': '^VIX'
}

def add_sentiment_scores(df):
    finbert_pipeline = pipeline('sentiment-analysis', model='ProsusAI/finbert')

    df['headline_finbert_sentiment'] = df['headline'].apply(lambda text: finbert_pipeline(text)[0]['label'])
    df['headline_finbert_score'] = df['headline'].apply(lambda text: finbert_pipeline(text)[0]['score'])

    df['desc_finbert_sentiment'] = df['short_description'].apply(lambda text: finbert_pipeline(text)[0]['label'])
    df['desc_finbert_score'] = df['short_description'].apply(lambda text: finbert_pipeline(text)[0]['score'])

    sentiment_to_score = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['headline_finbert_numerical'] = df['headline_finbert_sentiment'].map(sentiment_to_score)
    df['desc_finbert_numerical'] = df['desc_finbert_sentiment'].map(sentiment_to_score)

    df['avg_finbert_sentiment'] = df[['headline_finbert_numerical', 'desc_finbert_numerical']].mean(axis=1)

    return df

def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0:data.shape[1]])
        y.append(data[i, 0])  
    return np.array(X), np.array(y)
    
def fetch_and_process_data(ticker, start_date='2019-12-01', end_date='2022-10-01'):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    
    news_data_path = 'cleaned_data.xlsx'  
    news_data = pd.read_excel(news_data_path)

    news_data = add_sentiment_scores(news_data)

    sentiment_scores_filename = 'sentiment_scores.csv'
    news_data.to_csv(sentiment_scores_filename, index=False)
    news_data = pd.read_csv('sentiment_scores.csv')
    
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    news_data['date'] = pd.to_datetime(news_data['date'])
    combined_data = pd.merge(stock_data, news_data, left_on='Date', right_on='date', how='left')
    
    combined_data['avg_finbert_sentiment'].fillna(0, inplace=True)
    combined_data['rolling_avg_sentiment'] = combined_data['avg_finbert_sentiment'].rolling(window=5).mean().fillna(0)
    combined_data['sentiment_score_change'] = combined_data['avg_finbert_sentiment'].diff().fillna(0)

    combined_data.drop(columns=['date'], inplace=True)
    
    return combined_data

def lstm_model_with_score_computation(look_back, X_train, y_train, X_validation, y_validation, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=90, activation='softsign', return_sequences=True, input_shape=(look_back, X_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100, activation = 'softsign', return_sequences = True))
    model.add(Dropout(0.3))

    model.add(LSTM(units = 120, activation = 'softsign', return_sequences = True))
    model.add(Dropout(0.4))

    model.add(LSTM(units = 160, activation = 'softsign'))
    model.add(Dropout(0.5))

    model.add(Dense(units = 1))

    model.compile(optimizer='RMSprop', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=200, validation_data=(X_validation, y_validation), batch_size=32)

    model.save('lstm_model_with_score.keras')

    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    return model

def lstm_model_without_score_computation(look_back, X_train, y_train, X_validation, y_validation, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=90, activation='softsign', return_sequences=True, input_shape=(look_back, X_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100, activation = 'softsign', return_sequences = True))
    model.add(Dropout(0.3))

    model.add(LSTM(units = 120, activation = 'softsign', return_sequences = True))
    model.add(Dropout(0.4))

    model.add(LSTM(units = 160, activation = 'softsign'))
    model.add(Dropout(0.5))

    model.add(Dense(units = 1))

    model.compile(optimizer='RMSprop', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=200, validation_data=(X_validation, y_validation), batch_size=32)

    model.save('lstm_model_without_score.keras')

    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    return model

def predict_with_lstm(X_test, model, scaler):
    lstm_predictions = model.predict(X_test)
    dummy_feature = np.zeros((lstm_predictions.shape[0], 3))
    predictions_combined = np.hstack((lstm_predictions, dummy_feature))
    predictions_scaled = scaler.inverse_transform(predictions_combined)
    lstm_predictions_scaled = predictions_scaled[:, 0]
    
    return lstm_predictions_scaled

def prophet_model_without_score(train_data_date, train_data_price):

    train_df = pd.DataFrame({'ds': train_data_date.values, 'y': train_data_price.values})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(train_df)
    with open('prophet_model_without_score.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)

    return model

def prophet_model_with_score(train_data_date, train_data_price, train_data_score):
    train_df = pd.DataFrame({'ds': train_data_date.values, 'y': train_data_price.values})
    train_df['score'] = train_data_score.values 
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('score')
    model.fit(train_df)
    with open('prophet_model_with_score.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)

    return model

def calculate_rmse(actual_prices, predictions):
    rmse = sqrt(mean_squared_error(actual_prices[:len(predictions)], predictions))
    return rmse

def calculate_mape(actual_prices, predictions):
    actual_prices_trimmed = actual_prices[:len(predictions)]
    mape = np.mean(np.abs((actual_prices_trimmed - predictions) / actual_prices_trimmed)) * 100
    return mape

def main():
    st.title('Analyzing the Impact of Global Crises on Stock Markets')

    index_selection = st.selectbox('Select an Index:', list(index_tickers.keys()))
    ticker = index_tickers[index_selection]
    combined_data = fetch_and_process_data(ticker)
    
    train_start, train_end = '2019-12-01', '2021-01-01'
    validation_start, validation_end = '2021-01-02', '2022-01-01'
    test_start, test_end = '2022-01-02', '2022-10-01'  

    train_data = combined_data[(combined_data['Date'] >= train_start) & (combined_data['Date'] <= train_end)]
    validation_data = combined_data[(combined_data['Date'] >= validation_start) & (combined_data['Date'] <= validation_end)]
    test_data = combined_data[(combined_data['Date'] >= test_start) & (combined_data['Date'] <= test_end)]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled_with_score = scaler.fit_transform(train_data[['Close', 'avg_finbert_sentiment', 'rolling_avg_sentiment', 'sentiment_score_change']].values)
    validation_scaled_with_score = scaler.transform(validation_data[['Close', 'avg_finbert_sentiment', 'rolling_avg_sentiment', 'sentiment_score_change']].values)
    test_scaled_with_score = scaler.transform(test_data[['Close', 'avg_finbert_sentiment','rolling_avg_sentiment', 'sentiment_score_change']].values)

    train_scaled_without_score = scaler.fit_transform(train_data[['Close']].values)
    validation_scaled_without_score = scaler.transform(validation_data[['Close']].values)
    test_scaled_without_score = scaler.transform(test_data[['Close']].values)

    look_back = 10  
    X_train_with_score, y_train_with_score = create_dataset(train_scaled_with_score, look_back)
    X_validation_with_score, y_validation_with_score = create_dataset(validation_scaled_with_score, look_back)
    X_test_with_score, y_test_with_score = create_dataset(test_scaled_with_score, look_back)

    X_train_without_score, y_train_without_score = create_dataset(train_scaled_without_score, look_back)
    X_validation_without_score, y_validation_without_score = create_dataset(validation_scaled_without_score, look_back)
    X_test_without_score, y_test_without_score = create_dataset(test_scaled_without_score, look_back)

    lstm_model_with_score = lstm_model_with_score_computation(look_back, X_train_with_score, y_train_with_score, X_validation_with_score, y_validation_with_score, X_test_with_score, y_test_with_score)
    lstm_model_without_score = lstm_model_without_score_computation(look_back, X_train_without_score, y_train_without_score, X_validation_without_score, y_validation_without_score, X_test_without_score, y_test_without_score)

    lstm_model_with_score = load_model('lstm_model_with_score.keras')
    lstm_model_without_score = load_model('lstm_model_without_score.keras')

    actual_prices = test_data['Close'].values
    
    lstm_predictions_with_score = predict_with_lstm(X_test_with_score, lstm_model_with_score, scaler)
    lstm_predictions_without_score = predict_with_lstm(X_test_without_score, lstm_model_without_score, scaler)
    prophet_model_without_score_model = prophet_model_without_score(train_data['Date'], train_data['Close'])
    prophet_model_with_score_model = prophet_model_with_score(train_data['Date'], train_data['Close'], train_data['avg_finbert_sentiment'])

    with open('prophet_model_without_score.pkl', 'rb') as pkl:
        prophet_model_without_score_model = pickle.load(pkl)

    with open('prophet_model_with_score.pkl', 'rb') as pkl:
        prophet_model_with_score_model = pickle.load(pkl)

    future_dates_df = pd.DataFrame({'ds': test_data['Date'].values, 'score': test_data['avg_finbert_sentiment'].values})

    prophet_predict_without_score = prophet_model_without_score_model.predict(future_dates_df[['ds']])
    prophet_predict_with_score = prophet_model_with_score_model.predict(future_dates_df)
    prophet_predictions_without_score = pd.Series(prophet_predict_without_score['yhat'].values, index=test_data['Close'].index)
    prophet_predictions_with_score = pd.Series(prophet_predict_with_score['yhat'].values, index=test_data['Close'].index)

    lstm_rmse_with_score = calculate_rmse(actual_prices, lstm_predictions_with_score)
    lstm_mape_with_score = calculate_mape(actual_prices, lstm_predictions_with_score)
    lstm_rmse_without_score = calculate_rmse(actual_prices, lstm_predictions_without_score)
    lstm_mape_without_score = calculate_mape(actual_prices, lstm_predictions_without_score)
    prophet_rmse_without_score = calculate_rmse(actual_prices, prophet_predictions_without_score)
    prophet_mape_without_score = calculate_mape(actual_prices, prophet_predictions_without_score)
    prophet_rmse_with_score = calculate_rmse(actual_prices, prophet_predictions_with_score)
    prophet_mape_with_score = calculate_mape(actual_prices, prophet_predictions_with_score)

    test_dates = test_data['Date'].values
    
    st.subheader(f'{index_selection} Closing Prices Over Time')
    plt.figure(figsize=(10, 5))
    plt.plot(combined_data['Date'], combined_data['Close'], label='Close')
    plt.title(f'{index_selection} Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    st.subheader('LSTM Predictions vs Actual Prices')
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates[:len(lstm_predictions_with_score)], actual_prices[:len(lstm_predictions_with_score)], label='Actual Prices')
    plt.plot(test_dates[:len(lstm_predictions_with_score)], lstm_predictions_with_score, label='LSTM Predictions with Score')
    plt.plot(test_dates[:len(lstm_predictions_without_score)], lstm_predictions_without_score, label='LSTM Predictions without Scores')
    plt.title('LSTM Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    st.subheader('Prophet Predictions vs Actual Prices')
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, actual_prices, label='Actual Prices')
    plt.plot(test_dates[:len(prophet_predictions_without_score)], prophet_predictions_without_score, label='Prophet Predictions without Scores')
    plt.plot(test_dates[:len(prophet_predictions_with_score)], prophet_predictions_with_score, label='Prophet Predictions with Scores')
    plt.title('Prophet Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    st.subheader('Prophet Predictions, LSTM Predictions and Actual Prices')
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, actual_prices, label='Actual Prices')
    plt.plot(test_dates[:len(lstm_predictions_with_score)], lstm_predictions_with_score, label='LSTM Predictions with Scores')
    plt.plot(test_dates[:len(lstm_predictions_without_score)], lstm_predictions_without_score, label='LSTM Predictions without Scores')
    plt.plot(test_dates[:len(prophet_predictions_without_score)], prophet_predictions_without_score, label='Prophet Predictions without Scores')
    plt.plot(test_dates[:len(prophet_predictions_with_score)], prophet_predictions_with_score, label='Prophet Predictions with Scores')
    plt.title('Prophet Predictions, LSTM Predictions and Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    st.write(f"LSTM RMSE with Scores: {lstm_rmse_with_score:.3f}, LSTM MAPE with Scores: {lstm_mape_with_score:.2f}%")
    st.write(f"LSTM RMSE without Scores: {lstm_rmse_without_score:.3f}, LSTM MAPE without Scores: {lstm_mape_without_score:.2f}%")
    st.write(f"Prophet RMSE without Scores: {prophet_rmse_without_score:.3f}, Prophet MAPE without Scores: {prophet_mape_without_score:.2f}%")
    st.write(f"Prophet RMSE with Scores: {prophet_rmse_with_score:.3f}, Prophet MAPE with Scores: {prophet_mape_with_score:.2f}%")

if __name__ == "__main__":
    main()
