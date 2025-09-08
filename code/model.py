import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import joblib
from config import *  # Import settings
if SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
def get_vader_sentiment(text):
    if SentimentIntensityAnalyzer:
        return sia.polarity_scores(text)['compound']
    else:
        return 0.0
def engineer_features(df):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    for lag in range(1,4):
        df[f'log_return_lag{lag}'] = df['log_return'].shift(lag)
    df['sign_log_return'] = np.sign(df['log_return'])
    df['momentum_8h'] = df['close'] - df['close'].shift(8)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    if 'news_text' in df.columns:
        df['vader_sentiment_compound'] = df['news_text'].apply(get_vader_sentiment)
    else:
        df['vader_sentiment_compound'] = 0.0
    df = df.fillna(0)
    low_var_cols = [col for col in df.columns if df[col].var() < 1e-5]
    df = df.drop(columns=low_var_cols)
    return df
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except:
    tf = None
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(MODEL_PARAMS['lstm_units'], input_shape=input_shape))
    model.add(Dropout(MODEL_PARAMS['dropout']))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 10, 100)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    return np.random.rand()  # Placeholder
def optimize_model():
    if optuna:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        return study.best_params
    else:
        return {}
def train_model():
    df = pd.read_csv(training_price_data_path)
    df = engineer_features(df)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, scaler_file_path)
    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=MODEL_PARAMS['epochs'], batch_size=MODEL_PARAMS['batch_size'])
    model.save(model_file_path)