import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
from config import *

def load_data():
    data = pd.read_csv(training_price_data_path)
    data['vader_sentiment'] = data['text'].apply(calculate_vader_sentiment)
    return data

def preprocess_data(data):
    features = data[FEATURES]
    target = np.log(data['close'].shift(-1) / data['close'])
    features = features.dropna()
    target = target.loc[features.index]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, scaler_file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = lgb.LGBMRegressor(max_depth=5, num_leaves=31, learning_rate=0.05, n_estimators=100, reg_alpha=0.01, reg_lambda=0.01)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file_path)
    return model

def predict_log_return(data):
    features = pd.DataFrame(data, index=[0])[FEATURES]
    scaler = joblib.load(scaler_file_path)
    features_scaled = scaler.transform(features)
    model = joblib.load(model_file_path)
    prediction = model.predict(features_scaled)
    return {'log_return': prediction[0]}

def optimize_model():
    if optuna is None:
        return {'error': 'Optuna is not installed'}
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_params = study.best_params
    best_value = study.best_value
    
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = np.corrcoef(y_test, y_pred)[0, 1]**2
    directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    
    result = {
        'best_params': best_params,
        'best_value': best_value,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'correlation': correlation
    }
    
    with open(best_model_info_path, 'w') as f:
        json.dump(result, f)
    
    joblib.dump(model, model_file_path)
    return result