import os
from datetime import datetime
import numpy as np
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None
try:
    import optuna
except Exception:
    optuna = None
data_base_path = os.path.join(os.getcwd(), 'data')
model_file_path = os.path.join(data_base_path, 'model.pkl')
scaler_file_path = os.path.join(data_base_path, 'scaler.pkl')
training_price_data_path = os.path.join(data_base_path, 'price_data.csv')
selected_features_path = os.path.join(data_base_path, 'selected_features.json')
best_model_info_path = os.path.join(data_base_path, 'best_model.json')
sol_source_path = os.path.join(data_base_path, os.getenv('SOL_SOURCE', 'raw_sol.csv'))
eth_source_path = os.path.join(data_base_path, os.getenv('ETH_SOURCE', 'raw_eth.csv'))
features_sol_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH', 'features_sol.csv'))
features_eth_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH_ETH', 'features_eth.csv'))
TOKEN = os.getenv('TOKEN', 'BTC')
TIMEFRAME = os.getenv('TIMEFRAME', '8h')
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 365))
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LSTM_Hybrid')
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCt1A4y2k')
SELECTED_FEATURES = ['volatility_BTCUSDT', 'volume_change_BTCUSDT', 'momentum_BTCUSDT', 'rsi_BTCUSDT', 'ma5_BTCUSDT', 'ma20_BTCUSDT', 'macd_BTCUSDT', 'bb_upper_BTCUSDT', 'bb_lower_BTCUSDT', 'sign_log_return_lag1_BTCUSDT', 'garch_vol_BTCUSDT', 'volatility_ETHUSDT', 'volume_change_ETHUSDT', 'sol_btc_corr', 'sol_eth_corr', 'sol_btc_vol_ratio', 'sol_btc_volume_ratio', 'sol_eth_vol_ratio', 'sol_eth_momentum_ratio', 'sentiment_score', 'sol_tx_volume', 'hour_of_day', *[f'open_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'high_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'low_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'close_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'open_BTCUSDT_lag{i}' for i in range(1, 11)], *[f'high_BTCUSDT_lag{i}' for i in range(1, 11)], *[f'low_BTCUSDT_lag{i}' for i in range(1, 11)], *[f'close_BTCUSDT_lag{i}' for i in range(1, 11)]]
MODEL_PARAMS = {'n_estimators': 1000, 'learning_rate': 0.015, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'reg_alpha': 0.5, 'min_gain_to_split': 0.01, 'n_jobs': 1, 'hidden_size': 64, 'num_layers': 2}
OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', 50))
USE_SYNTHETIC_DATA = os.getenv('USE_SYNTHETIC_DATA', 'True').lower() == 'true'
PERFORMANCE_THRESHOLDS = {'RMSE': {'excellent': 0.015, 'good': 0.025, 'acceptable': 0.04, 'current': 0.025}, 'MZTAE': {'excellent': 0.5, 'good': 0.75, 'acceptable': 1.0, 'current': 0.75}, 'directional_accuracy': {'excellent': 0.65, 'good': 0.55, 'target': 0.51, 'acceptable': 0.52, 'minimum': 0.5}, 'correlation': {'excellent': 0.5, 'good': 0.3, 'acceptable': 0.2, 'minimum': 0.1}}

def evaluate_performance(rmse, mztae, directional_acc=None, correlation=None):
    """
    Evaluate model performance against thresholds.
    Returns a performance rating and detailed assessment.
    """
    performance = {'overall': 'poor', 'details': {}}
    if rmse <= PERFORMANCE_THRESHOLDS['RMSE']['excellent']:
        performance['details']['rmse'] = 'excellent'
    elif rmse <= PERFORMANCE_THRESHOLDS['RMSE']['good']:
        performance['details']['rmse'] = 'good'
    elif rmse <= PERFORMANCE_THRESHOLDS['RMSE']['acceptable']:
        performance['details']['rmse'] = 'acceptable'
    else:
        performance['details']['rmse'] = 'poor'
    if mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['excellent']:
        performance['details']['mztae'] = 'excellent'
    elif mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['good']:
        performance['details']['mztae'] = 'good'
    elif mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['acceptable']:
        performance['details']['mztae'] = 'acceptable'
    else:
        performance['details']['mztae'] = 'poor'
    if directional_acc is not None:
        if directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['excellent']:
            performance['details']['directional_accuracy'] = 'excellent'
        elif directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['good']:
            performance['details']['directional_accuracy'] = 'good'
        elif directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['target']:
            performance['details']['directional_accuracy'] = 'target_met'
        elif directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['acceptable']:
            performance['details']['directional_accuracy'] = 'acceptable'
        else:
            performance['details']['directional_accuracy'] = 'poor'
    if correlation is not None:
        if correlation >= PERFORMANCE_THRESHOLDS['correlation']['excellent']:
            performance['details']['correlation'] = 'excellent'
        elif correlation >= PERFORMANCE_THRESHOLDS['correlation']['good']:
            performance['details']['correlation'] = 'good'
        elif correlation >= PERFORMANCE_THRESHOLDS['correlation']['acceptable']:
            performance['details']['correlation'] = 'acceptable'
        else:
            performance['details']['correlation'] = 'poor'
    ratings = list(performance['details'].values())
    if all((r in ['excellent', 'good'] for r in ratings)):
        performance['overall'] = 'excellent' if 'excellent' in ratings else 'good'
    elif all((r in ['excellent', 'good', 'target_met', 'acceptable'] for r in ratings)):
        performance['overall'] = 'acceptable'
    else:
        performance['overall'] = 'poor'
    return performance

def meets_current_thresholds(rmse, mztae, directional_acc=None):
    """
    Check if model meets current performance thresholds.
    Returns True if RMSE, MZTAE, and directional accuracy meet current thresholds.
    """
    rmse_ok = rmse <= PERFORMANCE_THRESHOLDS['RMSE']['current']
    mztae_ok = mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['current']
    da_ok = True
    if directional_acc is not None:
        da_ok = directional_acc > PERFORMANCE_THRESHOLDS['directional_accuracy']['target']
    return rmse_ok and mztae_ok and da_ok

def get_performance_summary(rmse, mztae, directional_acc=None, correlation=None):
    """
    Generate a human-readable performance summary.
    """
    meets_threshold = meets_current_thresholds(rmse, mztae, directional_acc)
    evaluation = evaluate_performance(rmse, mztae, directional_acc, correlation)
    summary = []
    summary.append(f'Performance Summary for Competition 18:')
    summary.append(f"{'=' * 50}")
    summary.append(f"RMSE: {rmse:.6f} (Threshold: {PERFORMANCE_THRESHOLDS['RMSE']['current']}) - {evaluation['details']['rmse'].upper()}")
    summary.append(f"MZTAE: {mztae:.6f} (Threshold: {PERFORMANCE_THRESHOLDS['MZTAE']['current']}) - {evaluation['details']['mztae'].upper()}")
    if directional_acc is not None:
        da_status = evaluation['details'].get('directional_accuracy', 'N/A')
        da_display = da_status.replace('_', ' ').upper() if da_status != 'N/A' else 'N/A'
        target_threshold = PERFORMANCE_THRESHOLDS['directional_accuracy']['target']
        summary.append(f'Directional Accuracy: {directional_acc:.2%} (Target: >{target_threshold:.2%}) - {da_display}')
    if correlation is not None:
        summary.append(f"Correlation: {correlation:.4f} - {evaluation['details'].get('correlation', 'N/A').upper()}")
    summary.append(f"{'=' * 50}")
    summary.append(f"Overall Rating: {evaluation['overall'].upper()}")
    summary.append(f"Meets Current Thresholds: {('YES ✓' if meets_threshold else 'NO ✗')}")
    return '\n'.join(summary)