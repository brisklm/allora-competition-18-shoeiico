import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import math

app = Flask(__name__)
load_dotenv()

COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "64")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 9001))

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        else:
            return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        else:
            return float(obj)
    elif hasattr(obj, 'item'):
        return sanitize_for_json(obj.item())
    else:
        return str(obj)

TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results.",
        "parameters": {}
    },
    {
        "name": "write_code",
        "description": "Writes complete source code to a specified file, overwriting existing content.",
        "parameters": {
            "file_name": "str",
            "content": "str"
        }
    }
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        from model import predict_log_return
        prediction = predict_log_return(data)
        sanitized_prediction = sanitize_for_json(prediction)
        return jsonify(sanitized_prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        from model import optimize_model
        result = optimize_model()
        sanitized_result = sanitize_for_json(result)
        return jsonify(sanitized_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT)