"""
BehavioralDNA - Backend Server
Flask + Scikit-learn Isolation Forest for anomaly detection
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, csv
from datetime import datetime
from ml_model import BehavioralModel

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

DATA_FILE = "enrolled_profiles.json"
LOG_FILE  = "login_log.csv"

model = BehavioralModel()

# ─── Load existing profiles on startup ───────────────
def load_profiles():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(DATA_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)

def log_attempt(username, features, status, score):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp","username","status","score",
                             "avg_interval","avg_hold_time","typing_speed",
                             "backspace_count","total_keys","mouse_speed"])
        writer.writerow([
            datetime.now().isoformat(), username, status, f"{score:.4f}",
            features.get("avg_interval",0), features.get("avg_hold_time",0),
            features.get("typing_speed",0), features.get("backspace_count",0),
            features.get("total_keys",0), features.get("mouse_speed",0)
        ])

profiles = load_profiles()

# ─── Enroll Route ─────────────────────────────────────
@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.json
    username = data.get('username', '').strip().lower()
    features = data.get('features', {})

    if not username:
        return jsonify({"error": "username required"}), 400

    if username not in profiles:
        profiles[username] = []

    profiles[username].append(features)
    save_profiles(profiles)

    # Retrain model if enough sessions
    all_sessions = [s for sessions in profiles.values() for s in sessions]
    if len(all_sessions) >= 3:
        model.train(all_sessions)

    return jsonify({
        "status": "enrolled",
        "username": username,
        "session_count": len(profiles[username])
    })

# ─── Login / Detect Route ────────────────────────────
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip().lower()
    features = data.get('features', {})

    if not username:
        return jsonify({"error": "username required"}), 400

    if username not in profiles or len(profiles[username]) < 2:
        return jsonify({
            "status": "insufficient_data",
            "message": "Not enough enrolled sessions. Please enroll more.",
            "score": 0
        })

    # Run ML prediction
    result = model.predict(features, profiles[username])
    status = result['status']
    score  = result['score']

    log_attempt(username, features, status, score)

    return jsonify({
        "username": username,
        "status": status,   # 'normal' or 'anomaly'
        "score": score,
        "message": result['message']
    })

# ─── View logs route (for demo dashboard) ───────────
@app.route('/logs', methods=['GET'])
def get_logs():
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        return jsonify(list(reader))

@app.route('/profiles', methods=['GET'])
def get_profiles():
    return jsonify({u: len(s) for u, s in profiles.items()})

if __name__ == '__main__':
    print("🔐 BehavioralDNA Server starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)