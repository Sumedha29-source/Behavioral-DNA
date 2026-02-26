"""
BehavioralDNA - ML Model
Uses Isolation Forest (unsupervised) to detect behavioral anomalies.
Falls back to statistical z-score method when data is insufficient.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib, os

FEATURES_USED = [
    "avg_interval",       # milliseconds between keystrokes
    "avg_hold_time",      # how long keys are held
    "typing_speed",       # keys per second
    "backspace_count",    # error correction behavior
    "total_keys",         # session length
    "mouse_speed",        # pixels per second
]

ANOMALY_THRESHOLD = 0.3   # Isolation Forest score < -threshold → anomaly
Z_SCORE_THRESHOLD = 2.5   # Standard deviations for z-score fallback

class BehavioralModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        # Load if already saved
        if os.path.exists("model.pkl"):
            self._load()

    def _extract(self, session: dict) -> list:
        """Extract feature vector from a session dict."""
        return [float(session.get(f, 0)) for f in FEATURES_USED]

    def train(self, sessions: list):
        """Train Isolation Forest on all enrolled sessions."""
        X = np.array([self._extract(s) for s in sessions])
        if X.shape[0] < 3:
            return  # Not enough data yet

        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,    # expect ~10% anomalies
            random_state=42
        )
        self.model.fit(X_scaled)
        self.is_trained = True
        self._save()
        print(f"[ML] Model trained on {len(sessions)} sessions.")

    def predict(self, new_session: dict, user_sessions: list) -> dict:
        """
        Returns:
          { status: 'normal'|'anomaly', score: float 0-1, message: str }
        Score close to 1.0 = high anomaly risk.
        """
        if self.is_trained and self.model is not None:
            return self._predict_isolation_forest(new_session)
        else:
            return self._predict_zscore(new_session, user_sessions)

    def _predict_isolation_forest(self, session: dict) -> dict:
        x = np.array([self._extract(session)])
        x_scaled = self.scaler.transform(x)

        # score_samples returns negative values; more negative = more anomalous
        raw_score = self.model.score_samples(x_scaled)[0]
        # Normalize to 0–1 range (0 = normal, 1 = very anomalous)
        # Typical range is roughly -0.8 to -0.3
        normalized = np.clip((-raw_score - 0.3) / 0.5, 0, 1)
        prediction = self.model.predict(x_scaled)[0]  # 1=normal, -1=anomaly

        is_anomaly = prediction == -1
        return {
            "status": "anomaly" if is_anomaly else "normal",
            "score": float(normalized),
            "message": "Anomaly detected by Isolation Forest." if is_anomaly
                       else "Behavioral pattern within normal range.",
            "method": "isolation_forest"
        }

    def _predict_zscore(self, session: dict, user_sessions: list) -> dict:
        """Fallback: compare new session to user's own enrolled sessions via z-score."""
        if len(user_sessions) < 2:
            return {"status": "normal", "score": 0.0, "message": "Insufficient data.", "method": "none"}

        X_user = np.array([self._extract(s) for s in user_sessions])
        x_new  = np.array(self._extract(session))

        means = X_user.mean(axis=0)
        stds  = X_user.std(axis=0) + 1e-6  # avoid div by zero

        z_scores = np.abs((x_new - means) / stds)
        max_z = z_scores.max()
        avg_z = z_scores.mean()

        # Composite anomaly score (0-1)
        score = float(np.clip(avg_z / Z_SCORE_THRESHOLD, 0, 1))
        is_anomaly = max_z > Z_SCORE_THRESHOLD

        return {
            "status": "anomaly" if is_anomaly else "normal",
            "score": score,
            "message": f"Max z-score: {max_z:.2f}. {'Anomaly detected.' if is_anomaly else 'Within normal range.'}",
            "method": "zscore"
        }

    def _save(self):
        joblib.dump({"model": self.model, "scaler": self.scaler}, "model.pkl")

    def _load(self):
        try:
            data = joblib.load("model.pkl")
            self.model  = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True
            print("[ML] Loaded saved model.")
        except Exception as e:
            print(f"[ML] Could not load model: {e}")