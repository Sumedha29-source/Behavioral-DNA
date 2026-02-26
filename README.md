# Behavioral-DNA
# 🔐 BehavioralDNA — Fraud Detection System

## What It Does
A login system that captures **behavioral biometrics** (typing rhythm, keystroke timing, mouse movement) and uses **Machine Learning** to detect if the person logging in is really you.

---

## 📁 File Structure
```
behavioral_dna/
├── index.html          ← Frontend (HTML + CSS + JS)
├── app.py              ← Backend API (Flask)
├── ml_model.py         ← ML logic (Isolation Forest + Z-Score)
├── requirements.txt    ← Python dependencies
├── enrolled_profiles.json  ← Created automatically on enroll
├── login_log.csv           ← Created automatically on login
└── model.pkl               ← Saved ML model (after enough data)
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start the backend
```bash
python app.py
```
Server runs at: `http://127.0.0.1:5000`

### Step 3 — Open the frontend
Open `index.html` in your browser (just double-click or use Live Server in VS Code).

---

## 🧠 How the ML Works

### Features Captured:
| Feature | Description |
|---|---|
| `avg_interval` | Average time between keystrokes (ms) |
| `avg_hold_time` | Average key press duration (ms) |
| `typing_speed` | Keys per second |
| `backspace_count` | Number of corrections |
| `total_keys` | Session length |
| `mouse_speed` | Mouse movement speed (px/s) |

### Model:
1. **Enroll Phase:** User types normally a few times → features saved to JSON
2. **Training:** Once 3+ sessions exist, an **Isolation Forest** is trained
3. **Detection:** New session is scored — if it deviates too much → `ANOMALY`
4. **Fallback:** Z-score comparison against the user's own sessions (when not enough global data)

---

## 🔄 Flow Diagram

```
User Types → JS captures keydown/keyup/mousemove events
           → Computes features (interval, hold, speed, etc.)
           → POST to /enroll or /login

Backend    → Passes features to ML model
           → Returns { status: 'normal' | 'anomaly', score: 0-1 }

Frontend   → If anomaly → Show HIGH RISK alert + require 2FA (face scan)
           → If normal  → Show ACCESS GRANTED
```

---

## 🎯 Demo Flow (for Hackathon Judges)

1. Open the app → Switch to **Enroll** mode
2. Type normally a few times (username + password) — submit 3-5 times
3. Switch to **Login** mode
4. Type normally → Should get **Access Granted**
5. Type very differently (fast/slow) → Should trigger **High Risk alert**

> **Note:** Without a backend running, the frontend works in **demo mode** — it simulates responses randomly.

---

## 🔌 API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/enroll` | Save a behavioral session |
| POST | `/login` | Check if session is normal/anomaly |
| GET | `/logs` | View all login attempts (CSV data) |
| GET | `/profiles` | View enrolled users |

---

## 🏗️ Built With
- **Frontend:** HTML5, CSS3, Vanilla JS
- **Backend:** Python + Flask
- **ML:** Scikit-learn (Isolation Forest)
- **Storage:** JSON (profiles) + CSV (logs)
