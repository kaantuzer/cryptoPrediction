# 🧠 Crypto Trading Bot with Flask & Reinforcement Learning

This project is a fully-integrated cryptocurrency trading simulation platform built with **Flask**, featuring a complete user system and a **Reinforcement Learning (RL)**-powered backtesting engine.

---

## 🚀 Features

- 🔐 User authentication (Register, Login, Logout)
- 📊 Dashboard to select cryptocurrency pairs
- ⏱️ Interval selection (4h or 5m trading)
- 🤖 RL-based model training and testing (Backtesting)
- 📈 Visual and dynamic result rendering
- 🗃️ SQLite-based storage using SQLAlchemy

---

## 📁 Project Structure

hft_app/
│
├── app.py # Main Flask app
├── config.py # Configuration (SECRET_KEY, DB_URI, etc.)
├── models.py # SQLAlchemy models (User)
├── core/
│ └── rl_utils.py # RL training and backtesting functions
├── templates/ # HTML templates (Jinja2)
│ ├── login.html
│ ├── register.html
│ ├── dashboard.html
│ ├── interval.html
│ └── backtest.html
├── static/ # CSS/JS/static files
├── requirements.txt # Python dependencies
└── app.db # SQLite database


## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kaantuzer/cryptoPrediction.git
cd cryptoPrediction

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

SECRET_KEY=your-secret-key
SQLALCHEMY_DATABASE_URI=sqlite:///app.db

python app.py
```

## ⚙️ Dependencies
Flask

Flask-Login

SQLAlchemy

Werkzeug

Your own rl_utils.py with train_model() and run_backtest() functions



