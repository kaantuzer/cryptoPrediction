# app.py  –  Tam Entegre Flask + RL Trading Bot
import os
from flask import (
    Flask, render_template, redirect, url_for,
    flash, request
)
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# --- Flask / DB Kurulumu ----------------------------------------------------
app = Flask(__name__)
app.config.from_object("config.Config")

# SQLAlchemy
from models import db, User
db.init_app(app)

# Login yöneticisi
login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- RL Yardımcı Fonksiyonlar ----------------------------------------------
from core.rl_utils import train_model, run_backtest

# --- ROTALAR ----------------------------------------------------------------
@app.route("/")
@login_required
def index():
    return redirect(url_for("dashboard"))

# 1) Kayıt -------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        uname  = request.form.get("username")
        email  = request.form.get("email")
        pwd    = request.form.get("password")

        if User.query.filter_by(username=uname).first():
            flash("Bu kullanıcı adı zaten alınmış.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Bu e‑posta zaten kayıtlı.", "danger")
        else:
            hashed_pw = generate_password_hash(pwd)
            user = User(username=uname, email=email, password=hashed_pw)
            db.session.add(user)
            db.session.commit()
            flash("Kayıt başarılı! Giriş yapabilirsiniz.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")

# 2) Giriş -------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email")
        pwd   = request.form.get("password")
        user  = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, pwd):
            login_user(user)
            flash("Giriş başarılı!", "success")
            return redirect(url_for("index"))
        else:
            flash("E‑posta veya şifre yanlış.", "danger")

    return render_template("login.html")

# 3) Dashboard – Coin seçimi --------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    coins = [
        {"symbol": "BTCUSDT", "name": "Bitcoin",  "icon": "🟠"},
        {"symbol": "ETHUSDT", "name": "Ethereum", "icon": "🔵"},
        {"symbol": "BNBUSDT", "name": "BNB",      "icon": "🟡"},
        {"symbol": "XRPUSDT", "name": "XRP",      "icon": "⚪"},
        {"symbol": "SOLUSDT", "name": "Solana",   "icon": "🟣"},
    ]
    return render_template("dashboard.html", coins=coins)

# 4) Zaman dilimi seçimi ------------------------------------------------------
@app.route("/interval/<symbol>")
@login_required
def select_interval(symbol):
    intervals = [
        {"id": "4h", "label": "4 Saatlik Trading"},
        {"id": "5m", "label": "5 Dakikalık Trading"},
    ]
    return render_template("interval.html", symbol=symbol, intervals=intervals)

# 5) Backtest – Eğitim + Test -------------------------------------------------
@app.route("/backtest")
@login_required
def backtest():
    symbol   = request.args.get("symbol")
    interval = request.args.get("interval")

    if not symbol or not interval:
        flash("Coin ve zaman dilimi seçilmedi.", "warning")
        return redirect(url_for("dashboard"))

    try:
        # 1. Modeli eğit
        flash("⚙️ Model eğitiliyor, lütfen birkaç dakika bekleyin…", "info")
        train_model(symbol, interval)

        # 2. Hemen test et
        result = run_backtest(symbol, interval)

        return render_template(
            "backtest.html",
            result=result,
            symbol=symbol,
            interval=interval
        )

    except Exception as e:
        flash(f"Hata: {str(e)}", "danger")
        return redirect(url_for("dashboard"))

# 6) Çıkış --------------------------------------------------------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()          # user tablosu yoksa oluştur
    app.run(debug=True)
