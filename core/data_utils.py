# core/data_utils.py

import pandas as pd
import numpy as np
import ta
from binance.client import Client

client = Client()  # API key gerektirmez, genel veri çekmek için

def fetch_klines_and_compute_indicators(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """
    Binance'ten veri çeker ve teknik indikatörleri hesaplar.
    """
    # 1. Veriyi çek
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df = pd.DataFrame(klines, columns=[
        "timestamp", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Number_of_trades",
        "Taker_buy_base_asset_volume", "Taker_buy_quote_asset_volume", "Ignore"
    ])

    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)

    # 2. İndikatörleri hesapla
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    boll = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["Bollinger_High"] = boll.bollinger_hband()
    df["Bollinger_Low"] = boll.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["ATR"] = atr.average_true_range()

    # 3. NaN ve Inf temizle
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    return df
