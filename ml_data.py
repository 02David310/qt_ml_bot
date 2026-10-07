import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from confluence import run_confluence, ConfluenceResult
from filters import get_concept_weights

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "EUR/CHF": "EURCHF=X", "GBP/JPY": "GBPJPY=X", "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X", "EUR/GBP": "EURGBP=X", "EUR/AUD": "EURAUD=X",
    "CHF/JPY": "CHFJPY=X", "CAD/JPY": "CADJPY=X", "AUD/JPY": "AUDJPY=X",
    "AUD/CHF": "AUDCHF=X", "AUD/CAD": "AUDCAD=X", "EUR/CAD": "EURCAD=X",
    "GBP/CAD": "GBPCAD=X",
}


@dataclass
class QTSignalML:
    pair: str
    signal: str
    score: float
    confidence: str
    price: float
    confluence: ConfluenceResult
    vote_breakdown: list = field(default_factory=list)
    df_snapshot: dict = field(default_factory=dict)  # rich features for ML
    df_raw: object = None  # raw DataFrame for TP/SL calculation


def fetch_15m(yf_symbol: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(yf_symbol, period="5d", interval="15m", progress=False)
        if df.empty or len(df) < 40:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def extract_snapshot(df: pd.DataFrame) -> dict:
    """Extract rich feature snapshot from dataframe for ML logging."""
    closes  = df["Close"].values
    highs   = df["High"].values
    lows    = df["Low"].values
    volumes = df["Volume"].values
    opens   = df["Open"].values
    n       = len(closes)
    price   = float(closes[-1])

    snapshot = {}

    # RSI
    try:
        delta    = np.diff(closes)
        gain     = np.where(delta > 0, delta, 0)
        loss     = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs       = avg_gain / (avg_loss + 1e-10)
        snapshot["rsi"] = round(float(100 - 100 / (1 + rs)), 2)
    except Exception:
        snapshot["rsi"] = 50.0

    # MACD histogram
    try:
        ema12 = pd.Series(closes).ewm(span=12).mean().values
        ema26 = pd.Series(closes).ewm(span=26).mean().values
        macd  = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=9).mean().values
        snapshot["macd_hist"] = round(float(macd[-1] - signal_line[-1]), 6)
    except Exception:
        snapshot["macd_hist"] = 0.0

    # Bollinger Band position
    try:
        sma20    = np.mean(closes[-20:])
        std20    = np.std(closes[-20:])
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_range = bb_upper - bb_lower
        snapshot["bb_position"] = round(float((price - bb_lower) / (bb_range + 1e-10)), 4)
    except Exception:
        snapshot["bb_position"] = 0.5

    # EMA diff
    try:
        ema9  = float(pd.Series(closes).ewm(span=9).mean().iloc[-1])
        ema21 = float(pd.Series(closes).ewm(span=21).mean().iloc[-1])
        snapshot["ema_diff_pct"] = round((ema9 - ema21) / price * 100, 4)
    except Exception:
        snapshot["ema_diff_pct"] = 0.0

    # Volume ratio
    try:
        avg_vol = float(np.mean(volumes[-20:]))
        cur_vol = float(volumes[-1])
        snapshot["volume_ratio"] = round(cur_vol / (avg_vol + 1e-10), 3)
    except Exception:
        snapshot["volume_ratio"] = 1.0

    # ATR %
    try:
        tr_list = []
        for i in range(-14, 0):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i]  - closes[i-1]))
            tr_list.append(tr)
        atr = float(np.mean(tr_list))
        snapshot["atr_pct"] = round(atr / price * 100, 4)
    except Exception:
        snapshot["atr_pct"] = 0.0

    # Candle pattern
    try:
        o, c, h, l = float(opens[-1]), float(closes[-1]), float(highs[-1]), float(lows[-1])
        o2, c2     = float(opens[-2]), float(closes[-2])
        body       = abs(c - o)
        rng        = h - l + 1e-10
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if c2 < o2 and c > o and c > o2 and o < c2:
            pattern = "bullish_engulfing"
        elif c2 > o2 and c < o and c < o2 and o > c2:
            pattern = "bearish_engulfing"
        elif lower_wick > body * 2 and lower_wick > upper_wick * 2:
            pattern = "hammer"
        elif upper_wick > body * 2 and upper_wick > lower_wick * 2:
            pattern = "shooting_star"
        elif c > o and body > rng * 0.7:
            pattern = "strong_bull"
        elif c < o and body > rng * 0.7:
            pattern = "strong_bear"
        else:
            pattern = "none"
        snapshot["candle_pattern"] = pattern
    except Exception:
        snapshot["candle_pattern"] = "none"

    return snapshot


def analyze_pair(pair: str) -> Optional[QTSignalML]:
    yf_symbol = PAIRS.get(pair)
    if not yf_symbol:
        return None
    df = fetch_15m(yf_symbol)
    if df is None:
        return None

    # Extract features first so XGBoost can use them for the 8th vote
    snapshot = extract_snapshot(df)
    
    # Get dynamic concept weights from historical win rates
    weights = get_concept_weights()
    result   = run_confluence(df, weight_overrides=weights, snapshot=snapshot)
    price    = float(df["Close"].iloc[-1])

    return QTSignalML(
        pair=pair, signal=result.signal,
        score=result.score, confidence=result.confidence,
        price=price, confluence=result,
        vote_breakdown=result.votes,
        df_snapshot=snapshot,
        df_raw=df,
    )


def scan_all_pairs() -> list:
    results = []
    for pair in PAIRS:
        sig = analyze_pair(pair)
        if sig:
            results.append(sig)
    results.sort(key=lambda x: x.score, reverse=True)
    return results


def scan_strong_signals(min_votes: int = 5) -> list:
    """Only return signals where ALL concepts agree."""
    all_sigs = scan_all_pairs()
    strong   = []
    for s in all_sigs:
        if s.signal == "HOLD":
            continue
        agreeing = s.confluence.buy_votes if s.signal == "BUY" else s.confluence.sell_votes
        if agreeing >= min_votes:
            strong.append(s)
    return strong
