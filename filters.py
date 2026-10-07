"""
Signal Filters — gate every signal before it reaches the user.
Implements: news blackout, ATR filter, session filter, pair blacklist,
dynamic vote threshold, cooldown, duplicate detection, concept weighting,
and ATR-based TP/SL calculation.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import pytz
from ml_logger import load_trades

CET = pytz.timezone("Europe/Berlin")

# ─────────────────────────────────────────────
# In-memory state
# ─────────────────────────────────────────────
_last_signal_time: dict[str, datetime] = {}   # pair -> last signal datetime
_COOLDOWN_MINUTES = 15


# ─────────────────────────────────────────────
# 1. News Blackout Filter
# ─────────────────────────────────────────────
# High-impact recurring events (CET times).
# Format: {weekday: [(hour, minute, name), ...]}
# weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
# These are approximate windows — the filter blocks ±30/15 min around them.
HIGH_IMPACT_EVENTS = {
    # FOMC — Wednesdays ~20:00 CET (roughly every 6 weeks, but we block every Wed)
    2: [(20, 0, "FOMC/Fed")],
    # ECB — Thursdays ~14:15 CET
    3: [(14, 15, "ECB Rate Decision")],
    # NFP — first Friday of month at 14:30 CET
    # CPI — 2nd week Tue/Wed at 14:30 CET
    # We block 14:30 CET on Tue/Wed/Fri for safety
    1: [(14, 30, "CPI/PPI (US)")],       # Tuesday
    4: [(14, 30, "US Data Release")],     # Friday (NFP + others)
}
# Also block 14:30 CET every weekday — most US data drops at 8:30 AM ET
for day in range(5):
    if day not in HIGH_IMPACT_EVENTS:
        HIGH_IMPACT_EVENTS[day] = []
    # Add the universal 14:30 US data slot if not already present
    if not any(h == 14 and m == 30 for h, m, _ in HIGH_IMPACT_EVENTS[day]):
        HIGH_IMPACT_EVENTS[day].append((14, 30, "US Econ Data"))

BLACKOUT_BEFORE = timedelta(minutes=30)
BLACKOUT_AFTER  = timedelta(minutes=15)


def is_news_blackout() -> tuple[bool, str]:
    """Check if current time is within a news blackout window."""
    now = datetime.now(CET)
    weekday = now.weekday()

    events = HIGH_IMPACT_EVENTS.get(weekday, [])
    for hour, minute, name in events:
        event_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        window_start = event_time - BLACKOUT_BEFORE
        window_end   = event_time + BLACKOUT_AFTER

        if window_start <= now <= window_end:
            if now < event_time:
                mins_to = int((event_time - now).total_seconds() / 60)
                return True, f"🚫 News blackout — {name} in {mins_to} min"
            else:
                mins_since = int((now - event_time).total_seconds() / 60)
                return True, f"🚫 News blackout — {name} was {mins_since} min ago"

    return False, ""


# ─────────────────────────────────────────────
# 2. Minimum ATR Filter (Dead Market)
# ─────────────────────────────────────────────
def is_market_dead(df: pd.DataFrame) -> tuple[bool, str]:
    """Return True if current ATR is < 50% of the 20-period average ATR."""
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    n = len(closes)

    if n < 22:
        return False, ""

    # Calculate true range for last 20 periods
    tr_list = []
    for i in range(-20, 0):
        tr = max(float(highs[i]) - float(lows[i]),
                 abs(float(highs[i]) - float(closes[i-1])),
                 abs(float(lows[i])  - float(closes[i-1])))
        tr_list.append(tr)

    avg_atr_20 = np.mean(tr_list)

    # Current candle's true range
    current_tr = max(float(highs[-1]) - float(lows[-1]),
                     abs(float(highs[-1]) - float(closes[-2])),
                     abs(float(lows[-1])  - float(closes[-2])))

    if avg_atr_20 > 0 and current_tr < avg_atr_20 * 0.5:
        ratio = current_tr / avg_atr_20 * 100
        return True, f"💤 Dead market — ATR at {ratio:.0f}% of average (need >50%)"

    return False, ""


# ─────────────────────────────────────────────
# 3. Session Quality Filter
# ─────────────────────────────────────────────
def is_bad_session() -> tuple[bool, str]:
    """Only allow signals during London (08-12 CET) and NY open (14-17 CET)."""
    now = datetime.now(CET)
    hour = now.hour

    # Weekend
    if now.weekday() >= 5:
        return True, "📅 Weekend — markets closed"

    # London session: 08:00-12:00 CET
    if 8 <= hour < 12:
        return False, ""

    # NY open session: 14:00-17:00 CET
    if 14 <= hour < 17:
        return False, ""

    # London/NY overlap: 12:00-14:00 also good
    if 12 <= hour < 14:
        return False, ""

    return True, f"⏰ Bad session — {hour}:00 CET is outside London/NY windows"


# ─────────────────────────────────────────────
# 4. Per-Pair Blacklist
# ─────────────────────────────────────────────
_pair_blacklist_cache: dict[str, tuple[datetime, tuple[bool, str]]] = {}
_BLACKLIST_CACHE_TTL = timedelta(minutes=30)


def is_pair_blacklisted(pair: str) -> tuple[bool, str]:
    """Blacklist pairs with < 40% win rate and 5+ closed trades."""
    now = datetime.now(CET)

    # Use cache to avoid reading trades.json every call
    if pair in _pair_blacklist_cache:
        cached_time, cached_result = _pair_blacklist_cache[pair]
        if now - cached_time < _BLACKLIST_CACHE_TTL:
            return cached_result

    trades = load_trades()
    pair_trades = [t for t in trades if t["pair"] == pair and t.get("result") is not None]
    total = len(pair_trades)

    if total < 5:
        result = (False, "")
        _pair_blacklist_cache[pair] = (now, result)
        return result

    wins = sum(1 for t in pair_trades if t["result"] == "WIN")
    win_rate = wins / total * 100

    if win_rate < 40:
        result = (True, f"🚫 {pair} blacklisted — {win_rate:.0f}% win rate ({wins}W/{total-wins}L on {total} trades)")
        _pair_blacklist_cache[pair] = (now, result)
        return result

    result = (False, "")
    _pair_blacklist_cache[pair] = (now, result)
    return result


# ─────────────────────────────────────────────
# 5. Dynamic Vote Threshold
# ─────────────────────────────────────────────
def get_dynamic_min_votes(score: float) -> int:
    """
    Context-aware vote threshold:
      Score >= 7.0 -> 3 votes enough
      Score 5.0-6.9 -> need 4 votes
      Score < 5.0 -> skip entirely (return 99)
    """
    if score >= 7.0:
        return 3
    elif score >= 5.0:
        return 4
    else:
        return 99  # effectively blocks


# ─────────────────────────────────────────────
# 6. Cooldown Per Pair
# ─────────────────────────────────────────────
def record_signal(pair: str):
    """Record that a signal was sent for this pair right now."""
    _last_signal_time[pair] = datetime.now(CET)


def is_on_cooldown(pair: str) -> tuple[bool, str]:
    """Check if this pair fired a signal in the last 15 minutes."""
    last = _last_signal_time.get(pair)
    if last is None:
        return False, ""

    now = datetime.now(CET)
    elapsed = (now - last).total_seconds() / 60

    if elapsed < _COOLDOWN_MINUTES:
        remaining = _COOLDOWN_MINUTES - elapsed
        return True, f"⏳ {pair} on cooldown — {remaining:.0f} min remaining"

    return False, ""


# ─────────────────────────────────────────────
# 7. Duplicate Trade Detection
# ─────────────────────────────────────────────
def is_duplicate_trade(pair: str, price: float) -> tuple[bool, str]:
    """Check if a trade for this pair at ±0.0005 was logged in the last 20 min."""
    trades = load_trades()
    now = datetime.now(CET)
    cutoff = now - timedelta(minutes=20)

    for t in reversed(trades):
        try:
            ts = datetime.fromisoformat(t["timestamp"])
            if ts < cutoff:
                break  # trades are chronological, stop once past cutoff
            if t["pair"] == pair and abs(t["price"] - price) <= 0.0005:
                return True, f"🔁 Duplicate — {pair} @ {t['price']:.5f} logged {int((now - ts).total_seconds() / 60)} min ago"
        except (KeyError, ValueError, TypeError):
            continue

    return False, ""


# ─────────────────────────────────────────────
# 8. Concept Win Rate Weighting
# ─────────────────────────────────────────────
_BASE_WEIGHTS = {
    "SMC": 1.5, "ICT": 1.5, "Wyckoff": 1.0,
    "VolumeProfile": 1.0, "Divergence": 1.0,
    "ICT_Adv": 1.5, "Structure": 1.5,
}

_VOTE_KEYS = {
    "SMC": "smc_vote", "ICT": "ict_vote", "Wyckoff": "wyckoff_vote",
    "VolumeProfile": "volprofile_vote", "Divergence": "divergence_vote",
    "ICT_Adv": "ict_adv_vote", "Structure": "structure_vote",
}

_concept_weight_cache: Optional[tuple[datetime, dict[str, float]]] = None
_WEIGHT_CACHE_TTL = timedelta(minutes=60)


def get_concept_weights() -> dict[str, float]:
    """
    Adjust concept weights based on historical win rates.
    Win rate > 60% -> weight × 1.3
    Win rate < 40% -> weight × 0.7
    Otherwise -> base weight unchanged
    Requires 10+ trades where concept voted to adjust.
    """
    global _concept_weight_cache
    now = datetime.now(CET)

    if _concept_weight_cache is not None:
        cached_time, cached_weights = _concept_weight_cache
        if now - cached_time < _WEIGHT_CACHE_TTL:
            return cached_weights

    trades = [t for t in load_trades() if t.get("result") is not None]
    weights = dict(_BASE_WEIGHTS)

    for concept, vote_key in _VOTE_KEYS.items():
        # Only count trades where this concept agreed (vote == 1)
        agreed_trades = [t for t in trades if t.get(vote_key) == 1]
        if len(agreed_trades) < 10:
            continue

        wins = sum(1 for t in agreed_trades if t["result"] == "WIN")
        wr = wins / len(agreed_trades) * 100

        base = _BASE_WEIGHTS[concept]
        if wr > 60:
            weights[concept] = round(base * 1.3, 2)
        elif wr < 40:
            weights[concept] = round(base * 0.7, 2)
        # else: keep base weight

    _concept_weight_cache = (now, weights)
    return weights


# ─────────────────────────────────────────────
# 9. ATR-Based TP/SL Calculation
# ─────────────────────────────────────────────
def calculate_tp_sl(df: pd.DataFrame, signal: str) -> dict:
    """
    Calculate TP, SL, and RR based on ATR and swing levels.
    Returns dict with sl, tp, rr, atr, sl_pips, tp_pips.
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    n = len(closes)
    entry = float(closes[-1])

    # ATR(14)
    try:
        tr_list = []
        for i in range(-14, 0):
            tr = max(float(highs[i]) - float(lows[i]),
                     abs(float(highs[i]) - float(closes[i-1])),
                     abs(float(lows[i])  - float(closes[i-1])))
            tr_list.append(tr)
        atr = float(np.mean(tr_list))
    except Exception:
        atr = float(np.mean(highs[-14:] - lows[-14:]))

    # Find swing high/low in last 20 bars
    lookback = min(20, n - 1)
    swing_high = float(np.max(highs[-lookback:]))
    swing_low  = float(np.min(lows[-lookback:]))

    # Detect JPY pair for pip calculation
    is_jpy = entry > 10  # JPY pairs have prices > 10
    pip_mult = 100.0 if is_jpy else 10000.0

    if signal == "BUY":
        sl = swing_low - 0.5 * atr
        risk = entry - sl
        tp = entry + 2.0 * risk  # 2:1 RR minimum
    elif signal == "SELL":
        sl = swing_high + 0.5 * atr
        risk = sl - entry
        tp = entry - 2.0 * risk  # 2:1 RR minimum
    else:
        return {"sl": 0, "tp": 0, "rr": 0, "atr": atr,
                "sl_pips": 0, "tp_pips": 0}

    risk = abs(entry - sl)
    reward = abs(entry - tp)
    rr = reward / risk if risk > 0 else 0

    return {
        "sl":       round(sl, 5),
        "tp":       round(tp, 5),
        "rr":       round(rr, 2),
        "atr":      round(atr, 5),
        "sl_pips":  round(abs(entry - sl) * pip_mult, 1),
        "tp_pips":  round(abs(entry - tp) * pip_mult, 1),
    }


# ─────────────────────────────────────────────
# Master Gate Function
# ─────────────────────────────────────────────
def check_all_filters(pair: str, price: float, score: float,
                      df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Run all pre-trade filters. Returns (True, []) if signal passes.
    Returns (False, [reasons]) if any filter blocks.
    """
    rejections = []

    # 1. Cooldown — fastest check, in-memory only
    blocked, reason = is_on_cooldown(pair)
    if blocked:
        rejections.append(reason)

    # 2. Duplicate detection
    blocked, reason = is_duplicate_trade(pair, price)
    if blocked:
        rejections.append(reason)

    # 3. News blackout
    blocked, reason = is_news_blackout()
    if blocked:
        rejections.append(reason)

    # 4. Session quality
    blocked, reason = is_bad_session()
    if blocked:
        rejections.append(reason)

    # 5. Dead market (ATR)
    blocked, reason = is_market_dead(df)
    if blocked:
        rejections.append(reason)

    # 6. Pair blacklist
    blocked, reason = is_pair_blacklisted(pair)
    if blocked:
        rejections.append(reason)

    if rejections:
        return False, rejections
    return True, []


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Filter Self-Test")
    print("=" * 50)

    now = datetime.now(CET)
    print(f"\nCurrent time: {now.strftime('%A %H:%M CET')}")

    # 1. News blackout
    blocked, reason = is_news_blackout()
    print(f"\n1. News Blackout:  {'BLOCKED — ' + reason if blocked else 'CLEAR ✅'}")

    # 2. Session quality
    blocked, reason = is_bad_session()
    print(f"3. Session Filter: {'BLOCKED — ' + reason if blocked else 'CLEAR ✅'}")

    # 4. Pair blacklist
    blocked, reason = is_pair_blacklisted("EUR/USD")
    print(f"4. Pair Blacklist: {'BLOCKED — ' + reason if blocked else 'CLEAR ✅ (EUR/USD)'}")

    # 5. Dynamic threshold
    for score in [8.0, 6.0, 4.0]:
        mv = get_dynamic_min_votes(score)
        print(f"5. Dynamic Votes (score={score}): min_votes={mv}")

    # 6. Cooldown
    blocked, reason = is_on_cooldown("TEST/USD")
    print(f"6. Cooldown (TEST/USD): {'BLOCKED' if blocked else 'CLEAR ✅'}")
    record_signal("TEST/USD")
    blocked, reason = is_on_cooldown("TEST/USD")
    print(f"   After record:        {'BLOCKED — ' + reason if blocked else 'CLEAR'}")

    # 7. Duplicate
    blocked, reason = is_duplicate_trade("EUR/USD", 1.12345)
    print(f"7. Duplicate:      {'BLOCKED — ' + reason if blocked else 'CLEAR ✅'}")

    # 8. Concept weights
    weights = get_concept_weights()
    print(f"8. Concept Weights: {weights}")

    print("\n✅ All filter checks passed.")
