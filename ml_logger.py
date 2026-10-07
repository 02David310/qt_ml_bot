"""
ML Trade Logger
Stores rich feature vectors for every signal — ready for XGBoost training later.
Each trade is one row in trades.json with ALL features + WIN/LOSS label.
"""
import json
import os
import sys
from datetime import datetime
import pytz

CET = pytz.timezone("Europe/Berlin")


def get_data_path():
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "trades.json")


def load_trades() -> list:
    path = get_data_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_trades(trades: list):
    with open(get_data_path(), "w") as f:
        json.dump(trades, f, indent=2)


def log_trade(signal_id: str, pair: str, signal: str, price: float,
              confluence_result, df_snapshot: dict) -> dict:
    """
    Log a signal with ALL features for ML training.
    df_snapshot contains pre-calculated values from the dataframe.
    """
    now     = datetime.now(CET)
    hour    = now.hour
    weekday = now.weekday()  # 0=Mon, 6=Sun

    # Extract votes
    votes = {v.name: v.vote for v in confluence_result.votes}
    weights = {v.name: v.weight for v in confluence_result.votes}

    trade = {
        # Identity
        "id":           signal_id,
        "pair":         pair,
        "signal":       signal,
        "price":        price,
        "timestamp":    now.isoformat(),
        "result":       None,   # WIN / LOSS — filled later

        # Time features
        "hour_cet":     hour,
        "weekday":      weekday,
        "is_london":    1 if 8  <= hour < 17 else 0,
        "is_ny":        1 if 13 <= hour < 22 else 0,
        "is_overlap":   1 if 13 <= hour < 17 else 0,
        "is_tokyo":     1 if 0  <= hour < 9  else 0,

        # Confluence scores
        "buy_votes":    confluence_result.buy_votes,
        "sell_votes":   confluence_result.sell_votes,
        "neutral_votes":confluence_result.neutral_votes,
        "score":        confluence_result.score,
        "confidence":   confluence_result.confidence,

        # Individual concept votes (1=agree with signal, 0=neutral, -1=disagree)
        "smc_vote":        _encode_vote(votes.get("SMC",          "NEUTRAL"), signal),
        "ict_vote":        _encode_vote(votes.get("ICT",          "NEUTRAL"), signal),
        "wyckoff_vote":    _encode_vote(votes.get("Wyckoff",      "NEUTRAL"), signal),
        "volprofile_vote": _encode_vote(votes.get("VolumeProfile","NEUTRAL"), signal),
        "divergence_vote": _encode_vote(votes.get("Divergence",   "NEUTRAL"), signal),
        "ict_adv_vote":    _encode_vote(votes.get("ICT_Adv",      "NEUTRAL"), signal),
        "structure_vote":  _encode_vote(votes.get("Structure",    "NEUTRAL"), signal),

        # Raw price features from snapshot
        "rsi":          df_snapshot.get("rsi", 50.0),
        "macd_hist":    df_snapshot.get("macd_hist", 0.0),
        "bb_position":  df_snapshot.get("bb_position", 0.5),  # 0=lower, 1=upper
        "ema_diff_pct": df_snapshot.get("ema_diff_pct", 0.0), # (ema9-ema21)/price
        "volume_ratio": df_snapshot.get("volume_ratio", 1.0), # vol/avg_vol
        "atr_pct":      df_snapshot.get("atr_pct", 0.0),      # atr/price %
        "candle_pattern": df_snapshot.get("candle_pattern", "none"),

        # Pair encoding (for ML — one-hot done at training time)
        "pair_base":    pair[:3],
        "pair_quote":   pair[4:],
    }

    trades = load_trades()
    trades.append(trade)
    save_trades(trades)
    return trade


def _encode_vote(vote: str, signal: str) -> int:
    """1 = agrees with signal, 0 = neutral, -1 = disagrees."""
    if vote == signal:
        return 1
    if vote == "NEUTRAL":
        return 0
    return -1


def mark_result(signal_id: str, result: str) -> dict | None:
    """Mark WIN/LOSS. Returns the trade dict if found."""
    trades = load_trades()
    for t in trades:
        if t["id"] == signal_id and t["result"] is None:
            t["result"] = result.upper()
            save_trades(trades)
            return t
    return None


def get_pending_trades() -> list:
    return [t for t in load_trades() if t["result"] is None]

def delete_trade(signal_id: str) -> bool:
    """Delete a trade entirely (useful for weekend/closed market signals)."""
    trades = load_trades()
    filtered = [t for t in trades if t["id"] != signal_id]
    if len(filtered) < len(trades):
        save_trades(filtered)
        return True
    return False

def delete_all_pending() -> int:
    """Delete all trades that don't have a result yet."""
    trades = load_trades()
    filtered = [t for t in trades if t["result"] is not None]
    deleted = len(trades) - len(filtered)
    if deleted > 0:
        save_trades(filtered)
    return deleted


def get_stats() -> dict:
    trades  = load_trades()
    closed  = [t for t in trades if t["result"] is not None]
    wins    = [t for t in closed if t["result"] == "WIN"]
    losses  = [t for t in closed if t["result"] == "LOSS"]
    pending = len(trades) - len(closed)
    win_rate = len(wins) / len(closed) * 100 if closed else 0

    # Stats by pair
    pair_stats = {}
    for t in closed:
        p = t["pair"]
        if p not in pair_stats:
            pair_stats[p] = {"wins": 0, "losses": 0}
        pair_stats[p]["WIN" == t["result"] and "wins" or "losses"] + 1
        if t["result"] == "WIN":
            pair_stats[p]["wins"] += 1
        else:
            pair_stats[p]["losses"] += 1

    # Stats by confidence
    conf_stats = {}
    for t in closed:
        c = t.get("confidence", "UNKNOWN")
        if c not in conf_stats:
            conf_stats[c] = {"wins": 0, "losses": 0}
        if t["result"] == "WIN":
            conf_stats[c]["wins"] += 1
        else:
            conf_stats[c]["losses"] += 1

    # Stats by hour
    hour_wins = {}
    for t in wins:
        h = t.get("hour_cet", 0)
        hour_wins[h] = hour_wins.get(h, 0) + 1

    best_hour = max(hour_wins, key=hour_wins.get) if hour_wins else None

    # Concept win rates
    concept_win_rates = {}
    vote_keys = {
        "SMC": "smc_vote", "ICT": "ict_vote", "Wyckoff": "wyckoff_vote",
        "VolumeProfile": "volprofile_vote", "Divergence": "divergence_vote",
        "ICT_Adv": "ict_adv_vote", "Structure": "structure_vote",
    }
    for concept, key in vote_keys.items():
        agreed = [t for t in closed if t.get(key) == 1]
        if len(agreed) >= 3:
            cw = sum(1 for t in agreed if t["result"] == "WIN")
            concept_win_rates[concept] = {
                "wins": cw, "total": len(agreed),
                "rate": round(cw / len(agreed) * 100, 1),
            }

    return {
        "total":             len(trades),
        "closed":            len(closed),
        "pending":           pending,
        "wins":              len(wins),
        "losses":            len(losses),
        "win_rate":          win_rate,
        "pair_stats":        pair_stats,
        "conf_stats":        conf_stats,
        "best_hour":         best_hour,
        "concept_win_rates": concept_win_rates,
        "last_10":           closed[-10:] if len(closed) >= 10 else closed,
        "ml_ready":          len(closed) >= 100,  # flag when enough data
    }


def export_for_ml() -> str:
    """Export trades as CSV string for ML training."""
    import io
    import csv

    trades = [t for t in load_trades() if t["result"] is not None]
    if not trades:
        return ""

    # Define ML features
    feature_cols = [
        "hour_cet", "weekday", "is_london", "is_ny", "is_overlap", "is_tokyo",
        "buy_votes", "sell_votes", "neutral_votes", "score",
        "smc_vote", "ict_vote", "wyckoff_vote", "volprofile_vote", "divergence_vote",
        "ict_adv_vote", "structure_vote",
        "rsi", "macd_hist", "bb_position", "ema_diff_pct", "volume_ratio", "atr_pct",
        "pair_base", "pair_quote",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=feature_cols + ["result"])
    writer.writeheader()

    for t in trades:
        row = {col: t.get(col, "") for col in feature_cols}
        row["result"] = 1 if t["result"] == "WIN" else 0
        writer.writerow(row)

    return output.getvalue()
