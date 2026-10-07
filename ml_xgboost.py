import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
import pytz
from sklearn.model_selection import StratifiedKFold, cross_val_score
from ml_logger import load_trades
from confluence import ConceptVote

CET = pytz.timezone("Europe/Berlin")

# ---------------------------------------------------------------------------
# TUNEABLE VALUES
# ---------------------------------------------------------------------------
MIN_TRADES = 50        # Minimum closed trades before /train is allowed.
                       # Higher = more reliable model, but you wait longer.

CV_FOLDS = 5           # Cross-validation folds. The data is split into this many
                       # parts; the model trains on all but one and is tested on
                       # the held-out part. Gives an honest accuracy estimate
                       # (training accuracy alone is always inflated).

MIN_CV_EDGE = None     # None = always save the model after training.
                       # Set e.g. 0.05 to refuse saving a model whose CV accuracy
                       # does not beat the "always guess the majority class"
                       # baseline by at least 5 percentage points.

PROB_HIGH = 0.60       # Win probability above this -> model votes WITH the signal.
PROB_LOW = 0.40        # Win probability below this -> model votes AGAINST the signal.
                       # Between the two -> NEUTRAL. Widen the gap to make the
                       # model vote less often.

VOTE_WEIGHT = 2.0      # Weight of a decisive XGBoost vote (BUY/SELL).
NEUTRAL_WEIGHT = 1.0   # Weight of a NEUTRAL XGBoost vote.

# XGBoost model parameters
N_ESTIMATORS = 100     # Number of trees. More = can fit more, overfits easier.
LEARNING_RATE = 0.05   # Step size per tree. Lower = slower, more stable.
MAX_DEPTH = 3          # Tree depth. Keep small with little data.
# ---------------------------------------------------------------------------


def get_model_path():
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "xgboost_model.json")

# The exact features we extract for training and inference
FEATURES = [
    "hour_cet", "weekday", "is_london", "is_ny", "is_overlap", "is_tokyo",
    "smc_vote", "ict_vote", "wyckoff_vote", "volprofile_vote", "divergence_vote",
    "ict_adv_vote", "structure_vote",
    "rsi", "macd_hist", "bb_position", "ema_diff_pct", "volume_ratio", "atr_pct"
]


def _new_model():
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42,
        eval_metric="logloss"
    )


def train_xgboost() -> tuple[bool, str]:
    """Train XGBoost model on historical closed trades."""
    global _loaded_model

    trades = [t for t in load_trades() if t.get("result") is not None]
    if len(trades) < MIN_TRADES:
        return False, f"Not enough trades to train. Have {len(trades)}, need at least {MIN_TRADES}."

    df = pd.DataFrame(trades)

    # Make sure every feature column exists in the stored trades
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return False, f"Trade data is missing feature columns: {', '.join(missing)}"

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Target: 1 for WIN, 0 for LOSS
    y = (df["result"] == "WIN").astype(int)

    wins = int(y.sum())
    losses = int(len(y) - wins)
    if wins == 0 or losses == 0:
        return False, f"Need both WIN and LOSS trades to train. Have {wins} wins, {losses} losses."

    baseline = max(wins, losses) / len(y) * 100  # accuracy of always guessing the majority

    try:
        # Honest accuracy estimate via cross-validation
        cv_acc = None
        n_splits = min(CV_FOLDS, wins, losses)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(_new_model(), X, y, cv=cv, scoring="accuracy")
            cv_acc = float(np.mean(scores)) * 100

        if MIN_CV_EDGE is not None and cv_acc is not None:
            if cv_acc < baseline + MIN_CV_EDGE * 100:
                return False, (
                    f"Model not saved: CV accuracy {cv_acc:.1f}% does not beat baseline "
                    f"{baseline:.1f}% by {MIN_CV_EDGE * 100:.0f} points. Collect more trades."
                )

        # Final model on all data
        model = _new_model()
        model.fit(X, y)
        model.save_model(get_model_path())
        _loaded_model = None  # force reload so the bot uses the new model without a restart

        train_acc = float(np.mean(model.predict(X) == y)) * 100
        cv_text = f"{cv_acc:.1f}%" if cv_acc is not None else "n/a"
        return True, (
            f"Trained on {len(trades)} trades ({wins} wins, {losses} losses).\n"
            f"Training accuracy: {train_acc:.1f}% (inflated, model has seen this data)\n"
            f"Cross-validated accuracy: {cv_text}\n"
            f"Baseline (always guess majority): {baseline:.1f}%\n"
            f"If CV accuracy is not clearly above baseline, the model is mostly noise."
        )
    except Exception as e:
        return False, f"XGBoost training failed: {e}"

_loaded_model = None

def load_xgboost_model():
    global _loaded_model
    path = get_model_path()
    if not os.path.exists(path):
        return None

    if _loaded_model is None:
        try:
            _loaded_model = xgb.XGBClassifier()
            _loaded_model.load_model(path)
        except Exception:
            _loaded_model = None
            return None

    return _loaded_model

def _encode_vote(vote_str: str, signal: str) -> int:
    if vote_str == signal: return 1
    if vote_str == "NEUTRAL": return 0
    return -1

def get_ml_vote(snapshot: dict, votes: list[ConceptVote], signal: str) -> ConceptVote:
    """Predicts trade success probability and returns an 8th ML vote."""
    model = load_xgboost_model()
    if model is None:
        return ConceptVote("XGBoost", "NEUTRAL", NEUTRAL_WEIGHT, "Model not trained yet", "Reach 100 trades to activate")

    now = datetime.now(CET)
    hour = now.hour
    weekday = now.weekday()

    # Prepare feature vector exactly matching training
    vote_map = {v.name: v.vote for v in votes}

    row = {
        "hour_cet": hour,
        "weekday": weekday,
        "is_london": 1 if 8 <= hour < 17 else 0,
        "is_ny": 1 if 13 <= hour < 22 else 0,
        "is_overlap": 1 if 13 <= hour < 17 else 0,
        "is_tokyo": 1 if 0 <= hour < 9 else 0,
        "smc_vote": _encode_vote(vote_map.get("SMC", "NEUTRAL"), signal),
        "ict_vote": _encode_vote(vote_map.get("ICT", "NEUTRAL"), signal),
        "wyckoff_vote": _encode_vote(vote_map.get("Wyckoff", "NEUTRAL"), signal),
        "volprofile_vote": _encode_vote(vote_map.get("VolumeProfile", "NEUTRAL"), signal),
        "divergence_vote": _encode_vote(vote_map.get("Divergence", "NEUTRAL"), signal),
        "ict_adv_vote": _encode_vote(vote_map.get("ICT_Adv", "NEUTRAL"), signal),
        "structure_vote": _encode_vote(vote_map.get("Structure", "NEUTRAL"), signal),
        "rsi": snapshot.get("rsi", 50.0),
        "macd_hist": snapshot.get("macd_hist", 0.0),
        "bb_position": snapshot.get("bb_position", 0.5),
        "ema_diff_pct": snapshot.get("ema_diff_pct", 0.0),
        "volume_ratio": snapshot.get("volume_ratio", 1.0),
        "atr_pct": snapshot.get("atr_pct", 0.0)
    }

    # Convert to DataFrame to preserve feature names (which XGBoost requires)
    X = pd.DataFrame([row])[FEATURES].fillna(0)

    try:
        # Get probability of class 1 (WIN)
        prob = model.predict_proba(X)[0][1]

        if prob > PROB_HIGH:
            return ConceptVote("XGBoost", signal, VOTE_WEIGHT, f"High win probability ({prob*100:.1f}%)", "Historical data strongly favors this setup")
        elif prob < PROB_LOW:
            opp_signal = "SELL" if signal == "BUY" else "BUY"
            return ConceptVote("XGBoost", opp_signal, VOTE_WEIGHT, f"Low win probability ({prob*100:.1f}%)", "Historical data rejects this setup")
        else:
            return ConceptVote("XGBoost", "NEUTRAL", NEUTRAL_WEIGHT, f"Uncertain probability ({prob*100:.1f}%)", "Not enough historical edge")

    except Exception as e:
        return ConceptVote("XGBoost", "NEUTRAL", NEUTRAL_WEIGHT, "Prediction error", str(e))