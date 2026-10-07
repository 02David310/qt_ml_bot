"""
Confluence Engine — combines SMC, ICT, Wyckoff, Volume Profile, Divergence
Each concept votes independently. Signal fires only when enough concepts agree.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import pytz

CET = pytz.timezone("Europe/Berlin")


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class ConceptVote:
    name: str
    vote: str        # BUY / SELL / NEUTRAL
    weight: float    # 1.0 = normal, 1.5 = strong concept
    reason: str
    detail: str = ""


@dataclass
class ConfluenceResult:
    signal: str          # BUY / SELL / HOLD
    score: float         # weighted confluence score 0-10
    confidence: str      # LOW / MEDIUM / HIGH / VERY HIGH
    votes: list = field(default_factory=list)
    buy_votes: int = 0
    sell_votes: int = 0
    neutral_votes: int = 0
    reasons: list = field(default_factory=list)


# ─────────────────────────────────────────────
# SMC — Smart Money Concepts
# ─────────────────────────────────────────────
def smc_analysis(df: pd.DataFrame) -> ConceptVote:
    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    n      = len(closes)

    if n < 20:
        return ConceptVote("SMC", "NEUTRAL", 1.5, "Not enough data", "")

    buy_score = sell_score = 0
    details   = []

    # Market structure
    recent_highs = highs[-20:]
    recent_lows  = lows[-20:]
    swing_highs, swing_lows = [], []
    for i in range(2, len(recent_highs) - 2):
        if recent_highs[i] > max(recent_highs[i-2:i]) and recent_highs[i] > max(recent_highs[i+1:i+3]):
            swing_highs.append(recent_highs[i])
        if recent_lows[i] < min(recent_lows[i-2:i]) and recent_lows[i] < min(recent_lows[i+1:i+3]):
            swing_lows.append(recent_lows[i])

    structure = "ranging"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
            structure = "bullish"; buy_score += 1; details.append("Bullish structure HH/HL")
        elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
            structure = "bearish"; sell_score += 1; details.append("Bearish structure LH/LL")

    # Break of Structure
    if n >= 10:
        lookback     = min(15, n - 3)
        recent_high  = float(np.max(highs[-lookback:-2]))
        recent_low   = float(np.min(lows[-lookback:-2]))
        cur          = float(closes[-1])
        if structure == "bearish" and cur > recent_high:
            buy_score += 2; details.append("BOS bullish — trend flip")
        elif structure == "bullish" and cur < recent_low:
            sell_score += 2; details.append("BOS bearish — trend flip")

    # Order Block
    for i in range(n - 3, max(n - 20, 2), -1):
        body_i = closes[i] - opens[i]
        if body_i < 0 and closes[i+1] > opens[i+1] and closes[i+2] > opens[i+2]:
            move = closes[i+2] - closes[i]
            if move > abs(body_i) * 1.5:
                ob = (opens[i] + closes[i]) / 2
                if abs(float(closes[-1]) - ob) / float(closes[-1]) < 0.002:
                    buy_score += 3; details.append(f"Bullish OB at {ob:.5f}")
                break
        if body_i > 0 and closes[i+1] < opens[i+1] and closes[i+2] < opens[i+2]:
            move = closes[i] - closes[i+2]
            if move > abs(body_i) * 1.5:
                ob = (opens[i] + closes[i]) / 2
                if abs(float(closes[-1]) - ob) / float(closes[-1]) < 0.002:
                    sell_score += 3; details.append(f"Bearish OB at {ob:.5f}")
                break

    # Fair Value Gap
    for i in range(n - 1, max(n - 6, 2), -1):
        if lows[i] > highs[i-2]:
            buy_score += 2; details.append("Bullish FVG present"); break
        if highs[i] < lows[i-2]:
            sell_score += 2; details.append("Bearish FVG present"); break

    # Liquidity sweep
    if n >= 10:
        lookback     = min(10, n - 3)
        prev_high    = float(np.max(highs[-lookback:-3]))
        prev_low     = float(np.min(lows[-lookback:-3]))
        if highs[-2] > prev_high and closes[-2] < prev_high and closes[-1] < closes[-2]:
            sell_score += 3; details.append("Bearish liquidity sweep")
        elif lows[-2] < prev_low and closes[-2] > prev_low and closes[-1] > closes[-2]:
            buy_score += 3; details.append("Bullish liquidity sweep")

    if buy_score > sell_score and buy_score >= 3:
        return ConceptVote("SMC", "BUY", 1.5, f"Score {buy_score}", " | ".join(details))
    if sell_score > buy_score and sell_score >= 3:
        return ConceptVote("SMC", "SELL", 1.5, f"Score {sell_score}", " | ".join(details))
    return ConceptVote("SMC", "NEUTRAL", 1.5, "No clear SMC setup", " | ".join(details))


# ─────────────────────────────────────────────
# ICT — Inner Circle Trader
# ─────────────────────────────────────────────
def ict_analysis(df: pd.DataFrame) -> ConceptVote:
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    n      = len(closes)

    if n < 30:
        return ConceptVote("ICT", "NEUTRAL", 1.5, "Not enough data", "")

    buy_score = sell_score = 0
    details   = []

    # Kill Zone check (CET times)
    now = datetime.now(CET)
    hour = now.hour
    in_kill_zone = False
    kz_name = ""
    if 8 <= hour < 10:
        in_kill_zone = True; kz_name = "London Open KZ"
    elif 13 <= hour < 15:
        in_kill_zone = True; kz_name = "NY Open KZ"
    elif 16 <= hour < 17:
        in_kill_zone = True; kz_name = "London Close KZ"

    if in_kill_zone:
        buy_score += 1; sell_score += 1  # both get boost, direction decided by other factors
        details.append(f"In {kz_name} — high probability window")

    # OTE — Optimal Trade Entry (Fibonacci 61.8-79% retracement)
    if n >= 20:
        swing_high = float(np.max(highs[-20:]))
        swing_low  = float(np.min(lows[-20:]))
        swing_range = swing_high - swing_low
        price = float(closes[-1])

        ote_buy_low  = swing_low  + swing_range * 0.618
        ote_buy_high = swing_low  + swing_range * 0.79
        ote_sel_low  = swing_high - swing_range * 0.79
        ote_sel_high = swing_high - swing_range * 0.618

        if ote_buy_low <= price <= ote_buy_high:
            buy_score += 3; details.append(f"In OTE buy zone ({ote_buy_low:.5f}-{ote_buy_high:.5f})")
        elif ote_sel_low <= price <= ote_sel_high:
            sell_score += 3; details.append(f"In OTE sell zone ({ote_sel_low:.5f}-{ote_sel_high:.5f})")

    # Power of 3 — accumulation → manipulation → distribution
    # Look for: tight range (accumulation), then spike (manipulation), then strong move (distribution)
    if n >= 15:
        acc_range  = float(np.max(highs[-15:-10])) - float(np.min(lows[-15:-10]))
        manip_move = float(highs[-5]) - float(lows[-5])
        last_close = float(closes[-1])
        last_open  = float(df["Open"].values[-1])

        if acc_range > 0 and manip_move > acc_range * 1.5:
            # Manipulation spike detected — distribution should follow
            if last_close > last_open:
                buy_score += 2; details.append("Power of 3: bullish distribution phase")
            else:
                sell_score += 2; details.append("Power of 3: bearish distribution phase")

    # Inducement — fake breakout of minor level before real move
    if n >= 8:
        minor_high = float(np.max(highs[-8:-4]))
        minor_low  = float(np.min(lows[-8:-4]))
        # Price swept above minor high then closed back below → bearish inducement
        if float(highs[-3]) > minor_high and float(closes[-2]) < minor_high and float(closes[-1]) < closes[-2]:
            sell_score += 2; details.append("Bearish inducement — fake breakout above minor high")
        # Price swept below minor low then closed back above → bullish inducement
        if float(lows[-3]) < minor_low and float(closes[-2]) > minor_low and float(closes[-1]) > closes[-2]:
            buy_score += 2; details.append("Bullish inducement — fake breakdown below minor low")

    if buy_score > sell_score and buy_score >= 3:
        return ConceptVote("ICT", "BUY", 1.5, f"Score {buy_score}", " | ".join(details))
    if sell_score > buy_score and sell_score >= 3:
        return ConceptVote("ICT", "SELL", 1.5, f"Score {sell_score}", " | ".join(details))
    return ConceptVote("ICT", "NEUTRAL", 1.5, "No clear ICT setup", " | ".join(details))


# ─────────────────────────────────────────────
# Wyckoff — Accumulation / Distribution
# ─────────────────────────────────────────────
def wyckoff_analysis(df: pd.DataFrame) -> ConceptVote:
    closes  = df["Close"].values
    highs   = df["High"].values
    lows    = df["Low"].values
    volumes = df["Volume"].values
    n       = len(closes)

    if n < 30:
        return ConceptVote("Wyckoff", "NEUTRAL", 1.0, "Not enough data", "")

    details   = []
    buy_score = sell_score = 0

    # Spring — price briefly dips below support then reverses up on high volume
    # (Wyckoff accumulation Phase C)
    support   = float(np.min(lows[-30:-5]))
    last_low  = float(lows[-3])
    last_close = float(closes[-1])
    avg_vol   = float(np.mean(volumes[-20:]))
    last_vol  = float(volumes[-3])

    if last_low < support and last_close > support and last_vol > avg_vol * 1.2:
        buy_score += 3; details.append(f"Wyckoff Spring — dipped below support {support:.5f} then recovered")

    # Upthrust — price briefly spikes above resistance then reverses down
    # (Wyckoff distribution Phase C)
    resistance  = float(np.max(highs[-30:-5]))
    last_high   = float(highs[-3])
    last_close2 = float(closes[-2])

    if last_high > resistance and last_close2 < resistance and last_vol > avg_vol * 1.2:
        sell_score += 3; details.append(f"Wyckoff Upthrust — spiked above resistance {resistance:.5f} then reversed")

    # Accumulation range — tight consolidation with declining volume
    recent_range  = float(np.max(highs[-10:])) - float(np.min(lows[-10:]))
    broader_range = float(np.max(highs[-30:])) - float(np.min(lows[-30:]))
    recent_vol    = float(np.mean(volumes[-10:]))
    broader_vol   = float(np.mean(volumes[-30:]))

    if recent_range < broader_range * 0.3 and recent_vol < broader_vol * 0.7:
        # Tight range + low volume = accumulation or distribution
        # Determine which by trend direction
        trend = float(closes[-1]) - float(closes[-20])
        if trend > 0:
            buy_score += 1; details.append("Wyckoff accumulation range — tight consolidation before markup")
        else:
            sell_score += 1; details.append("Wyckoff distribution range — tight consolidation before markdown")

    # Sign of Strength / Weakness
    last_3_closes = closes[-3:]
    last_3_vols   = volumes[-3:]
    if all(last_3_closes[i] > last_3_closes[i-1] for i in range(1, 3)) and \
       all(last_3_vols[i] >= last_3_vols[i-1] for i in range(1, 3)):
        buy_score += 2; details.append("Wyckoff Sign of Strength — rising closes on rising volume")
    elif all(last_3_closes[i] < last_3_closes[i-1] for i in range(1, 3)) and \
         all(last_3_vols[i] >= last_3_vols[i-1] for i in range(1, 3)):
        sell_score += 2; details.append("Wyckoff Sign of Weakness — falling closes on rising volume")

    if buy_score > sell_score and buy_score >= 2:
        return ConceptVote("Wyckoff", "BUY", 1.0, f"Score {buy_score}", " | ".join(details))
    if sell_score > buy_score and sell_score >= 2:
        return ConceptVote("Wyckoff", "SELL", 1.0, f"Score {sell_score}", " | ".join(details))
    return ConceptVote("Wyckoff", "NEUTRAL", 1.0, "No clear Wyckoff phase", " | ".join(details))


# ─────────────────────────────────────────────
# Volume Profile — High Volume Nodes
# ─────────────────────────────────────────────
def volume_profile_analysis(df: pd.DataFrame) -> ConceptVote:
    closes  = df["Close"].values
    volumes = df["Volume"].values
    n       = len(closes)

    if n < 20:
        return ConceptVote("VolumeProfile", "NEUTRAL", 1.0, "Not enough data", "")

    details   = []
    buy_score = sell_score = 0
    price     = float(closes[-1])

    # Build simple volume profile — bin prices and sum volume
    price_range = np.max(closes[-50:]) - np.min(closes[-50:])
    if price_range == 0:
        return ConceptVote("VolumeProfile", "NEUTRAL", 1.0, "No price range", "")

    bins       = 20
    bin_size   = price_range / bins
    min_price  = float(np.min(closes[-50:]))
    vol_by_bin = np.zeros(bins)

    for i in range(min(50, n)):
        bin_idx = int((closes[-(i+1)] - min_price) / bin_size)
        bin_idx = min(bin_idx, bins - 1)
        vol_by_bin[bin_idx] += volumes[-(i+1)]

    # Point of Control — highest volume bin
    poc_bin   = int(np.argmax(vol_by_bin))
    poc_price = min_price + (poc_bin + 0.5) * bin_size
    poc_dist  = abs(price - poc_price) / price

    # High Volume Node acts as support/resistance
    if poc_dist < 0.003:
        # Price at POC — likely to bounce or break
        trend = float(closes[-1]) - float(closes[-5])
        if trend > 0:
            buy_score += 2; details.append(f"Price at Point of Control {poc_price:.5f} — acting as support")
        else:
            sell_score += 2; details.append(f"Price at Point of Control {poc_price:.5f} — acting as resistance")

    # Low Volume Node — price moves fast through these areas
    current_bin = int((price - min_price) / bin_size)
    current_bin = min(max(current_bin, 0), bins - 1)
    avg_vol_bin = np.mean(vol_by_bin)

    if vol_by_bin[current_bin] < avg_vol_bin * 0.3:
        # In a low volume node — trend likely to continue
        trend = float(closes[-1]) - float(closes[-3])
        if trend > 0:
            buy_score += 1; details.append("In Low Volume Node — upward momentum likely to continue")
        else:
            sell_score += 1; details.append("In Low Volume Node — downward momentum likely to continue")

    # Volume trend — increasing volume confirms direction
    recent_vol  = float(np.mean(volumes[-5:]))
    previous_vol = float(np.mean(volumes[-15:-5]))
    if recent_vol > previous_vol * 1.3:
        trend = float(closes[-1]) - float(closes[-5])
        if trend > 0:
            buy_score += 1; details.append("Volume increasing on upward move — buyers in control")
        else:
            sell_score += 1; details.append("Volume increasing on downward move — sellers in control")

    if buy_score > sell_score and buy_score >= 2:
        return ConceptVote("VolumeProfile", "BUY", 1.0, f"Score {buy_score}", " | ".join(details))
    if sell_score > buy_score and sell_score >= 2:
        return ConceptVote("VolumeProfile", "SELL", 1.0, f"Score {sell_score}", " | ".join(details))
    return ConceptVote("VolumeProfile", "NEUTRAL", 1.0, "No clear volume signal", " | ".join(details))


# ─────────────────────────────────────────────
# Divergence — RSI divergence
# ─────────────────────────────────────────────
def divergence_analysis(df: pd.DataFrame) -> ConceptVote:
    closes = df["Close"].values
    n      = len(closes)

    if n < 30:
        return ConceptVote("Divergence", "NEUTRAL", 1.0, "Not enough data", "")

    # Calculate RSI manually (no ta import needed here)
    delta  = np.diff(closes)
    gain   = np.where(delta > 0, delta, 0)
    loss   = np.where(delta < 0, -delta, 0)
    period = 14

    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))

    if len(rsi) < 15:
        return ConceptVote("Divergence", "NEUTRAL", 1.0, "RSI too short", "")

    details   = []
    buy_score = sell_score = 0

    # Find recent swing highs/lows in price and RSI
    price_recent   = closes[-15:]
    rsi_recent     = rsi[-15:]

    price_high1_idx = int(np.argmax(price_recent[:8]))
    price_high2_idx = int(np.argmax(price_recent[7:])) + 7
    rsi_high1       = float(rsi_recent[price_high1_idx])
    rsi_high2       = float(rsi_recent[price_high2_idx])

    price_low1_idx  = int(np.argmin(price_recent[:8]))
    price_low2_idx  = int(np.argmin(price_recent[7:])) + 7
    rsi_low1        = float(rsi_recent[price_low1_idx])
    rsi_low2        = float(rsi_recent[price_low2_idx])

    # Bearish divergence: price makes higher high but RSI makes lower high
    if price_recent[price_high2_idx] > price_recent[price_high1_idx] and rsi_high2 < rsi_high1 - 3:
        sell_score += 3; details.append(f"Bearish RSI divergence — price HH but RSI LH ({rsi_high1:.1f}→{rsi_high2:.1f})")

    # Bullish divergence: price makes lower low but RSI makes higher low
    if price_recent[price_low2_idx] < price_recent[price_low1_idx] and rsi_low2 > rsi_low1 + 3:
        buy_score += 3; details.append(f"Bullish RSI divergence — price LL but RSI HL ({rsi_low1:.1f}→{rsi_low2:.1f})")

    # Extreme RSI levels
    current_rsi = float(rsi[-1])
    if current_rsi < 30:
        buy_score += 1; details.append(f"RSI oversold at {current_rsi:.1f}")
    elif current_rsi > 70:
        sell_score += 1; details.append(f"RSI overbought at {current_rsi:.1f}")

    if buy_score > sell_score and buy_score >= 2:
        return ConceptVote("Divergence", "BUY", 1.0, f"Score {buy_score}", " | ".join(details))
    if sell_score > buy_score and sell_score >= 2:
        return ConceptVote("Divergence", "SELL", 1.0, f"Score {sell_score}", " | ".join(details))
    return ConceptVote("Divergence", "NEUTRAL", 1.0, "No divergence detected", " | ".join(details))


# ─────────────────────────────────────────────
# ICT Advanced — CISD + Breaker Blocks + Silver Bullet
# ─────────────────────────────────────────────
def ict_advanced_analysis(df: pd.DataFrame) -> ConceptVote:
    """
    Concept 6: Three high-precision ICT patterns missing from the base ICT detector.

    1. CISD (Change in State of Delivery) — price closes on the wrong side of
       a key level, signalling the delivery engine has flipped.
    2. Breaker Block — a failed Order Block that price has returned to retest.
       These are extremely high-probability reaction zones.
    3. Silver Bullet window — 15:00–16:00 CET (NY 09:00–10:00 AM open).
       Backtested at 61%+ win rate; distinct from the broader kill zone.
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    n      = len(closes)

    if n < 30:
        return ConceptVote("ICT_Adv", "NEUTRAL", 1.5, "Not enough data", "")

    buy_score = sell_score = 0
    details   = []

    # ── Silver Bullet Window (15:00–16:00 CET) ──────────────────────────────
    now  = datetime.now(CET)
    hour = now.hour
    in_silver_bullet = (hour == 15)   # 15:00–15:59 CET = NY 09:00 open hour
    if in_silver_bullet:
        buy_score  += 1
        sell_score += 1
        details.append("Silver Bullet window (15:00 CET) — highest probability hour")

    # ── CISD — Change in State of Delivery ──────────────────────────────────
    # Find the last significant swing high/low over the past 20 bars.
    # CISD up: price closes ABOVE the most recent swing high → bullish delivery.
    # CISD down: price closes BELOW the most recent swing low → bearish delivery.
    if n >= 20:
        lookback     = min(20, n - 3)
        swing_high   = float(np.max(highs[-lookback:-3]))
        swing_low    = float(np.min(lows[-lookback:-3]))
        last_close   = float(closes[-1])
        prev_close   = float(closes[-2])

        # Bullish CISD: previous close was below swing high, current close is above it
        if prev_close < swing_high and last_close > swing_high:
            buy_score += 3
            details.append(f"Bullish CISD — closed above swing high {swing_high:.5f}, delivery flipped long")

        # Bearish CISD: previous close was above swing low, current close is below it
        elif prev_close > swing_low and last_close < swing_low:
            sell_score += 3
            details.append(f"Bearish CISD — closed below swing low {swing_low:.5f}, delivery flipped short")

    # ── Breaker Block Detection ──────────────────────────────────────────────
    # A Breaker Block is a former Order Block that price has since violated.
    # It now acts as a magnet for price to return and react from.
    #
    # Bullish Breaker: find a bearish OB (down-close candle before a drop)
    # that was subsequently broken by a bullish move — price returning to it = BUY.
    # Bearish Breaker: find a bullish OB broken by bearish displacement — SELL on retest.
    if n >= 25:
        price = float(closes[-1])

        # Search recent candles for violated order blocks
        for i in range(n - 15, max(n - 25, 3), -1):
            body_i = closes[i] - opens[i]

            # --- Potential bullish OB: down-close candle ---
            if body_i < 0:
                ob_top = max(opens[i], closes[i])
                ob_bot = min(opens[i], closes[i])
                # Check if a subsequent strong up-move violated the OB (making it a breaker)
                post_high = float(np.max(highs[i+1: min(i+8, n)]))
                if post_high > ob_top:                         # OB was violated → it's now a Breaker
                    # Current price retesting the breaker zone from above
                    if ob_bot <= price <= ob_top * 1.001:
                        buy_score += 3
                        details.append(f"Bullish Breaker Block retest {ob_bot:.5f}–{ob_top:.5f}")
                        break

            # --- Potential bearish OB: up-close candle ---
            if body_i > 0:
                ob_top = max(opens[i], closes[i])
                ob_bot = min(opens[i], closes[i])
                post_low = float(np.min(lows[i+1: min(i+8, n)]))
                if post_low < ob_bot:                          # OB was violated → Breaker
                    if ob_bot * 0.999 <= price <= ob_top:
                        sell_score += 3
                        details.append(f"Bearish Breaker Block retest {ob_bot:.5f}–{ob_top:.5f}")
                        break

    if buy_score > sell_score and buy_score >= 3:
        return ConceptVote("ICT_Adv", "BUY",  1.5, f"Score {buy_score}", " | ".join(details))
    if sell_score > buy_score and sell_score >= 3:
        return ConceptVote("ICT_Adv", "SELL", 1.5, f"Score {sell_score}", " | ".join(details))
    return ConceptVote("ICT_Adv", "NEUTRAL", 1.5, "No CISD/Breaker/Silver Bullet setup", " | ".join(details))


# ─────────────────────────────────────────────
# Structure — Clean MSS + OTE+FVG Unicorn + ATR Displacement
# ─────────────────────────────────────────────
def structure_analysis(df: pd.DataFrame) -> ConceptVote:
    """
    Concept 7: Three high-value structural patterns.

    1. Clean Market Structure Shift (MSS) — trend context shift confirmed by
       a sweep of the last structural low/high followed by a BOS.
       The existing SMC BOS doesn't require the prior sweep, making it noisier.

    2. OTE + FVG Unicorn — price is simultaneously inside an FVG AND inside
       the 61.8–79% Fibonacci retracement zone. This dual-filter dramatically
       reduces false entries.

    3. ATR Displacement Check — the BOS candle body must be > 1.5x ATR(14).
       Filters slow grinds that technically break structure but have no
       institutional conviction behind them.
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    n      = len(closes)

    if n < 40:
        return ConceptVote("Structure", "NEUTRAL", 1.5, "Not enough data", "")

    buy_score = sell_score = 0
    details   = []

    # ── ATR(14) baseline ────────────────────────────────────────────────────
    try:
        tr_vals = [max(highs[i] - lows[i],
                       abs(highs[i] - closes[i-1]),
                       abs(lows[i]  - closes[i-1]))
                   for i in range(-14, 0)]
        atr14 = float(np.mean(tr_vals))
    except Exception:
        atr14 = float(np.mean(highs[-14:] - lows[-14:]))

    # ── Clean MSS (Market Structure Shift) ──────────────────────────────────
    # Algorithm:
    #   Step 1: Identify prevailing trend over last 30 bars (HH/HL = bull, LH/LL = bear).
    #   Step 2: Detect a liquidity grab of the last swing extreme in that trend.
    #   Step 3: Confirm a BOS in the OPPOSITE direction with ATR displacement.
    if n >= 35:
        # Identify swing highs/lows in last 30 bars
        window = 30
        sh_list, sl_list = [], []
        for i in range(2, window - 2):
            idx = n - window + i
            if highs[idx] > highs[idx-2] and highs[idx] > highs[idx-1] \
               and highs[idx] > highs[idx+1] and highs[idx] > highs[idx+2]:
                sh_list.append((idx, highs[idx]))
            if lows[idx] < lows[idx-2] and lows[idx] < lows[idx-1] \
               and lows[idx] < lows[idx+1] and lows[idx] < lows[idx+2]:
                sl_list.append((idx, lows[idx]))

        price = float(closes[-1])

        # Bearish trend → look for bullish MSS
        # Condition: LH/LL structure → sweep of last LL → BOS up with displacement
        if len(sh_list) >= 2 and len(sl_list) >= 2:
            last_sh = sh_list[-1][1]; prev_sh = sh_list[-2][1]
            last_sl = sl_list[-1][1]; prev_sl = sl_list[-2][1]

            # Bearish trend context (LH + LL)
            if last_sh < prev_sh and last_sl < prev_sl:
                # Look for a sweep of the last LL followed by recovery above last SH
                recent_low = float(np.min(lows[-5:]))
                if recent_low < last_sl:                    # sweep of structural low
                    bos_candle_body = abs(float(closes[-1]) - float(opens[-1]))
                    if price > last_sh and bos_candle_body > atr14 * 1.5:
                        buy_score += 4
                        details.append(f"Bullish MSS — swept LL {last_sl:.5f}, BOS above {last_sh:.5f} "
                                       f"with displacement ({bos_candle_body/atr14:.1f}x ATR)")

            # Bullish trend context (HH + HL) → bearish MSS
            elif last_sh > prev_sh and last_sl > prev_sl:
                recent_high = float(np.max(highs[-5:]))
                if recent_high > last_sh:                   # sweep of structural high
                    bos_candle_body = abs(float(closes[-1]) - float(opens[-1]))
                    if price < last_sl and bos_candle_body > atr14 * 1.5:
                        sell_score += 4
                        details.append(f"Bearish MSS — swept HH {last_sh:.5f}, BOS below {last_sl:.5f} "
                                       f"with displacement ({bos_candle_body/atr14:.1f}x ATR)")

    # ── OTE + FVG Unicorn ────────────────────────────────────────────────────
    # "Unicorn" setup: price inside a Fair Value Gap AND inside OTE zone (61.8–79%)
    # simultaneously. Both filters must pass for the score to apply.
    if n >= 25:
        swing_high = float(np.max(highs[-25:]))
        swing_low  = float(np.min(lows[-25:]))
        swing_range = swing_high - swing_low
        price = float(closes[-1])

        if swing_range > 0:
            ote_buy_lo  = swing_low  + swing_range * 0.618
            ote_buy_hi  = swing_low  + swing_range * 0.79
            ote_sell_lo = swing_high - swing_range * 0.79
            ote_sell_hi = swing_high - swing_range * 0.618

            in_ote_buy  = ote_buy_lo  <= price <= ote_buy_hi
            in_ote_sell = ote_sell_lo <= price <= ote_sell_hi

            # Check for FVG simultaneously
            bullish_fvg = bearish_fvg = False
            for i in range(n - 1, max(n - 8, 2), -1):
                if lows[i] > highs[i-2]:
                    bullish_fvg = True; break
                if highs[i] < lows[i-2]:
                    bearish_fvg = True; break

            if in_ote_buy and bullish_fvg:
                buy_score += 3
                details.append(f"Unicorn setup — OTE buy zone ({ote_buy_lo:.5f}–{ote_buy_hi:.5f}) "
                                f"+ bullish FVG present")
            elif in_ote_sell and bearish_fvg:
                sell_score += 3
                details.append(f"Unicorn setup — OTE sell zone ({ote_sell_lo:.5f}–{ote_sell_hi:.5f}) "
                                f"+ bearish FVG present")
            elif in_ote_buy:
                buy_score += 1
                details.append(f"In OTE buy zone — no FVG yet (partial)")
            elif in_ote_sell:
                sell_score += 1
                details.append(f"In OTE sell zone — no FVG yet (partial)")

    if buy_score > sell_score and buy_score >= 3:
        return ConceptVote("Structure", "BUY",  1.5, f"Score {buy_score}", " | ".join(details))
    if sell_score > buy_score and sell_score >= 3:
        return ConceptVote("Structure", "SELL", 1.5, f"Score {sell_score}", " | ".join(details))
    return ConceptVote("Structure", "NEUTRAL", 1.5, "No MSS/Unicorn setup", " | ".join(details))


# ─────────────────────────────────────────────
# Master confluence runner
# ─────────────────────────────────────────────
def run_confluence(df: pd.DataFrame, weight_overrides: dict | None = None, snapshot: dict | None = None) -> ConfluenceResult:
    """Run all 7 concepts and combine into one signal.

    Args:
        df: 15m OHLCV DataFrame
        weight_overrides: optional dict mapping concept name -> adjusted weight
                          (from filters.get_concept_weights())
        snapshot: optional dict of ML features for the XGBoost vote
    """
    votes = [
        smc_analysis(df),
        ict_analysis(df),
        wyckoff_analysis(df),
        volume_profile_analysis(df),
        divergence_analysis(df),
        ict_advanced_analysis(df),   # CISD + Breaker Blocks + Silver Bullet
        structure_analysis(df),      # Clean MSS + OTE+FVG Unicorn + ATR Displacement
    ]

    # Calculate tentative signal for XGBoost
    b_w = sum(v.weight for v in votes if v.vote == "BUY")
    s_w = sum(v.weight for v in votes if v.vote == "SELL")
    tentative_signal = "BUY" if b_w > s_w else "SELL" if s_w > b_w else "HOLD"

    if snapshot is not None and tentative_signal != "HOLD":
        try:
            from ml_xgboost import get_ml_vote
            ml_vote = get_ml_vote(snapshot, votes, tentative_signal)
            votes.append(ml_vote)
        except Exception as e:
            import logging
            logging.error(f"XGBoost vote failed: {e}")

    # Apply dynamic weight overrides from concept win rate analysis
    if weight_overrides:
        for v in votes:
            if v.name in weight_overrides:
                v.weight = weight_overrides[v.name]

    buy_weight  = 0.0
    sell_weight = 0.0
    buy_count   = sell_count = neutral_count = 0
    reasons     = []

    for v in votes:
        if v.vote == "BUY":
            buy_weight  += v.weight
            buy_count   += 1
            reasons.append(f"[{v.name}] BUY — {v.reason}")
        elif v.vote == "SELL":
            sell_weight += v.weight
            sell_count  += 1
            reasons.append(f"[{v.name}] SELL — {v.reason}")
        else:
            neutral_count += 1

    total_weight = sum(v.weight for v in votes)
    buy_pct      = buy_weight  / total_weight * 10
    sell_pct     = sell_weight / total_weight * 10

    # Need at least 3 concepts agreeing for a signal
    if buy_count >= 3 and buy_weight > sell_weight:
        signal = "BUY"
        score  = round(min(buy_pct, 10), 1)
    elif sell_count >= 3 and sell_weight > buy_weight:
        signal = "SELL"
        score  = round(min(sell_pct, 10), 1)
    else:
        signal = "HOLD"
        score  = 0.0

    if score >= 8:   confidence = "VERY HIGH"
    elif score >= 6: confidence = "HIGH"
    elif score >= 4: confidence = "MEDIUM"
    else:            confidence = "LOW"

    return ConfluenceResult(
        signal=signal, score=score, confidence=confidence,
        votes=votes, buy_votes=buy_count, sell_votes=sell_count,
        neutral_votes=neutral_count, reasons=reasons,
    )


if __name__ == "__main__":
    import yfinance as yf
    print("Testing confluence engine on EUR/USD 15m...")
    df = yf.download("EURUSD=X", period="5d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    result = run_confluence(df)
    print(f"\nSignal: {result.signal} | Score: {result.score}/10 | Confidence: {result.confidence}")
    print(f"Votes: {result.buy_votes} BUY / {result.sell_votes} SELL / {result.neutral_votes} NEUTRAL")
    print("\nConcept breakdown:")
    for v in result.votes:
        print(f"  {v.name:15} -> {v.vote:7} | {v.reason}")
        if v.detail:
            print(f"  {'':15}   {v.detail}")