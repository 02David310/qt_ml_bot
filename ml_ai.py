from groq import Groq
import yfinance as yf
import numpy as np
import re as _re
from ml_data import QTSignalML
from ml_logger import get_stats, load_trades
from config_loader import load_config
from filters import calculate_tp_sl

_cfg   = load_config()
client = Groq(api_key=_cfg.get("GROQ_API_KEY", ""))

# ── Tuneable AI settings (all overridable in Config.txt) ─────────────────────
# GROQ_MODEL:            main model for all AI verdicts / post-trade reviews
# GROQ_FALLBACK_MODEL:   used automatically if the main model fails with 404 or
#                        429 (rate limit). Limits are per model, so a different
#                        model gives you a separate quota.
# AI_MAX_TOKENS:         hard cap on tokens per reply. gpt-oss models "think"
#                        first and thinking counts toward this cap, so keep it
#                        well above 350 or replies can come back empty.
# AI_TEMPERATURE:        0 = deterministic/dry, 1 = creative/varied
# AI_REASONING_EFFORT:   low / medium / high — how long gpt-oss thinks before
#                        answering. "low" = fastest and cheapest, fine for
#                        short verdicts. Only applied to gpt-oss models.
GROQ_MODEL          = _cfg.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_FALLBACK_MODEL = _cfg.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
AI_MAX_TOKENS       = int(_cfg.get("AI_MAX_TOKENS", 1000))
AI_TEMPERATURE      = float(_cfg.get("AI_TEMPERATURE", 0.7))
AI_REASONING_EFFORT = _cfg.get("AI_REASONING_EFFORT", "low")


def esc(s):
    for c in r"_*[]()~`>#+-=|{}.!":
        s = str(s).replace(c, f"\\{c}")
    return s


# Yahoo Finance ticker map (same as ml_data.py)
_YF_MAP = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "EUR/CHF": "EURCHF=X", "GBP/JPY": "GBPJPY=X", "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X", "EUR/GBP": "EURGBP=X", "EUR/AUD": "EURAUD=X",
    "CHF/JPY": "CHFJPY=X", "CAD/JPY": "CADJPY=X", "AUD/JPY": "AUDJPY=X",
    "AUD/CHF": "AUDCHF=X", "AUD/CAD": "AUDCAD=X", "EUR/CAD": "EURCAD=X",
    "GBP/CAD": "GBPCAD=X",
}


def get_mtf_context(pair: str) -> str:
    """Fetch 1H and Daily data from Yahoo Finance to give the AI higher-timeframe context.

    Returns a compact 2–3 line summary injected into every AI prompt so the model
    knows whether the 5M signal is with or against the higher-timeframe trend.
    """
    ticker = _YF_MAP.get(pair)
    if not ticker:
        return "MTF data unavailable for this pair."

    try:
        lines = []

        # ── 1H trend (last 48 candles ≈ 2 days) ─────────────────────────────
        df1h = yf.download(ticker, period="5d", interval="1h", progress=False)
        if isinstance(df1h.columns, __import__('pandas').MultiIndex):
            df1h.columns = df1h.columns.get_level_values(0)

        if not df1h.empty and len(df1h) >= 12:
            closes_1h = df1h["Close"].values
            highs_1h  = df1h["High"].values
            lows_1h   = df1h["Low"].values

            trend_1h  = float(closes_1h[-1]) - float(closes_1h[-12])
            trend_pct  = trend_1h / float(closes_1h[-12]) * 100
            direction = "bullish" if trend_pct > 0.05 else "bearish" if trend_pct < -0.05 else "sideways"

            # ATR on 1H
            tr_1h = [max(highs_1h[i] - lows_1h[i],
                         abs(highs_1h[i] - closes_1h[i-1]),
                         abs(lows_1h[i]  - closes_1h[i-1]))
                     for i in range(-10, 0)]
            atr_1h = float(np.mean(tr_1h))

            # Support/resistance: 1H high/low of last 24 candles
            h24 = float(np.max(highs_1h[-24:]))
            l24 = float(np.min(lows_1h[-24:]))

            lines.append(
                f"1H trend: {direction} ({trend_pct:+.2f}% over 12h) | "
                f"ATR(10): {atr_1h:.5f} | 24h range: {l24:.5f}–{h24:.5f}"
            )

        # ── Daily trend (last 10 days) ────────────────────────────────────────
        df1d = yf.download(ticker, period="15d", interval="1d", progress=False)
        if isinstance(df1d.columns, __import__('pandas').MultiIndex):
            df1d.columns = df1d.columns.get_level_values(0)

        if not df1d.empty and len(df1d) >= 5:
            closes_d = df1d["Close"].values
            highs_d  = df1d["High"].values
            lows_d   = df1d["Low"].values

            d5_pct   = (float(closes_d[-1]) - float(closes_d[-5])) / float(closes_d[-5]) * 100
            d_dir    = "uptrend" if d5_pct > 0.1 else "downtrend" if d5_pct < -0.1 else "range"
            wk_high  = float(np.max(highs_d[-5:]))
            wk_low   = float(np.min(lows_d[-5:]))

            lines.append(
                f"Daily: {d_dir} ({d5_pct:+.2f}% over 5 days) | "
                f"Weekly range: {wk_low:.5f}–{wk_high:.5f}"
            )

        # ── Alignment warning ─────────────────────────────────────────────────
        if len(lines) == 2:
            h_bull = "bullish" in lines[0]
            h_bear = "bearish" in lines[0]
            d_bull = "uptrend" in lines[1]
            d_bear = "downtrend" in lines[1]
            if (h_bull and d_bear) or (h_bear and d_bull):
                lines.append("⚠ 1H and Daily trends are CONFLICTING — caution on this signal")
            elif (h_bull and d_bull):
                lines.append("✓ 1H and Daily both bullish — trend alignment confirmed")
            elif (h_bear and d_bear):
                lines.append("✓ 1H and Daily both bearish — trend alignment confirmed")

        return "\n".join(lines) if lines else "MTF data fetch returned empty."

    except Exception as e:
        return f"MTF context unavailable: {e}"


def _groq_call(model: str, prompt: str) -> str:
    """Single chat-completion call to Groq. Raises on any API error."""
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=AI_MAX_TOKENS,
        temperature=AI_TEMPERATURE,
    )
    # gpt-oss models support reasoning_effort; other models would reject it
    if "gpt-oss" in model and AI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = AI_REASONING_EFFORT

    response = client.chat.completions.create(**kwargs)
    text = (response.choices[0].message.content or "").strip()
    # Strip <think>...</think> blocks in case a model (e.g. Qwen) returns them inline
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    return text


def _ask_groq(prompt: str) -> tuple[str | None, str]:
    """Send prompt to Groq. Returns (None, answer).

    Tries GROQ_MODEL first; on 404, 429 or an empty reply it retries once with
    GROQ_FALLBACK_MODEL before giving up.
    """
    models = [GROQ_MODEL]
    if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
        models.append(GROQ_FALLBACK_MODEL)

    last_err = ""
    for i, model in enumerate(models):
        try:
            answer = _groq_call(model, prompt)
            if answer:
                return None, answer
            last_err = "empty response (try raising AI_MAX_TOKENS)"
            if i < len(models) - 1:
                continue
            break
        except Exception as e:
            last_err = str(e)
            retryable = ("404" in last_err or "429" in last_err
                         or "rate" in last_err.lower() or "not found" in last_err.lower())
            if retryable and i < len(models) - 1:
                continue
            break

    if "429" in last_err or "rate" in last_err.lower() or "quota" in last_err.lower():
        return None, "AI rate limit hit — signal data above is still valid"
    return None, f"AI unavailable: {last_err}"


def get_pattern_context(pair: str, signal: str) -> str:
    trades = [t for t in load_trades() if t["result"] is not None]
    if not trades:
        return "No past trade data yet."

    similar = [t for t in trades if t["pair"] == pair or t["signal"] == signal]
    if not similar:
        similar = trades[-20:]

    wins   = [t for t in similar if t["result"] == "WIN"]
    losses = [t for t in similar if t["result"] == "LOSS"]

    concept_win_rates = {}
    for concept in ["smc_vote", "ict_vote", "wyckoff_vote", "volprofile_vote", "divergence_vote"]:
        agreed_wins   = sum(1 for t in wins   if t.get(concept, 0) == 1)
        agreed_losses = sum(1 for t in losses if t.get(concept, 0) == 1)
        total_agreed  = agreed_wins + agreed_losses
        if total_agreed > 0:
            concept_win_rates[concept.replace("_vote", "")] = round(agreed_wins / total_agreed * 100)

    hour_results = {}
    for t in similar:
        h = t.get("hour_cet", 0)
        if h not in hour_results:
            hour_results[h] = {"wins": 0, "losses": 0}
        if t["result"] == "WIN":
            hour_results[h]["wins"] += 1
        else:
            hour_results[h]["losses"] += 1

    best_hours = sorted(
        [(h, v["wins"] / (v["wins"] + v["losses"]) * 100)
         for h, v in hour_results.items() if v["wins"] + v["losses"] >= 2],
        key=lambda x: x[1], reverse=True
    )[:3]

    context = f"Past {len(similar)} similar trades: {len(wins)}W/{len(losses)}L\n"
    if concept_win_rates:
        context += "Concept win rates on similar trades: " + ", ".join(
            f"{k}: {v}%" for k, v in sorted(concept_win_rates.items(), key=lambda x: x[1], reverse=True)
        ) + "\n"
    if best_hours:
        context += "Best hours (CET): " + ", ".join(f"{h}:00 ({wr:.0f}%)" for h, wr in best_hours)

    return context


def explain_signal(sig: QTSignalML) -> tuple[str | None, str]:
    """Returns (None, answer) — short 2-3 sentence signal verdict."""
    agreeing   = sig.confluence.buy_votes if sig.signal == "BUY" else sig.confluence.sell_votes
    total      = len(sig.vote_breakdown)
    agreeing_names = ", ".join(
        v.name for v in sig.vote_breakdown if v.vote == sig.signal
    )
    against_names = ", ".join(
        v.name for v in sig.vote_breakdown if v.vote != sig.signal and v.vote != "NEUTRAL"
    )
    pattern_ctx = get_pattern_context(sig.pair, sig.signal)
    mtf_ctx     = get_mtf_context(sig.pair)

    prompt = (
        f"You are a professional quantitative FX trader. We have a {sig.signal} signal on {sig.pair}. "
        f"Provide a concise, highly analytical technical breakdown (3-4 sentences). "
        f"Do NOT just repeat the data. Instead, synthesize why this setup works based on the agreeing concepts, "
        f"and point out the primary risk factor or invalidation level based on the disagreeing concepts or RSI.\n"
        f"Data: {agreeing}/{total} agree ({agreeing_names}). Against: {against_names or 'none'}. "
        f"Score {sig.score}/10. RSI {sig.df_snapshot.get('rsi','?')}. "
        f"MTF Context: {mtf_ctx.split(chr(10))[0][:120]}. History Context: {pattern_ctx.split(chr(10))[0][:120]}."
    )
    return _ask_groq(prompt)


def explain_result(trade: dict, result: str) -> tuple[str | None, str]:
    """Returns (None, answer) — short 2-sentence post-trade review."""
    aligned = ", ".join(
        c for c, v in [
            ("SMC",  trade.get("smc_vote",0)), ("ICT", trade.get("ict_vote",0)),
            ("Wyckoff", trade.get("wyckoff_vote",0)), ("VP", trade.get("volprofile_vote",0)),
            ("Div", trade.get("divergence_vote",0)), ("ICT_Adv", trade.get("ict_adv_vote",0)),
            ("Structure", trade.get("structure_vote",0)),
        ] if v == 1
    ) or "none"
    opposed = ", ".join(
        c for c, v in [
            ("SMC",  trade.get("smc_vote",0)), ("ICT", trade.get("ict_vote",0)),
            ("Wyckoff", trade.get("wyckoff_vote",0)), ("VP", trade.get("volprofile_vote",0)),
            ("Div", trade.get("divergence_vote",0)), ("ICT_Adv", trade.get("ict_adv_vote",0)),
            ("Structure", trade.get("structure_vote",0)),
        ] if v == -1
    ) or "none"

    prompt = (
        f"You are a professional quantitative FX trader doing a post-trade review. "
        f"Provide a highly analytical 3-4 sentence debrief on this {result}.\n"
        f"Focus on: {'What structural/indicator confluence confirmed this WIN?' if result == 'WIN' else 'What structural failure, missed context, or indicator divergence caused this LOSS?'}\n"
        f"Also state what specific filter or condition should be added next time.\n"
        f"Trade: {trade['pair']} {trade['signal']} -> {result}. "
        f"Aligned: {aligned}. Against: {opposed}. "
        f"RSI {trade.get('rsi','?')}, hour: {trade.get('hour_cet','?')}:00 CET."
    )
    return _ask_groq(prompt)


def _decode_vote(encoded: int, signal: str) -> str:
    if encoded == 1:  return signal
    if encoded == -1: return "BUY" if signal == "SELL" else "SELL"
    return "NEUTRAL"


def format_signal_msg(
    sig: QTSignalML,
    answer: str,
    signal_id: str,
    thinking: str | None = None,
) -> str:
    direction  = sig.signal
    label      = {"BUY": "🟢 BUY", "SELL": "🔴 SELL"}.get(direction, "➡️ HOLD")
    conf_emoji = {"VERY HIGH": "🔥", "HIGH": "⭐⭐⭐", "MEDIUM": "⭐⭐", "LOW": "⭐"}.get(sig.confidence, "⭐")
    total      = len(sig.vote_breakdown)
    agreeing   = sig.confluence.buy_votes if direction == "BUY" else sig.confluence.sell_votes

    vote_lines = []
    for v in sig.vote_breakdown:
        icon = "✅" if v.vote == direction else "❌" if v.vote not in [direction, "NEUTRAL"] else "➡️"
        weight_str = f" \\(w:{esc(f'{v.weight:.1f}')}\\)" if v.weight != 1.0 and v.weight != 1.5 else ""
        vote_lines.append(f"{icon} {esc(v.name)}: `{v.vote}`{weight_str}")
    votes_block = "\n".join(vote_lines)

    snap    = sig.df_snapshot
    pattern = snap.get("candle_pattern", "none").replace("_", " ").title()

    # TP/SL/RR calculation
    tp_sl_block = ""
    if sig.df_raw is not None and direction in ("BUY", "SELL"):
        try:
            levels = calculate_tp_sl(sig.df_raw, direction)
            rr_emoji = "🟢" if levels["rr"] >= 2.0 else "🟡" if levels["rr"] >= 1.5 else "🔴"
            
            sl_pips_str = esc(f"{levels['sl_pips']:.1f}")
            tp_pips_str = esc(f"{levels['tp_pips']:.1f}")
            
            tp_sl_block = (
                f"\n🎯 *Risk Management:*\n"
                f"• SL: `{levels['sl']:.5f}` \\({sl_pips_str} pips\\)\n"
                f"• TP: `{levels['tp']:.5f}` \\({tp_pips_str} pips\\)\n"
                f"• RR: `{levels['rr']:.1f}:1` {rr_emoji} \\| ATR: `{levels['atr']:.5f}`\n"
            )
        except Exception:
            tp_sl_block = ""

    # AI verdict — length is steered by the prompt (3-4 sentences); AI_MAX_TOKENS is only a ceiling
    ai_block = f"🤖 *AI Verdict:*\n{esc(answer)}"

    msg = (
        f"⚡ *QT Signal — {esc(sig.pair)}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🗳 *{agreeing}/{total} concepts agree*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Direction: *{label}* {conf_emoji}\n"
        f"Confidence: `{sig.confidence}` \\| Score: `{sig.score}/10`\n"
        f"Price: `{sig.price:.5f}` \\| Expiry: `5 min`\n\n"
        f"📊 *Indicators:*\n"
        f"• RSI: `{snap.get('rsi', 'N/A')}` \\| BB pos: `{snap.get('bb_position', 0.5):.2f}`\n"
        f"• Pattern: `{esc(pattern)}`\n"
        f"{tp_sl_block}\n"
        f"🗳 *Concept Votes:*\n"
        f"{votes_block}\n\n"
        f"{ai_block}\n\n"
        f"📝 *Log result:*\n"
        f"/win\\_{signal_id} — WIN\n"
        f"/loss\\_{signal_id} — LOSS"
    )
    return msg


def format_weak_msg(sig: QTSignalML) -> str:
    total     = len(sig.vote_breakdown)
    buy_v     = sig.confluence.buy_votes
    sell_v    = sig.confluence.sell_votes
    direction = "BUY" if buy_v > sell_v else "SELL" if sell_v > buy_v else "MIXED"
    agreeing  = max(buy_v, sell_v)
    dir_emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "➡️"

    vote_lines = []
    for v in sig.vote_breakdown:
        icon = "✅" if v.vote == direction else "❌" if v.vote not in [direction, "NEUTRAL"] else "➡️"
        vote_lines.append(f"{icon} {esc(v.name)}: `{v.vote}`")
    votes_block = "\n".join(vote_lines)

    stats        = get_stats()
    total_trades = stats["total"]
    ml_progress  = min(int(total_trades / 5), 100)
    progress_bar = "█" * (ml_progress // 10) + "░" * (10 - ml_progress // 10)

    msg = (
        f"⏱ *Auto\\-Scan Update*\n"
        f"🤖 Bot running — {total} concepts checked\n\n"
        f"⚠️ *Signal too weak — skip this one*\n"
        f"Top pair: *{esc(sig.pair)}* {dir_emoji} {direction}\n"
        f"Agreeing: *{agreeing}/{total}* \\(need 4/5\\)\n\n"
        f"🗳 *Breakdown:*\n{votes_block}\n\n"
        f"📈 *ML Training Progress:*\n"
        f"`{progress_bar}` {total_trades}/500 trades\n\n"
        f"_Next scan in 1 min \\| /best to check manually_"
    )
    return msg


def format_result_msg(
    trade: dict,
    result: str,
    answer: str,
    thinking: str | None = None,
) -> str:
    emoji    = "✅" if result == "WIN" else "❌"
    stats    = get_stats()
    win_rate = stats["win_rate"]
    wr_emoji = "🔥" if win_rate >= 70 else "✅" if win_rate >= 55 else "⚠️"

    total_trades = stats["total"]
    progress_bar = "█" * min(int(total_trades / 50), 10) + "░" * max(10 - int(total_trades / 50), 0)

    # AI verdict — length is steered by the prompt (3-4 sentences); AI_MAX_TOKENS is only a ceiling
    ai_block = f"🧠 *AI Post\\-Trade Analysis:*\n{esc(answer)}"

    msg = (
        f"{emoji} *Trade {esc(trade['id'])} — {result}*\n"
        f"Pair: {esc(trade['pair'])} \\| Signal: {trade['signal']}\n"
        f"Win rate: `{win_rate:.1f}%` {wr_emoji} \\({stats['wins']}W/{stats['losses']}L\\)\n\n"
        f"{ai_block}\n\n"
        f"📈 *ML Progress:* `{progress_bar}` {total_trades}/500"
    )
    return msg


def format_stats_msg(stats: dict) -> str:
    win_rate = stats["win_rate"]
    emoji    = "🔥" if win_rate >= 70 else "✅" if win_rate >= 55 else "⚠️" if win_rate >= 40 else "❌"
    total    = stats["total"]
    progress = "█" * min(int(total / 50), 10) + "░" * max(10 - int(total / 50), 0)

    lines = [
        "📊 *Trade Statistics*\n",
        f"• Total: `{total}` \\| Closed: `{stats['closed']}` \\| Pending: `{stats['pending']}`",
        f"• Wins: `{stats['wins']}` \\| Losses: `{stats['losses']}`",
        f"• Win rate: `{win_rate:.1f}%` {emoji}\n",
        f"📈 *ML Training Progress:*",
        f"`{progress}` {total}/500 trades\n",
    ]

    if stats.get("conf_stats"):
        lines.append("*By confidence level:*")
        for conf, cs in sorted(stats["conf_stats"].items()):
            t  = cs["wins"] + cs["losses"]
            wr = cs["wins"] / t * 100 if t else 0
            lines.append(f"• {esc(conf)}: `{cs['wins']}W/{cs['losses']}L` \\({wr:.0f}%\\)")
        lines.append("")

    if stats.get("pair_stats"):
        lines.append("*By pair:*")
        sorted_pairs = sorted(stats["pair_stats"].items(),
                              key=lambda x: x[1]["wins"] / max(x[1]["wins"] + x[1]["losses"], 1),
                              reverse=True)
        for pair, ps in sorted_pairs[:5]:
            t  = ps["wins"] + ps["losses"]
            wr = ps["wins"] / t * 100 if t else 0
            lines.append(f"• {esc(pair)}: `{ps['wins']}W/{ps['losses']}L` \\({wr:.0f}%\\)")

    if stats.get("concept_win_rates"):
        lines.append("\n*By concept \\(agreed trades\\):*")
        sorted_concepts = sorted(stats["concept_win_rates"].items(),
                                 key=lambda x: x[1]["rate"], reverse=True)
        for concept, cws in sorted_concepts:
            lines.append(f"• {esc(concept)}: `{cws['wins']}W/{cws['total'] - cws['wins']}L` \\({cws['rate']:.0f}%\\)")

    if stats.get("best_hour") is not None:
        lines.append(f"\n⏰ Best hour: `{stats['best_hour']}:00 CET`")

    if stats.get("last_10"):
        row = "".join("✅" if t["result"] == "WIN" else "❌" for t in stats["last_10"])
        lines.append(f"\n*Last 10:* {row}")

    if stats.get("ml_ready"):
        lines.append("\n🎉 *100\\+ trades logged — ready for Phase 2 ML training\\!*\nUse /exportml to get your dataset\\.")

    return "\n".join(lines)