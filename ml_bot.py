import logging
import uuid
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode
from ml_data import scan_all_pairs, scan_strong_signals, analyze_pair, PAIRS
from ml_ai import (explain_signal, explain_result, format_signal_msg,
                   format_weak_msg, format_result_msg, format_stats_msg)
from ml_logger import (log_trade, mark_result, get_stats, get_pending_trades,
                       export_for_ml, delete_trade, delete_all_pending)
from config_loader import load_config
import filters as signal_filters
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

_cfg           = load_config()
TELEGRAM_TOKEN = _cfg["TELEGRAM_TOKEN"]

subscribed_users  = set()
auto_scan_enabled = {}   # uid -> bool
last_signal_sent  = {}   # pair -> signal direction
weak_update_count = {}   # uid -> count since last weak msg

WEAK_UPDATE_EVERY = 5    # send weak update every 5 scans (~5 min)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    subscribed_users.add(uid)
    auto_scan_enabled[uid] = False
    await update.message.reply_text(
        "🤖 *QT ML Signal Bot v2*\n\n"
        "7 concepts: SMC, ICT, ICT\\_Adv, Wyckoff, Volume Profile, Divergence, Structure\n"
        "Smart filters: session, news, ATR, cooldown, pair blacklist\n"
        "Auto\\-signal with dynamic vote threshold \\+ TP/SL/RR 📊\n\n"
        "Commands:\n"
        "/as — toggle auto\\-scan on/off\n"
        "/best — best setup now \\(any strength\\)\n"
        "/scan — quick overview of all pairs\n"
        "/pair EURUSD — analyze specific pair\n"
        "/filters — check which filters are active now\n"
        "/stats — win rate, ML progress, best pairs\n"
        "/pending — trades awaiting result\n"
        "/exportml — download training data CSV\n"
        "/train — train XGBoost Phase 2 on recent data\n"
        "/cancelall — delete all pending trades\n\n"
        "After each trade:\n"
        "/win\\_ID, /loss\\_ID, or /cancel\\_ID",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_as(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    subscribed_users.add(uid)
    current = auto_scan_enabled.get(uid, False)
    auto_scan_enabled[uid] = not current
    if auto_scan_enabled[uid]:
        msg = "🤖 Auto\\-scan: *ON* ✅\nScanning every 1 minute for 4/5 signals\nUse /as again to turn off"
    else:
        msg = "🤖 Auto\\-scan: *OFF* ❌\nNo more auto alerts\nUse /as again to turn on"
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_best(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    subscribed_users.add(update.effective_user.id)
    msg = await update.message.reply_text("⚡ Scanning\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    try:
        all_sigs = scan_all_pairs()
        if not all_sigs:
            await msg.edit_text("❌ Could not fetch data\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        strong = [s for s in all_sigs if s.signal != "HOLD"]

        if strong:
            best   = strong[0]
            sig_id = str(uuid.uuid4())[:6].upper()
            thinking, answer = explain_signal(best)
            text = format_signal_msg(best, answer, sig_id, thinking=thinking)
            log_trade(sig_id, best.pair, best.signal, best.price, best.confluence, best.df_snapshot)
        else:
            best = all_sigs[0]
            text = format_weak_msg(best)

        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        await msg.edit_text(f"❌ Error: {esc(str(e))}", parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    subscribed_users.add(update.effective_user.id)
    msg = await update.message.reply_text("🔍 Scanning\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    try:
        all_sigs = scan_all_pairs()
        strong   = [s for s in all_sigs if s.signal != "HOLD"]

        if not strong:
            best     = all_sigs[0] if all_sigs else None
            agreeing = 0
            if best:
                agreeing = max(best.confluence.buy_votes, best.confluence.sell_votes)
            await msg.edit_text(
                f"🔍 *Scan Complete*\n\nNo 4/5 signals\\. Best: `{best.pair if best else 'N/A'}` with `{agreeing}/5` concepts\\.",
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return

        lines = ["⚡ *Confluence Scan:*\n"]
        for sig in strong[:6]:
            label    = "🟢" if sig.signal == "BUY" else "🔴"
            agreeing = sig.confluence.buy_votes if sig.signal == "BUY" else sig.confluence.sell_votes
            total    = len(sig.vote_breakdown)
            pair_esc = sig.pair.replace("/", "\\/")
            lines.append(f"{label} *{pair_esc}* — `{agreeing}/{total}` agree \\| Score: `{sig.score}`")

        lines.append("\n_/best for full analysis_")
        await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        await msg.edit_text(f"❌ Error: {esc(str(e))}", parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_pair(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    subscribed_users.add(update.effective_user.id)
    if not ctx.args:
        await update.message.reply_text("Example: /pair EURUSD")
        return
    raw  = ctx.args[0].upper().replace("/", "")
    pair = raw[:3] + "/" + raw[3:]
    if pair not in PAIRS:
        await update.message.reply_text(f"❌ Unknown pair: {pair}")
        return
    msg = await update.message.reply_text(f"Analyzing {pair}\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    sig = analyze_pair(pair)
    if not sig:
        await msg.edit_text("❌ Could not fetch data\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return
    sig_id         = str(uuid.uuid4())[:6].upper()
    thinking, answer = explain_signal(sig)
    text = format_signal_msg(sig, answer, sig_id, thinking=thinking)
    log_trade(sig_id, sig.pair, sig.signal, sig.price, sig.confluence, sig.df_snapshot)
    await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats = get_stats()
    if stats["total"] == 0:
        await update.message.reply_text(
            "No trades yet\\. Use /best to get your first signal\\.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    await update.message.reply_text(format_stats_msg(stats), parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_pending(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pending = get_pending_trades()
    if not pending:
        await update.message.reply_text("No pending trades\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return

    def e(s):
        for c in r"_*[]()~`>#+-=|{}.!":
            s = str(s).replace(c, f"\\{c}")
        return s

    lines = ["⏳ *Pending trades:*\n"]
    for t in pending[-10:]:
        label = "🟢" if t["signal"] == "BUY" else "🔴"
        lines.append(f"{label} `{t['id']}` — {e(t['pair'])} @ `{t['price']:.5f}`\n   /win\\_{t['id']} \\| /loss\\_{t['id']}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_exportml(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats = get_stats()
    if stats["closed"] < 10:
        await update.message.reply_text(
            f"Only {stats['closed']} closed trades\\. Need more data for ML training\\.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    csv_data = export_for_ml()
    if not csv_data:
        await update.message.reply_text("No data to export\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return

    import io
    bio = io.BytesIO(csv_data.encode())
    bio.name = "ml_training_data.csv"
    await update.message.reply_document(
        document=bio,
        filename="ml_training_data.csv",
        caption=f"📊 ML training data — {stats['closed']} labeled trades\\. Ready for XGBoost training\\!",
    )


async def cmd_train(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "⏳ Training XGBoost Phase 2 model\\.\\.\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )
    try:
        from ml_xgboost import train_xgboost
        success, info = train_xgboost()

        if success:
            await msg.edit_text(f"✅ *Training Complete*\n{esc(info)}", parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await msg.edit_text(f"❌ *Training Failed*\n{esc(info)}", parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        await msg.edit_text(f"❌ *Error:*\n{esc(str(e))}", parse_mode=ParseMode.MARKDOWN_V2)

async def cmd_cancelall(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    deleted = delete_all_pending()
    if deleted > 0:
        await update.message.reply_text(f"🗑 Deleted `{deleted}` pending trades\\.", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("No pending trades to delete\\.", parse_mode=ParseMode.MARKDOWN_V2)

async def handle_result(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    if text.startswith("/win_"):
        result, signal_id = "WIN", text[5:].upper()
    elif text.startswith("/loss_"):
        result, signal_id = "LOSS", text[6:].upper()
    elif text.startswith("/cancel_"):
        signal_id = text[8:].upper()
        if delete_trade(signal_id):
            await update.message.reply_text(f"🗑 Trade `{signal_id}` deleted\\.", parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await update.message.reply_text(f"❌ Trade `{signal_id}` not found\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return
    else:
        return

    trade = mark_result(signal_id, result)
    if trade:
        thinking, answer = explain_result(trade, result)
        msg_text         = format_result_msg(trade, result, answer, thinking=thinking)
        await update.message.reply_text(msg_text, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text(
            f"❌ Trade `{signal_id}` not found or already closed\\.",
            parse_mode=ParseMode.MARKDOWN_V2
        )


async def auto_scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Runs every 1 minute. Gates signals through all filters before alerting."""
    active_users = [uid for uid, enabled in auto_scan_enabled.items() if enabled]
    if not active_users:
        return

    try:
        # Quick pre-checks: session and news (apply globally, not per-pair)
        blocked, reason = signal_filters.is_bad_session()
        if blocked:
            log.info(f"Auto-scan skipped: {reason}")
            return

        blocked, reason = signal_filters.is_news_blackout()
        if blocked:
            log.info(f"Auto-scan skipped: {reason}")
            return

        all_sigs = scan_all_pairs()
        if not all_sigs:
            return

        # Filter to signals that aren't HOLD
        candidates = [s for s in all_sigs if s.signal != "HOLD"]

        # Apply dynamic vote threshold + per-pair filters
        best = None
        rejection_log = []
        for sig in candidates:
            agreeing = sig.confluence.buy_votes if sig.signal == "BUY" else sig.confluence.sell_votes
            min_votes = signal_filters.get_dynamic_min_votes(sig.score)

            if agreeing < min_votes:
                rejection_log.append(f"{sig.pair}: {agreeing} votes < {min_votes} needed (score {sig.score})")
                continue

            # Same-direction duplicate check (old logic)
            if last_signal_sent.get(sig.pair) == sig.signal:
                continue

            # Run all per-pair filters (cooldown, duplicate, ATR, blacklist)
            passed, reasons = signal_filters.check_all_filters(
                sig.pair, sig.price, sig.score,
                sig.df_raw if sig.df_raw is not None else pd.DataFrame()
            )
            if not passed:
                rejection_log.append(f"{sig.pair}: {', '.join(reasons)}")
                continue

            best = sig
            break  # take the first passing signal (highest score)

        if rejection_log:
            log.info(f"Filtered out: {'; '.join(rejection_log[:3])}")

        if best:
            sig_id           = str(uuid.uuid4())[:6].upper()
            thinking, answer = explain_signal(best)
            text             = format_signal_msg(best, answer, sig_id, thinking=thinking)
            log_trade(sig_id, best.pair, best.signal, best.price, best.confluence, best.df_snapshot)
            last_signal_sent[best.pair] = best.signal
            signal_filters.record_signal(best.pair)  # record cooldown

            for uid in active_users:
                try:
                    await ctx.bot.send_message(chat_id=uid, text=text, parse_mode=ParseMode.MARKDOWN_V2)
                    log.info(f"Signal sent to {uid}: {best.pair} {best.signal} (score {best.score})")
                except Exception as e:
                    log.error(f"Send failed {uid}: {e}")
        else:
            # Weak update every 5 scans — no AI call
            if not all_sigs:
                return

            for uid in active_users:
                count = weak_update_count.get(uid, 0) + 1
                weak_update_count[uid] = count
                if count >= WEAK_UPDATE_EVERY:
                    weak_update_count[uid] = 0
                    try:
                        text = format_weak_msg(all_sigs[0])
                        await ctx.bot.send_message(chat_id=uid, text=text, parse_mode=ParseMode.MARKDOWN_V2)
                        log.info(f"Weak update sent to {uid}")
                    except Exception as e:
                        log.error(f"Weak update failed {uid}: {e}")

    except Exception as e:
        log.error(f"Auto scan error: {e}")


def esc(s):
    """Escape MarkdownV2 special characters."""
    for c in r"_*[]()~`>#+-=|{}.!":
        s = str(s).replace(c, f"\\{c}")
    return s


async def cmd_filters(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show current filter status — what's blocking/allowing signals right now."""
    from datetime import datetime
    import pytz
    CET = pytz.timezone("Europe/Berlin")
    now = datetime.now(CET)

    lines = [f"🛡 *Filter Status* — {esc(now.strftime('%H:%M CET, %A'))}\n"]

    # Session
    blocked, reason = signal_filters.is_bad_session()
    lines.append(f"{'🔴' if blocked else '🟢'} Session: {'`BLOCKED` — ' + esc(reason) if blocked else '`ACTIVE` ✅'}")

    # News
    blocked, reason = signal_filters.is_news_blackout()
    lines.append(f"{'🔴' if blocked else '🟢'} News: {'`BLACKOUT` — ' + esc(reason) if blocked else '`CLEAR` ✅'}")

    # Dynamic threshold
    for score_ex in [8.0, 6.0, 4.0]:
        mv = signal_filters.get_dynamic_min_votes(score_ex)
        label = f"{mv}/7" if mv < 99 else "SKIP"
        lines.append(f"   Score `{esc(str(score_ex))}`: need `{label}` votes")

    # Concept weights
    weights = signal_filters.get_concept_weights()
    adjusted = {k: v for k, v in weights.items()
                if v != signal_filters._BASE_WEIGHTS.get(k, 1.0)}
    if adjusted:
        lines.append(f"\n📊 *Adjusted Concept Weights:*")
        for name, w in adjusted.items():
            base = signal_filters._BASE_WEIGHTS[name]
            direction = "⬆️" if w > base else "⬇️"
            lines.append(f"  {direction} {esc(name)}: `{base}` → `{w}`")
    else:
        lines.append(f"\n📊 Concept weights: all at defaults \\(need 10\\+ trades per concept\\)")

    # Blacklisted pairs
    blacklisted = []
    for pair in PAIRS:
        bl, reason = signal_filters.is_pair_blacklisted(pair)
        if bl:
            blacklisted.append(pair)
    if blacklisted:
        lines.append(f"\n🚫 *Blacklisted Pairs:*")
        for p in blacklisted:
            _, reason = signal_filters.is_pair_blacklisted(p)
            lines.append(f"  {esc(reason)}")
    else:
        lines.append(f"\n✅ No blacklisted pairs")

    # Active cooldowns
    cooldowns = []
    for pair in PAIRS:
        cd, reason = signal_filters.is_on_cooldown(pair)
        if cd:
            cooldowns.append(reason)
    if cooldowns:
        lines.append(f"\n⏳ *Active Cooldowns:*")
        for c in cooldowns:
            lines.append(f"  {esc(c)}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("as",       cmd_as))
    app.add_handler(CommandHandler("best",     cmd_best))
    app.add_handler(CommandHandler("scan",     cmd_scan))
    app.add_handler(CommandHandler("pair",     cmd_pair))
    app.add_handler(CommandHandler("filters",  cmd_filters))
    app.add_handler(CommandHandler("stats",    cmd_stats))
    app.add_handler(CommandHandler("pending",  cmd_pending))
    app.add_handler(CommandHandler("cancelall", cmd_cancelall))
    app.add_handler(CommandHandler("exportml", cmd_exportml))
    app.add_handler(CommandHandler("train",    cmd_train))
    app.add_handler(MessageHandler(filters.Regex(r"^/(win|loss|cancel)_"), handle_result))

    app.job_queue.run_repeating(auto_scan_job, interval=60, first=10)

    log.info("QT ML Bot v2 started — smart filters active, scanning every 60s.")
    app.run_polling()


if __name__ == "__main__":
    main()