# -*- coding: utf-8 -*-
"""Stock AI 통합 웹 대시보드.

SQLite의 ML/LLM 분석 결과와 yfinance 실시간 시세를 하나의 Flask 서버에서
제공한다. 실행: ``python app.py`` / 접속: http://localhost:5000
"""

import json
import os
import threading
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from flask import Flask, Response, jsonify, request, send_from_directory

from database import DB_PATH, connect_db, table_exists


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "5000"))
POLL_INTERVAL = float(os.environ.get("PRICE_POLL_INTERVAL", "15"))

WATCHLIST = {
    "삼성전자": {"symbol": "005930.KS", "currency": "KRW", "exchange": "KRX"},
    "엔비디아": {"symbol": "NVDA", "currency": "USD", "exchange": "NASDAQ"},
    "인텔": {"symbol": "INTC", "currency": "USD", "exchange": "NASDAQ"},
    "애플": {"symbol": "AAPL", "currency": "USD", "exchange": "NASDAQ"},
    "테슬라": {"symbol": "TSLA", "currency": "USD", "exchange": "NASDAQ"},
    "마이크로소프트": {"symbol": "MSFT", "currency": "USD", "exchange": "NASDAQ"},
}
SYMBOL_TO_NAME = {meta["symbol"]: name for name, meta in WATCHLIST.items()}

# Flask의 자동 정적 라우트를 끄고 공개할 파일만 아래에서 명시합니다.
app = Flask(__name__, static_folder=None)

_price_cache = {}
_candle_cache = {}
_cache_lock = threading.Lock()
_poll_thread = None


# ---------------------------------------------------------------------------
# 공통 변환 및 실시간 데이터
# ---------------------------------------------------------------------------
def _to_float(value):
    if value is None:
        return None
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


def _quote_for(name, meta):
    """1분봉을 우선 사용해 정규장·장전·장후 최신 가격을 조회한다."""
    symbol = meta["symbol"]
    ticker = yf.Ticker(symbol)
    fast_info = ticker.fast_info
    previous = _to_float(getattr(fast_info, "previous_close", None))
    average_volume = _to_float(
        getattr(fast_info, "three_month_average_volume", None)
    )
    price = None
    volume = None
    price_timestamp = datetime.now()

    try:
        # fast_info.last_price는 장전·장후에 종가로 고정될 수 있어 1분봉을 우선합니다.
        intraday = ticker.history(
            period="1d",
            interval="1m",
            prepost=True,
            auto_adjust=True,
        )
        if not intraday.empty:
            price = _to_float(intraday["Close"].dropna().iloc[-1])
            volume = _to_float(intraday["Volume"].fillna(0).sum())
            price_timestamp = intraday.index[-1]
    except Exception as exc:
        print(f"[실시간 시세] {symbol} 1분봉 조회 실패: {exc}")

    if price is None:
        price = _to_float(getattr(fast_info, "last_price", None))
        volume = average_volume

    if price is None:
        history = ticker.history(period="2d", interval="1d", auto_adjust=True)
        if not history.empty:
            price = _to_float(history["Close"].iloc[-1])
            previous = _to_float(history["Close"].iloc[-2]) if len(history) > 1 else price

    if price is None:
        return None

    previous = previous or price
    change = price - previous
    percent = (change / previous * 100) if previous else 0.0
    return {
        "name": name,
        "symbol": symbol,
        "price": round(price, 4),
        "change": round(change, 4),
        "pct": round(percent, 4),
        "volume": int(volume) if volume else 0,
        "ts": price_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "currency": meta["currency"],
        "exchange": meta["exchange"],
    }


def _poll_prices():
    while True:
        for name, meta in WATCHLIST.items():
            try:
                quote = _quote_for(name, meta)
                if quote:
                    with _cache_lock:
                        _price_cache[meta["symbol"]] = quote
            except Exception as exc:
                print(f"[실시간 시세] {meta['symbol']} 조회 실패: {exc}")
        time.sleep(POLL_INTERVAL)


def start_price_poller():
    """개발 재로더에서도 폴링 스레드가 중복 생성되지 않도록 보호한다."""
    global _poll_thread
    if _poll_thread and _poll_thread.is_alive():
        return
    _poll_thread = threading.Thread(target=_poll_prices, name="stock-price-poller", daemon=True)
    _poll_thread.start()


@app.before_request
def ensure_price_poller():
    """python app.py와 flask run 모두에서 시세 수집기를 한 번 시작합니다."""
    start_price_poller()


def _calculate_indicators(frame):
    frame = frame.copy()
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    frame["MA5"] = close.rolling(5).mean()
    frame["MA20"] = close.rolling(20).mean()
    frame["MA60"] = close.rolling(60).mean()
    rolling_std = close.rolling(20).std()
    frame["BB_UP"] = frame["MA20"] + (2 * rolling_std)
    frame["BB_LOW"] = frame["MA20"] - (2 * rolling_std)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    relative_strength = gain / (loss + 1e-9)
    frame["RSI"] = 100 - (100 / (1 + relative_strength))
    return frame


def _frame_to_candles(frame):
    candles = []
    for timestamp, row in frame.iterrows():
        if isinstance(timestamp, pd.Timestamp):
            has_time = bool(timestamp.hour or timestamp.minute)
            label = timestamp.strftime("%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d")
        else:
            label = str(timestamp)

        candles.append(
            {
                "t": label,
                "o": _to_float(row.get("Open")),
                "h": _to_float(row.get("High")),
                "l": _to_float(row.get("Low")),
                "c": _to_float(row.get("Close")),
                "v": _to_float(row.get("Volume")),
                "ma5": _to_float(row.get("MA5")),
                "ma20": _to_float(row.get("MA20")),
                "ma60": _to_float(row.get("MA60")),
                "bb_up": _to_float(row.get("BB_UP")),
                "bb_low": _to_float(row.get("BB_LOW")),
                "rsi": _to_float(row.get("RSI")),
            }
        )
    return candles


INTERVAL_MAP = {
    "1m": ("1d", "1m"),
    "5m": ("5d", "5m"),
    "15m": ("5d", "15m"),
    "1h": ("1mo", "1h"),
    "1d": ("1y", "1d"),
    "1wk": ("5y", "1wk"),
}


# ---------------------------------------------------------------------------
# SQLite 조회
# ---------------------------------------------------------------------------
def get_connection():
    return connect_db(row_factory=True)


def _latest_analysis_map(connection):
    if not table_exists(connection, "analysis_log"):
        return {}
    rows = connection.execute(
        """
        SELECT a.*
        FROM analysis_log a
        INNER JOIN (
            SELECT name, MAX(id) AS latest_id
            FROM analysis_log
            GROUP BY name
        ) latest ON latest.latest_id = a.id
        """
    ).fetchall()
    return {row["name"]: dict(row) for row in rows}


def _latest_persona_map(connection):
    if not table_exists(connection, "persona_discussion_log"):
        return {}
    rows = connection.execute(
        """
        SELECT p.ticker_name, p.score, p.opinion, p.timestamp
        FROM persona_discussion_log p
        INNER JOIN (
            SELECT ticker_name, MAX(id) AS latest_id
            FROM persona_discussion_log
            WHERE persona='최종결정자'
            GROUP BY ticker_name
        ) latest ON latest.latest_id = p.id
        """
    ).fetchall()
    return {
        row["ticker_name"]: {
            "score": row["score"],
            "opinion": row["opinion"] or "",
            "timestamp": row["timestamp"],
        }
        for row in rows
    }


def _trend(value, mode="rsi"):
    if value is None:
        return "⚪", "분석 없음"
    value = float(value or 0)
    if mode == "rsi":
        if value > 70:
            return "🔴", "과매수"
        if value < 30:
            return "🟢", "과매도(반등)"
    else:
        if value > 0.1:
            return "🟢", "매수 우세"
        if value < -0.1:
            return "🔴", "매도 우세"
    return "⚖️", "중립"


def _stock_payload(name, analysis=None, persona=None):
    analysis = analysis or {}
    persona = persona or {}
    watch = WATCHLIST.get(name, {})
    symbol = analysis.get("symbol") or watch.get("symbol", "")

    with _cache_lock:
        quote = dict(_price_cache.get(symbol, {}))

    rsi = _to_float(analysis.get("rsi"))
    if rsi is not None:
        trend_emoji, trend_label = _trend(rsi, "rsi")
    elif persona and persona.get("score") is None:
        trend_emoji, trend_label = "❌", "AI 분석 실패"
    else:
        trend_emoji, trend_label = _trend(persona.get("score"), "score")
    source_parts = []
    if analysis:
        source_parts.append("ml")
    if persona:
        source_parts.append("llm")
    if watch:
        source_parts.append("live")

    return {
        "name": name,
        "symbol": symbol,
        "currency": quote.get("currency") or watch.get("currency") or (
            "KRW" if symbol.endswith(".KS") else "USD"
        ),
        "exchange": quote.get("exchange") or watch.get("exchange", ""),
        "price": quote.get("price"),
        "change": quote.get("change"),
        "pct": quote.get("pct"),
        "volume": quote.get("volume"),
        "live_timestamp": quote.get("ts", ""),
        "timestamp": analysis.get("timestamp") or persona.get("timestamp", ""),
        "last_close": analysis.get("last_close"),
        "rsi": rsi,
        "trend_emoji": trend_emoji,
        "trend_label": trend_label,
        "support": analysis.get("support"),
        "resistance": analysis.get("resistance"),
        "usd_krw": analysis.get("usd_krw"),
        "sox": analysis.get("sox"),
        "sentiment": analysis.get("sentiment"),
        "pred_low": analysis.get("pred_low"),
        "pred_high": analysis.get("pred_high"),
        "risk_score": analysis.get("risk_score"),
        "final_score": persona.get("score"),
        "final_opinion": persona.get("opinion", "")[:200],
        "source": "+".join(source_parts) or "none",
    }


# ---------------------------------------------------------------------------
# 정적 페이지 및 통합 API
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/app.js")
def frontend_script():
    return send_from_directory(BASE_DIR, "app.js")


@app.route("/style.css")
def frontend_style():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/api/status")
def api_status():
    connection = get_connection()
    try:
        has_analysis = table_exists(connection, "analysis_log")
        has_persona = table_exists(connection, "persona_discussion_log")
        analysis_count = (
            connection.execute("SELECT COUNT(*) FROM analysis_log").fetchone()[0]
            if has_analysis
            else 0
        )
        persona_count = (
            connection.execute("SELECT COUNT(*) FROM persona_discussion_log").fetchone()[0]
            if has_persona
            else 0
        )
        with _cache_lock:
            live_count = len(_price_cache)
        return jsonify(
            {
                "has_analysis": has_analysis,
                "has_persona": has_persona,
                "analysis_count": analysis_count,
                "persona_count": persona_count,
                "live_count": live_count,
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    finally:
        connection.close()


@app.route("/api/stocks")
def api_stocks():
    connection = get_connection()
    try:
        analyses = _latest_analysis_map(connection)
        personas = _latest_persona_map(connection)
        names = list(WATCHLIST)
        names.extend(sorted((set(analyses) | set(personas)) - set(names)))
        stocks = [
            _stock_payload(name, analyses.get(name), personas.get(name))
            for name in names
        ]
        return jsonify({"stocks": stocks})
    finally:
        connection.close()


@app.route("/api/stock/<string:name>")
def api_stock_detail(name):
    connection = get_connection()
    try:
        analyses = _latest_analysis_map(connection)
        personas = _latest_persona_map(connection)
        analysis = analyses.get(name)
        persona = personas.get(name)
        if name not in WATCHLIST and not analysis and not persona:
            return jsonify({"error": f"종목 '{name}' 데이터가 없습니다."}), 404

        history = []
        if analysis:
            history = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM analysis_log WHERE name=? ORDER BY id DESC LIMIT 30",
                    (name,),
                ).fetchall()
            ]

        stock = _stock_payload(name, analysis, persona)
        stock.update(
            {
                "latest": analysis or {
                    "name": name,
                    "symbol": stock["symbol"],
                    "last_close": None,
                    "rsi": None,
                    "support": None,
                    "resistance": None,
                    "usd_krw": None,
                    "sox": None,
                    "sentiment": None,
                    "pred_low": None,
                    "pred_high": None,
                    "risk_score": None,
                },
                "history_count": len(history),
                "analysis_chart": {
                    "labels": [row.get("timestamp", "")[:16] for row in reversed(history)],
                    "prices": [row.get("last_close") for row in reversed(history)],
                    "pred_low": [row.get("pred_low") for row in reversed(history)],
                    "pred_high": [row.get("pred_high") for row in reversed(history)],
                },
                "message": None if analysis else "이 종목은 실시간 데이터만 제공됩니다.",
            }
        )
        return jsonify(stock)
    finally:
        connection.close()


@app.route("/api/stock/<string:name>/ai")
def api_stock_ai(name):
    connection = get_connection()
    try:
        if not table_exists(connection, "persona_discussion_log"):
            return jsonify({"error": "AI 분석 결과가 없습니다. llm_stock.py를 먼저 실행하세요."}), 404

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(persona_discussion_log)"
            ).fetchall()
        }
        latest_run = None
        if "run_id" in columns:
            latest_run = connection.execute(
                """
                SELECT run_id
                FROM persona_discussion_log
                WHERE ticker_name=? AND persona='최종결정자'
                  AND run_id IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()

        if latest_run:
            rows = connection.execute(
                """
                SELECT *
                FROM persona_discussion_log
                WHERE ticker_name=? AND run_id=?
                ORDER BY id
                """,
                (name, latest_run["run_id"]),
            ).fetchall()
        else:
            # run_id가 없는 기존 데이터는 페르소나별 최신 행을 사용합니다.
            rows = connection.execute(
                """
                SELECT p.*
                FROM persona_discussion_log p
                INNER JOIN (
                    SELECT persona, MAX(id) AS latest_id
                    FROM persona_discussion_log
                    WHERE ticker_name=?
                    GROUP BY persona
                ) latest ON latest.latest_id=p.id
                ORDER BY p.id
                """,
                (name,),
            ).fetchall()
        if not rows:
            return jsonify({"error": f"'{name}' AI 분석 결과가 없습니다."}), 404

        personas = []
        for row in rows:
            data = dict(row)
            raw_score = data.get("score")
            score = float(raw_score) if raw_score is not None else None
            if score is None:
                signal = "분석 실패"
            elif score > 0.1:
                signal = "매수"
            elif score < -0.1:
                signal = "매도"
            else:
                signal = "중립"
            personas.append(
                {
                    "persona": data.get("persona", ""),
                    "score": round(score, 3) if score is not None else None,
                    "signal": signal,
                    "opinion": data.get("opinion", ""),
                    "timestamp": data.get("timestamp", ""),
                }
            )

        final = next((item for item in personas if item["persona"] == "최종결정자"), None)
        return jsonify({"name": name, "personas": personas, "final": final})
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 실시간 API (동일 포트)
# ---------------------------------------------------------------------------
@app.route("/api/live/watchlist")
def api_live_watchlist():
    connection = get_connection()
    try:
        analyses = _latest_analysis_map(connection)
        personas = _latest_persona_map(connection)
        return jsonify(
            {
                "stocks": [
                    _stock_payload(name, analyses.get(name), personas.get(name))
                    for name in WATCHLIST
                ],
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    finally:
        connection.close()


@app.route("/api/live/candles/<string:symbol>")
def api_live_candles(symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOL_TO_NAME:
        return jsonify({"error": f"지원하지 않는 종목입니다: {symbol}"}), 404

    interval = request.args.get("interval", "1d")
    if interval not in INTERVAL_MAP:
        return jsonify({"error": f"지원하지 않는 구간입니다: {interval}"}), 400

    cache_key = (symbol, interval)
    with _cache_lock:
        cached = _candle_cache.get(cache_key)
        quote = dict(_price_cache.get(symbol, {}))
    if cached and time.time() - cached["saved_at"] < 20:
        return jsonify({**cached["payload"], "meta": quote, "cached": True})

    period, yf_interval = INTERVAL_MAP[interval]
    try:
        frame = yf.Ticker(symbol).history(
            period=period, interval=yf_interval, auto_adjust=True
        )
        if frame.empty:
            return jsonify({"symbol": symbol, "interval": interval, "candles": [], "meta": quote})

        candles = _frame_to_candles(_calculate_indicators(frame))
        payload = {
            "symbol": symbol,
            "interval": interval,
            "candles": candles,
            "meta": quote,
            "count": len(candles),
        }
        with _cache_lock:
            _candle_cache[cache_key] = {"saved_at": time.time(), "payload": payload}
        return jsonify(payload)
    except Exception as exc:
        app.logger.exception("캔들 데이터 조회 실패")
        return jsonify({"error": str(exc)}), 502


@app.route("/api/live/stream")
def api_live_stream():
    def generate():
        yield "retry: 6000\n\n"
        while True:
            with _cache_lock:
                snapshot = list(_price_cache.values())
            yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
            time.sleep(5)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    start_price_poller()
    print("=" * 60)
    print("  Stock AI 통합 대시보드")
    print(f"  DB  : {DB_PATH}")
    print(f"  URL : http://localhost:{PORT}")
    print("=" * 60)
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=os.environ.get("FLASK_DEBUG") == "1",
        threaded=True,
        use_reloader=False,
    )
