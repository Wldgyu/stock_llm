# -*- coding: utf-8 -*-
"""
app.py
───────────────────────────────────────────────────────────────────
[역할] Stock AI 웹 대시보드 Flask API 서버
  - stock_analysis.db (SQLite) 를 읽어 REST API 제공
  - analysis_log 없어도 persona_discussion_log 만으로 동작 가능
  - 프론트엔드(index.html)에 JSON 데이터 반환

[실행] python app.py
[접속] http://localhost:5000
"""

import os
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# ── 설정 ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "stock_analysis.db"))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)


# ── DB 헬퍼 ──────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


# ── 정적 파일 ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ── API: DB 상태 ──────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    conn = get_conn()
    try:
        has_analysis = table_exists(conn, "analysis_log")
        has_persona  = table_exists(conn, "persona_discussion_log")
        analysis_count = conn.execute("SELECT COUNT(*) FROM analysis_log").fetchone()[0] if has_analysis else 0
        persona_count  = conn.execute("SELECT COUNT(*) FROM persona_discussion_log").fetchone()[0] if has_persona else 0
        return jsonify({
            "db_path":        DB_PATH,
            "has_analysis":   has_analysis,
            "has_persona":    has_persona,
            "analysis_count": analysis_count,
            "persona_count":  persona_count,
            "server_time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    finally:
        conn.close()


# ── 공통 헬퍼: persona 최종결정자 점수 맵 ─────────────────────────
def _get_persona_scores(conn):
    """ticker_name → {score, opinion, timestamp} 딕셔너리 반환"""
    scores = {}
    if not table_exists(conn, "persona_discussion_log"):
        return scores
    rows = conn.execute("""
        SELECT p.ticker_name, p.score, p.opinion, p.timestamp
        FROM persona_discussion_log p
        INNER JOIN (
            SELECT ticker_name, MAX(id) AS max_id
            FROM persona_discussion_log
            WHERE persona = '최종결정자'
            GROUP BY ticker_name
        ) m ON p.ticker_name = m.ticker_name AND p.id = m.max_id
    """).fetchall()
    for r in rows:
        scores[r["ticker_name"]] = {
            "score":     r["score"],
            "opinion":   (r["opinion"] or "")[:200],
            "timestamp": r["timestamp"],
        }
    return scores


def _trend(rsi_or_score, is_rsi=True):
    """RSI 또는 AI 점수 → (emoji, label) 반환"""
    v = float(rsi_or_score or 0)
    if is_rsi:
        if v > 70:   return "🔴", "과매수"
        if v < 30:   return "🟢", "과매도(반등)"
        return "⚖️", "중립"
    else:
        if v > 0.1:  return "🟢", "매수 우세"
        if v < -0.1: return "🔴", "매도 우세"
        return "⚖️", "중립"


# ── API: 전체 종목 목록 ──────────────────────────────────────────
@app.route("/api/stocks")
def api_stocks():
    """
    종목 목록 반환.
    - analysis_log 있음 → ML 지표 + AI 점수
    - analysis_log 없음 → persona_discussion_log 로 종목 구성
    """
    conn = get_conn()
    try:
        persona_scores = _get_persona_scores(conn)
        result = []

        # ── Case 1: analysis_log 있음 ───────────────────────────
        if table_exists(conn, "analysis_log"):
            rows = conn.execute("""
                SELECT a.* FROM analysis_log a
                INNER JOIN (
                    SELECT name, MAX(timestamp) AS max_ts
                    FROM analysis_log GROUP BY name
                ) b ON a.name = b.name AND a.timestamp = b.max_ts
                ORDER BY a.name
            """).fetchall()

            for row in rows:
                d   = dict(row)
                nm  = d.get("name", "")
                rsi = float(d.get("rsi") or 50)
                te, tl = _trend(rsi, is_rsi=True)
                ps     = persona_scores.get(nm, {})
                result.append({
                    "name":          nm,
                    "symbol":        d.get("symbol", ""),
                    "timestamp":     d.get("timestamp", ""),
                    "last_close":    d.get("last_close"),
                    "rsi":           round(rsi, 2),
                    "trend_emoji":   te,
                    "trend_label":   tl,
                    "support":       d.get("support"),
                    "resistance":    d.get("resistance"),
                    "usd_krw":       d.get("usd_krw"),
                    "sox":           d.get("sox"),
                    "sentiment":     d.get("sentiment"),
                    "pred_low":      d.get("pred_low"),
                    "pred_high":     d.get("pred_high"),
                    "risk_score":    d.get("risk_score"),
                    "final_score":   ps.get("score"),
                    "final_opinion": ps.get("opinion", ""),
                    "source":        "ml+llm",
                })

        # ── Case 2: persona_log 만 있음 ─────────────────────────
        else:
            if not persona_scores:
                return jsonify({"stocks": [], "message": "DB에 데이터가 없습니다. ml_stock.py 또는 llm_stock.py를 먼저 실행하세요."})

            for nm, ps in persona_scores.items():
                score = ps.get("score")
                te, tl = _trend(score, is_rsi=False)
                result.append({
                    "name":          nm,
                    "symbol":        "",
                    "timestamp":     ps.get("timestamp", ""),
                    "last_close":    None,
                    "rsi":           None,
                    "trend_emoji":   te,
                    "trend_label":   tl,
                    "support":       None,
                    "resistance":    None,
                    "usd_krw":       None,
                    "sox":           None,
                    "sentiment":     None,
                    "pred_low":      None,
                    "pred_high":     None,
                    "risk_score":    None,
                    "final_score":   score,
                    "final_opinion": ps.get("opinion", ""),
                    "source":        "llm_only",
                })

        return jsonify({"stocks": result})
    finally:
        conn.close()


# ── API: 종목 상세 ────────────────────────────────────────────────
@app.route("/api/stock/<string:name>")
def api_stock_detail(name: str):
    """
    특정 종목 상세 반환.
    - analysis_log 있음 → 차트/예측/지표 포함
    - analysis_log 없음 → persona 기반 최소 정보 반환 (404 아님)
    """
    conn = get_conn()
    try:
        # ── Case 1: analysis_log 있음 ───────────────────────────
        if table_exists(conn, "analysis_log"):
            rows = conn.execute("""
                SELECT * FROM analysis_log
                WHERE name = ?
                ORDER BY timestamp DESC
                LIMIT 30
            """, (name,)).fetchall()

            if rows:
                latest  = dict(rows[0])
                history = [dict(r) for r in rows]
                asc     = list(reversed(history))

                rsi = float(latest.get("rsi") or 50)
                te, tl = _trend(rsi, is_rsi=True)

                return jsonify({
                    "name":          name,
                    "symbol":        latest.get("symbol", ""),
                    "latest":        latest,
                    "rsi":           round(rsi, 2),
                    "trend_emoji":   te,
                    "trend_label":   tl,
                    "chart": {
                        "labels":    [r.get("timestamp", "")[:16] for r in asc],
                        "prices":    [r.get("last_close") for r in asc],
                        "pred_low":  [r.get("pred_low") for r in asc],
                        "pred_high": [r.get("pred_high") for r in asc],
                    },
                    "history_count": len(history),
                    "source":        "ml+llm",
                })

        # ── Case 2: persona_log 만 있음 ─────────────────────────
        if not table_exists(conn, "persona_discussion_log"):
            return jsonify({"error": "DB에 데이터가 없습니다. ml_stock.py 또는 llm_stock.py를 먼저 실행하세요."}), 404

        cnt = conn.execute(
            "SELECT COUNT(*) FROM persona_discussion_log WHERE ticker_name=?", (name,)
        ).fetchone()[0]
        if cnt == 0:
            return jsonify({"error": f"종목 '{name}' 데이터 없음"}), 404

        # 최종결정자 점수 → 트렌드
        final = conn.execute("""
            SELECT score, timestamp FROM persona_discussion_log
            WHERE ticker_name=? AND persona='최종결정자'
            ORDER BY id DESC LIMIT 1
        """, (name,)).fetchone()

        score = float(final["score"]) if final else 0
        ts    = final["timestamp"]    if final else ""
        te, tl = _trend(score, is_rsi=False)

        empty_latest = {
            "name": name, "symbol": "", "timestamp": ts,
            "last_close": None, "rsi": None, "trend": None,
            "support": None, "resistance": None,
            "usd_krw": None, "sox": None,
            "sentiment": None, "pred_low": None, "pred_high": None,
            "risk_score": None,
        }

        return jsonify({
            "name":          name,
            "symbol":        "",
            "latest":        empty_latest,
            "rsi":           None,
            "trend_emoji":   te,
            "trend_label":   tl,
            "chart":         {"labels": [], "prices": [], "pred_low": [], "pred_high": []},
            "history_count": 0,
            "source":        "llm_only",
            "message":       "ml_stock.py 실행 후 차트와 기술지표를 볼 수 있습니다.",
        })

    finally:
        conn.close()


# ── API: AI 페르소나 분석 ─────────────────────────────────────────
@app.route("/api/stock/<string:name>/ai")
def api_stock_ai(name: str):
    """특정 종목의 LLM 페르소나 토론 결과 반환 (최신 1세트)"""
    conn = get_conn()
    try:
        if not table_exists(conn, "persona_discussion_log"):
            return jsonify({"error": "persona_discussion_log 없음. llm_stock.py를 먼저 실행하세요."}), 404

        cnt = conn.execute(
            "SELECT COUNT(*) FROM persona_discussion_log WHERE ticker_name=?", (name,)
        ).fetchone()[0]
        if cnt == 0:
            return jsonify({"error": f"'{name}' AI 분석 결과 없음. llm_stock.py를 먼저 실행하세요."}), 404

        personas = conn.execute("""
            SELECT * FROM persona_discussion_log
            WHERE ticker_name=?
            ORDER BY id DESC
            LIMIT 3
        """, (name,)).fetchall()

        result = []
        for p in personas:
            d     = dict(p)
            score = float(d.get("score") or 0)
            if score > 0.1:
                signal, signal_color = "매수", "#00e5a0"
            elif score < -0.1:
                signal, signal_color = "매도", "#ff4d6d"
            else:
                signal, signal_color = "중립", "#f5c518"

            result.append({
                "persona":      d.get("persona", ""),
                "score":        round(score, 3),
                "signal":       signal,
                "signal_color": signal_color,
                "opinion":      d.get("opinion", ""),
                "timestamp":    d.get("timestamp", ""),
            })

        final = next((p for p in result if p["persona"] == "최종결정자"), None)
        return jsonify({"name": name, "personas": result, "final": final})

    finally:
        conn.close()


# ── 실행 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  📈 Stock AI Dashboard — Flask API 서버")
    print(f"  DB  : {DB_PATH}")
    print("  URL : http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
