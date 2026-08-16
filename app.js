// ================================================================
// app.js — Stock AI Dashboard Frontend Logic
// ================================================================

const API = "http://localhost:5000";

// ── State ─────────────────────────────────────────────────────────
let currentStock = null;
let chartInstance = null;
let aiPanelOpen  = false;

// ── DOM refs ──────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Page Router ───────────────────────────────────────────────────
function showPage(pageId) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  $(pageId).classList.add("active");
}

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, duration = 4000) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = `⚠️ ${msg}`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── Formatters ────────────────────────────────────────────────────
function formatPrice(val, symbol) {
  if (val == null || val === "") return "—";
  const n = parseFloat(val);
  if (isNaN(n)) return "—";
  // 한국 종목(KS)이면 원화, 아니면 달러
  if (symbol && symbol.includes(".KS")) {
    return n.toLocaleString("ko-KR", { maximumFractionDigits: 0 }) + " ₩";
  }
  return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatNum(val, decimals = 2) {
  if (val == null || val === "") return "—";
  const n = parseFloat(val);
  return isNaN(n) ? "—" : n.toFixed(decimals);
}

function scoreColor(score) {
  if (score == null) return "#8892a4";
  const s = parseFloat(score);
  if (s > 0.1)  return "#00e5a0";
  if (s < -0.1) return "#ff4d6d";
  return "#f5c518";
}

function scoreLabel(score) {
  if (score == null) return "데이터 없음";
  const s = parseFloat(score);
  if (s > 0.3)  return "강력 매수";
  if (s > 0.1)  return "매수 우세";
  if (s < -0.3) return "강력 매도";
  if (s < -0.1) return "매도 우세";
  return "중립";
}

function personaIcon(persona) {
  const map = { "주식전문가": "📊", "뉴스기업전문가": "📰", "최종결정자": "⚖️" };
  return map[persona] || "🤖";
}

function personaIconBg(persona) {
  const map = {
    "주식전문가":     "rgba(0,212,255,0.12)",
    "뉴스기업전문가": "rgba(245,197,24,0.12)",
    "최종결정자":     "rgba(110,86,255,0.15)",
  };
  return map[persona] || "rgba(255,255,255,0.06)";
}

// ── Status check ─────────────────────────────────────────────────
async function loadStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    const dot  = $("status-dot");
    const text = $("status-text");
    if (dot && text) {
      dot.style.background = "#00e5a0";
      text.textContent = `DB: analysis ${data.analysis_count}건 | persona ${data.persona_count}건`;
    }
  } catch {
    const dot  = $("status-dot");
    const text = $("status-text");
    if (dot) dot.style.background = "#ff4d6d";
    if (text) text.textContent = "서버 연결 오류";
  }
}

// ── Main Page: Load stock list ─────────────────────────────────────
async function loadStocks() {
  const grid = $("stocks-grid");
  grid.innerHTML = renderSkeletonCards(3);

  try {
    const res  = await fetch(`${API}/api/stocks`);
    const data = await res.json();

    if (!data.stocks || data.stocks.length === 0) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1">
          <div class="icon">📭</div>
          <h3>분석 데이터가 없습니다</h3>
          <p>먼저 <code>python ml_stock.py</code>를 실행하여<br>분석 데이터를 생성해주세요.</p>
          ${data.message ? `<p style="margin-top:12px;font-size:13px;color:var(--accent-yellow)">${data.message}</p>` : ""}
        </div>`;
      return;
    }

    // Timestamp
    const ts = data.stocks[0]?.timestamp?.slice(0, 16) || "";
    const tsEl = $("last-update");
    if (tsEl && ts) tsEl.textContent = "마지막 분석: " + ts;

    grid.innerHTML = data.stocks.map(s => renderStockCard(s)).join("");

    // Attach click events
    grid.querySelectorAll(".stock-card").forEach(card => {
      card.addEventListener("click", () => {
        const name = card.dataset.name;
        openDetail(name);
      });
    });

    // Animate score bars
    requestAnimationFrame(() => {
      data.stocks.forEach(s => {
        const bar = document.querySelector(`.ai-score-bar[data-name="${s.name}"]`);
        if (!bar || s.final_score == null) return;
        const score = parseFloat(s.final_score);
        const pct = ((score + 1) / 2) * 100; // -1~+1 → 0~100%
        const center = 50;
        if (score >= 0) {
          bar.style.left   = center + "%";
          bar.style.width  = (pct - center) + "%";
          bar.style.background = "linear-gradient(90deg, #00e5a0, #00d4ff)";
        } else {
          bar.style.left   = pct + "%";
          bar.style.width  = (center - pct) + "%";
          bar.style.background = "linear-gradient(90deg, #ff4d6d, #ff6b8a)";
        }
      });
    });

  } catch (e) {
    console.error(e);
    grid.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1">
        <div class="icon">🔌</div>
        <h3>서버에 연결할 수 없습니다</h3>
        <p><code>python app.py</code>를 실행하고 새로고침해주세요.</p>
      </div>`;
  }
}

function renderStockCard(s) {
  const priceStr   = formatPrice(s.last_close, s.symbol);
  const rsiColor   = s.rsi > 70 ? "#ff4d6d" : s.rsi < 30 ? "#00e5a0" : "#f5c518";
  const sentColor  = s.sentiment > 0 ? "#00e5a0" : s.sentiment < 0 ? "#ff4d6d" : "#f5c518";
  const scoreStr   = s.final_score != null ? (parseFloat(s.final_score) >= 0 ? "+" : "") + parseFloat(s.final_score).toFixed(2) : "N/A";
  const sc         = scoreColor(s.final_score);

  return `
    <div class="stock-card" data-name="${s.name}" id="card-${s.name}">
      <div class="card-accent-line"></div>
      <div class="card-header">
        <div class="card-ticker-wrap">
          <div class="card-name">${s.name}</div>
          <div class="card-symbol">${s.symbol || ""}</div>
        </div>
        <div class="card-trend-badge">
          ${s.trend_emoji} <span style="color:var(--text-secondary)">${s.trend_label}</span>
        </div>
      </div>

      <div class="card-price">
        ${priceStr}
        <span class="card-price-label">현재가</span>
      </div>

      <div class="card-metrics">
        <div class="metric-item">
          <div class="metric-label">RSI</div>
          <div class="metric-value" style="color:${rsiColor}">${formatNum(s.rsi, 1)}</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">감성점수</div>
          <div class="metric-value" style="color:${sentColor}">${s.sentiment != null ? (s.sentiment >= 0 ? "+" : "") + formatNum(s.sentiment, 2) : "—"}</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">리스크</div>
          <div class="metric-value">${s.risk_score != null ? s.risk_score + "/10" : "—"}</div>
        </div>
      </div>

      <div class="card-ai-score">
        <span class="ai-score-label">🤖 AI 점수</span>
        <div class="ai-score-bar-wrap">
          <div class="ai-score-bar" data-name="${s.name}" style="width:0%"></div>
        </div>
        <span class="ai-score-value" style="color:${sc}">${scoreStr}</span>
      </div>
    </div>`;
}

function renderSkeletonCards(n) {
  return Array.from({ length: n }, () => `
    <div class="stock-card" style="pointer-events:none">
      <div class="skeleton" style="height:18px;width:50%;margin-bottom:8px"></div>
      <div class="skeleton" style="height:12px;width:30%;margin-bottom:24px"></div>
      <div class="skeleton" style="height:32px;width:60%;margin-bottom:20px"></div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px">
        <div class="skeleton" style="height:56px"></div>
        <div class="skeleton" style="height:56px"></div>
        <div class="skeleton" style="height:56px"></div>
      </div>
      <div class="skeleton" style="height:42px"></div>
    </div>`).join("");
}

// ── Detail Page ────────────────────────────────────────────────────
async function openDetail(name) {
  currentStock = name;
  aiPanelOpen  = false;

  showPage("detail-page");

  // Reset AI panel
  const panel = $("ai-panel");
  panel.classList.remove("open");
  $("ai-btn").classList.remove("active");
  $("ai-panel-content").innerHTML = "";

  // Reset content
  $("detail-name").textContent    = name;
  $("detail-symbol").textContent  = "로딩 중...";
  $("detail-price").textContent   = "—";
  $("detail-timestamp").textContent = "";
  $("kpi-row").innerHTML          = renderKpiSkeletons(5);
  $("prediction-cards").innerHTML = `<div class="skeleton" style="height:80px"></div><div class="skeleton" style="height:80px"></div><div class="skeleton" style="height:80px"></div>`;

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  try {
    const res  = await fetch(`${API}/api/stock/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    const l = data.latest || {};
    $("detail-symbol").textContent    = data.symbol || "";
    $("detail-price").textContent     = formatPrice(l.last_close, data.symbol);
    $("detail-timestamp").textContent = l.timestamp ? "분석: " + l.timestamp.slice(0, 16) : "";
    $("detail-trend-badge").innerHTML = `${data.trend_emoji} ${data.trend_label}`;

    // KPI row
    $("kpi-row").innerHTML = renderKpiCards(l, data);

    // llm_only 모드 — 차트/예측 없음 안내
    if (data.source === "llm_only" || !data.chart || data.chart.labels.length === 0) {
      document.querySelector(".chart-section").innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:200px;gap:14px;color:var(--text-secondary)">
          <div style="font-size:40px">📊</div>
          <div style="font-weight:600;color:var(--text-primary)">차트 데이터 없음</div>
          <div style="font-size:13px;text-align:center;line-height:1.7">
            ${data.message || "ml_stock.py를 실행하면 가격 차트와 기술지표를 볼 수 있습니다."}
          </div>
        </div>`;
      $("prediction-cards").innerHTML = `
        <div style="grid-column:1/-1;color:var(--text-muted);font-size:13px;text-align:center;padding:20px">
          LSTM 예측가는 ml_stock.py 실행 후 표시됩니다.
        </div>`;
    } else {
      // Chart
      renderChart(data.chart, data.symbol);
      // Prediction cards
      renderPredictionCards(l, data.symbol);
    }

  } catch (e) {
    showToast("데이터 로드 실패: " + e.message);
  }
}

function renderKpiSkeletons(n) {
  return Array.from({ length: n }, () =>
    `<div class="kpi-card"><div class="skeleton" style="height:12px;width:60%;margin-bottom:12px"></div><div class="skeleton" style="height:22px;width:80%"></div></div>`
  ).join("");
}

function renderKpiCards(l, data) {
  const items = [
    { label: "RSI",    value: formatNum(l.rsi, 1),    sub: data.trend_label,   color: l.rsi > 70 ? "#ff4d6d" : l.rsi < 30 ? "#00e5a0" : "#f5c518" },
    { label: "지지선",  value: formatPrice(l.support, data.symbol),  sub: "Support",  color: "#00e5a0" },
    { label: "저항선",  value: formatPrice(l.resistance, data.symbol), sub: "Resistance", color: "#ff4d6d" },
    { label: "감성점수", value: l.sentiment != null ? (l.sentiment >= 0 ? "+" : "") + formatNum(l.sentiment, 2) : "—", sub: "뉴스 감성", color: l.sentiment > 0 ? "#00e5a0" : l.sentiment < 0 ? "#ff4d6d" : "#f5c518" },
    { label: "USD/KRW", value: l.usd_krw ? parseFloat(l.usd_krw).toFixed(1) : "—", sub: "환율", color: "#00d4ff" },
    { label: "SOX 지수", value: l.sox ? parseFloat(l.sox).toFixed(1) : "—", sub: "반도체 지수", color: "#9b6fff" },
    { label: "리스크",  value: l.risk_score != null ? l.risk_score + " / 10" : "—", sub: "Risk Score", color: l.risk_score > 7 ? "#ff4d6d" : l.risk_score < 4 ? "#00e5a0" : "#f5c518" },
  ];

  return items.map(item => `
    <div class="kpi-card">
      <div class="kpi-label">${item.label}</div>
      <div class="kpi-value" style="color:${item.color}">${item.value}</div>
      <div class="kpi-sub">${item.sub}</div>
    </div>`).join("");
}

function renderChart(chart, symbol) {
  const ctx = $("price-chart");
  if (!ctx) return;

  if (chartInstance) chartInstance.destroy();

  const labels = chart.labels || [];
  const prices = chart.prices || [];
  const predLow  = chart.pred_low || [];
  const predHigh = chart.pred_high || [];

  // Gradient fill
  const chartCtx = ctx.getContext("2d");
  const gradient = chartCtx.createLinearGradient(0, 0, 0, 280);
  gradient.addColorStop(0, "rgba(110, 86, 255, 0.25)");
  gradient.addColorStop(1, "rgba(110, 86, 255, 0.00)");

  const gradCyan = chartCtx.createLinearGradient(0, 0, 0, 280);
  gradCyan.addColorStop(0, "rgba(0, 212, 255, 0.12)");
  gradCyan.addColorStop(1, "rgba(0, 212, 255, 0.00)");

  chartInstance = new Chart(chartCtx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "실제가",
          data: prices,
          borderColor: "#6e56ff",
          backgroundColor: gradient,
          borderWidth: 2.5,
          pointRadius: labels.length > 15 ? 0 : 4,
          pointHoverRadius: 6,
          pointBackgroundColor: "#6e56ff",
          tension: 0.4,
          fill: true,
        },
        {
          label: "예측 하단",
          data: predLow,
          borderColor: "rgba(0,229,160,0.6)",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.4,
        },
        {
          label: "예측 상단",
          data: predHigh,
          borderColor: "rgba(255,77,109,0.6)",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, easing: "easeInOutQuart" },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(10,15,35,0.95)",
          borderColor: "rgba(110,86,255,0.4)",
          borderWidth: 1,
          titleColor: "#f0f4ff",
          bodyColor: "#8892a4",
          padding: 12,
          callbacks: {
            label: ctx => {
              const v = ctx.parsed.y;
              if (v == null) return null;
              return ` ${ctx.dataset.label}: ${formatPrice(v, symbol)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.04)", drawBorder: false },
          ticks: { color: "#4a5568", font: { size: 11, family: "JetBrains Mono" }, maxTicksLimit: 8, maxRotation: 0 },
        },
        y: {
          position: "right",
          grid: { color: "rgba(255,255,255,0.04)", drawBorder: false },
          ticks: {
            color: "#4a5568",
            font: { size: 11, family: "JetBrains Mono" },
            callback: v => symbol?.includes(".KS") ? v.toLocaleString("ko-KR") + "₩" : "$" + v.toLocaleString("en-US"),
          }
        }
      }
    }
  });
}

function renderPredictionCards(l, symbol) {
  const predLow  = l.pred_low;
  const predHigh = l.pred_high;
  const midVal   = (predLow != null && predHigh != null)
    ? (parseFloat(predLow) + parseFloat(predHigh)) / 2
    : null;

  $("prediction-cards").innerHTML = `
    <div class="prediction-card">
      <div class="pred-day">🔮 T+1 (내일)</div>
      <div class="pred-price">${formatPrice(midVal, symbol)}</div>
      <div class="pred-range">예측 중앙값</div>
    </div>
    <div class="prediction-card">
      <div class="pred-day">📉 예측 하단</div>
      <div class="pred-price" style="color:#ff4d6d">${formatPrice(predLow, symbol)}</div>
      <div class="pred-range">LSTM 하한</div>
    </div>
    <div class="prediction-card">
      <div class="pred-day">📈 예측 상단</div>
      <div class="pred-price" style="color:#00e5a0">${formatPrice(predHigh, symbol)}</div>
      <div class="pred-range">LSTM 상한</div>
    </div>`;
}

// ── AI Panel ───────────────────────────────────────────────────────
async function toggleAiPanel() {
  if (!currentStock) return;

  aiPanelOpen = !aiPanelOpen;
  const panel     = $("ai-panel");
  const btn       = $("ai-btn");
  const mainArea  = $("detail-main");
  const nameLabel = $("ai-panel-stock-name");

  if (aiPanelOpen) {
    panel.classList.add("open");
    btn.classList.add("active");
    if (mainArea)  mainArea.style.paddingBottom = "500px";
    if (nameLabel) nameLabel.textContent = currentStock;
    await loadAiPanel(currentStock);
  } else {
    panel.classList.remove("open");
    btn.classList.remove("active");
    if (mainArea) mainArea.style.paddingBottom = "";
  }
}

function closeAiPanel() {
  aiPanelOpen = false;
  $("ai-panel").classList.remove("open");
  $("ai-btn").classList.remove("active");
  const mainArea = $("detail-main");
  if (mainArea) mainArea.style.paddingBottom = "";
}

async function loadAiPanel(name) {
  const content = $("ai-panel-content");
  content.innerHTML = `
    <div class="ai-loading">
      <div class="spinner"></div>
      <span>AI 분석 데이터 로드 중...</span>
    </div>`;

  try {
    const res  = await fetch(`${API}/api/stock/${encodeURIComponent(name)}/ai`);
    if (!res.ok) {
      const err = await res.json();
      content.innerHTML = `
        <div class="empty-state" style="padding:40px 20px">
          <div class="icon">🤖</div>
          <h3>AI 분석 없음</h3>
          <p>${err.error || "llm_stock.py를 먼저 실행하세요."}</p>
        </div>`;
      return;
    }

    const data    = await res.json();
    const personas = data.personas || [];
    const final    = data.final;

    // ── 상단: 최종 결정 요약 바 ────────────────────────────────
    let summaryHtml = "";
    if (final) {
      const fs   = parseFloat(final.score);
      const fsc  = scoreColor(fs);
      const sign = fs >= 0 ? "+" : "";
      summaryHtml = `
        <div class="final-decision-summary">
          <div class="final-summary-score" style="color:${fsc}">${sign}${fs.toFixed(2)}</div>
          <div class="final-summary-info">
            <div class="final-summary-label">⚖️ 최종결정자 판단</div>
            <div class="final-summary-verdict" style="color:${fsc}">${scoreLabel(fs)}</div>
          </div>
          <div class="final-summary-opinion">${final.opinion || ""}</div>
        </div>`;
    }

    // ── 하단: 3인 페르소나 가로 그리드 ────────────────────────
    // 순서 고정: 주식전문가 → 뉴스기업전문가 → 최종결정자
    const ORDER = ["주식전문가", "뉴스기업전문가", "최종결정자"];
    const sorted = ORDER.map(nm => personas.find(p => p.persona === nm)).filter(Boolean);

    const cardsHtml = sorted.map(p => {
      const score  = parseFloat(p.score);
      const sc     = scoreColor(score);
      const sign   = score >= 0 ? "+" : "";
      const pct    = ((score + 1) / 2) * 100;
      const gaugeLeft  = score >= 0 ? 50 : pct;
      const gaugeWidth = Math.abs(pct - 50);
      const gaugeColor = score >= 0 ? "#00e5a0" : "#ff4d6d";
      const isFinal    = p.persona === "최종결정자";

      return `
        <div class="persona-card" style="${isFinal ? "border-color:rgba(110,86,255,0.3);background:rgba(110,86,255,0.05)" : ""}">
          <div class="persona-header">
            <div class="persona-name">
              <div class="persona-icon" style="background:${personaIconBg(p.persona)}">${personaIcon(p.persona)}</div>
              ${p.persona}
            </div>
            <div class="persona-score-wrap">
              <div class="persona-score" style="color:${sc}">${sign}${Math.abs(score).toFixed(2)}</div>
              <div class="persona-signal" style="color:${sc}">${p.signal}</div>
            </div>
          </div>
          <div class="score-gauge">
            <div class="score-gauge-center"></div>
            <div class="score-gauge-fill" style="left:${gaugeLeft}%;width:${gaugeWidth}%;background:${gaugeColor}"></div>
          </div>
          <div class="persona-opinion">${p.opinion || "의견 없음"}</div>
        </div>`;
    }).join("");

    content.innerHTML = summaryHtml + `<div class="ai-personas-grid">${cardsHtml}</div>`;

  } catch (e) {
    content.innerHTML = `
      <div class="empty-state" style="padding:40px 20px">
        <div class="icon">❌</div>
        <h3>로드 실패</h3>
        <p>${e.message}</p>
      </div>`;
  }
}

// ── Nav: Back to list ─────────────────────────────────────────────
function goBack() {
  currentStock = null;
  aiPanelOpen  = false;
  const panel = $("ai-panel");
  if (panel) panel.classList.remove("open");
  showPage("main-page");
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  showPage("main-page");
  loadStatus();
  loadStocks();
});
