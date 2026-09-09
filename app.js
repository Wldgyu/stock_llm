const API = "";

const state = {
  stocks: [],
  currentName: null,
  currentSymbol: null,
  currentExchange: "",
  currency: "USD",
  searchResults: [],
  interval: "1d",
  priceChart: null,
  rsiChart: null,
  eventSource: null,
  candleTimer: null,
  requestId: 0,
  aiOpen: false,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPrice(value, currency = state.currency) {
  const number = Number(value);
  if (value == null || Number.isNaN(number)) return "—";
  if (currency === "KRW" || currency?.endsWith(".KS")) {
    return `${number.toLocaleString("ko-KR", { maximumFractionDigits: 0 })} ₩`;
  }
  return `$${number.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatNumber(value, decimals = 2) {
  const number = Number(value);
  return value == null || Number.isNaN(number) ? "—" : number.toFixed(decimals);
}

function formatVolume(value) {
  const number = Number(value);
  if (!number) return "—";
  if (number >= 1e9) return `${(number / 1e9).toFixed(2)}B`;
  if (number >= 1e6) return `${(number / 1e6).toFixed(2)}M`;
  if (number >= 1e3) return `${(number / 1e3).toFixed(1)}K`;
  return number.toLocaleString();
}

function scoreColor(score) {
  const number = Number(score);
  if (score == null || Number.isNaN(number)) return "#8892a4";
  if (number > 0.1) return "#00e5a0";
  if (number < -0.1) return "#ff4d6d";
  return "#f5c518";
}

function scoreLabel(score) {
  const number = Number(score);
  if (score == null || Number.isNaN(number)) return "데이터 없음";
  if (number > 0.3) return "강력 매수";
  if (number > 0.1) return "매수 우세";
  if (number < -0.3) return "강력 매도";
  if (number < -0.1) return "매도 우세";
  return "중립";
}

function showPage(pageId) {
  document.querySelectorAll(".page").forEach((page) => page.classList.remove("active"));
  $(pageId)?.classList.add("active");
}

function showToast(message) {
  document.querySelector(".toast")?.remove();
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function loadStatus() {
  try {
    const data = await fetchJson(`${API}/api/status`);
    $("status-dot").classList.remove("offline");
    const searchCount = data.search_live_count
      ? ` · 검색 ${data.search_live_count}개`
      : "";
    $("status-text").textContent =
      `분석 ${data.analysis_count}건 · AI ${data.persona_count}건 · 실시간 ${data.live_count}/${data.watchlist_count || 6}${searchCount}`;
    $("server-time").textContent = data.server_time?.slice(11) || "--:--:--";
  } catch {
    $("status-dot").classList.add("offline");
    $("status-text").textContent = "서버 연결 오류";
  }
}

function renderSkeletonCards(count = 6) {
  return Array.from({ length: count }, () => `
    <div class="stock-card skeleton-card">
      <div class="skeleton skeleton-title"></div>
      <div class="skeleton skeleton-price"></div>
      <div class="skeleton skeleton-metrics"></div>
    </div>
  `).join("");
}

function renderStockCard(stock, index) {
  const displayPrice = stock.price ?? stock.last_close;
  const isLive = stock.price != null;
  const change = Number(stock.pct);
  const changeClass = change > 0 ? "up" : change < 0 ? "down" : "flat";
  const changeText = stock.pct == null ? "시세 대기" : `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`;
  const rsi = Number(stock.rsi);
  const rsiColor = stock.rsi == null ? "#8892a4" : rsi > 70 ? "#ff4d6d" : rsi < 30 ? "#00e5a0" : "#f5c518";
  const sentiment = Number(stock.sentiment);
  const sentimentColor = stock.sentiment == null ? "#8892a4" : sentiment > 0 ? "#00e5a0" : sentiment < 0 ? "#ff4d6d" : "#f5c518";
  const finalScore = stock.final_score == null ? "—" : `${Number(stock.final_score) >= 0 ? "+" : ""}${Number(stock.final_score).toFixed(2)}`;

  return `
    <article class="stock-card" data-stock-index="${index}" tabindex="0" role="button">
      <div class="card-accent-line"></div>
      <div class="card-header">
        <div class="card-ticker-wrap">
          <span class="card-name">${escapeHtml(stock.name)}</span>
          <span class="card-symbol">${escapeHtml(stock.symbol)}</span>
        </div>
        <span class="card-trend-badge">${escapeHtml(stock.trend_emoji)} ${escapeHtml(stock.trend_label)}</span>
      </div>
      <div class="card-price-row">
        <div class="card-price" data-role="price">${formatPrice(displayPrice, stock.currency)}</div>
        <span class="price-source">${isLive ? "실시간" : "최근 분석 종가"}</span>
      </div>
      <div class="card-change ${changeClass}" data-role="change">${changeText}</div>
      <div class="card-metrics">
        <div class="metric-item">
          <span class="metric-label">RSI</span>
          <strong class="metric-value" style="color:${rsiColor}">${formatNumber(stock.rsi, 1)}</strong>
        </div>
        <div class="metric-item">
          <span class="metric-label">감성</span>
          <strong class="metric-value" style="color:${sentimentColor}">${formatNumber(stock.sentiment, 2)}</strong>
        </div>
        <div class="metric-item">
          <span class="metric-label">AI 점수</span>
          <strong class="metric-value" style="color:${scoreColor(stock.final_score)}">${finalScore}</strong>
        </div>
      </div>
    </article>
  `;
}

function bindStockCards() {
  $("stocks-grid").querySelectorAll(".stock-card[data-stock-index]").forEach((card) => {
    const open = () => openDetail(state.stocks[Number(card.dataset.stockIndex)]);
    card.addEventListener("click", open);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
  });
}

async function loadStocks() {
  const grid = $("stocks-grid");
  grid.innerHTML = renderSkeletonCards();
  try {
    const data = await fetchJson(`${API}/api/stocks`);
    state.stocks = data.stocks || [];
    if (!state.stocks.length) {
      grid.innerHTML = '<div class="empty-state"><h3>표시할 종목이 없습니다.</h3></div>';
      return;
    }
    grid.innerHTML = state.stocks.map(renderStockCard).join("");
    bindStockCards();
    const timestamp = state.stocks.find((stock) => stock.timestamp)?.timestamp;
    $("last-update").textContent = timestamp ? `최근 분석 ${timestamp.slice(0, 16)}` : "분석 데이터 대기 중";
  } catch (error) {
    grid.innerHTML = `
      <div class="empty-state">
        <h3>서버에서 종목을 불러오지 못했습니다.</h3>
        <p>${escapeHtml(error.message)}</p>
      </div>`;
  }
}

function clearSearchResults() {
  state.searchResults = [];
  $("stock-search-results").hidden = true;
  $("stock-search-results").innerHTML = "";
}

function renderSearchResults(results) {
  const container = $("stock-search-results");
  state.searchResults = results;
  if (!results.length) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }

  container.innerHTML = results.map((stock, index) => `
    <button class="stock-search-result" type="button" data-search-index="${index}">
      <span class="search-result-name">
        <strong>${escapeHtml(stock.name)}</strong>
        <span class="search-result-symbol">${escapeHtml(stock.symbol)}</span>
      </span>
      <span class="search-result-meta">
        ${stock.tracked ? '<span class="search-result-tracked">분석 종목</span>' : ""}
        <span>${escapeHtml(stock.exchange || stock.quote_type || "")}</span>
      </span>
    </button>
  `).join("");
  container.hidden = false;

  container.querySelectorAll("[data-search-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const stock = state.searchResults[Number(button.dataset.searchIndex)];
      clearSearchResults();
      openDetail(stock);
    });
  });
}

async function searchStocks() {
  const query = $("stock-search-input").value.trim();
  if (!query) {
    $("stock-search-status").textContent = "";
    clearSearchResults();
    return;
  }

  $("stock-search-status").textContent = "종목 검색 중...";
  clearSearchResults();
  try {
    const data = await fetchJson(`${API}/api/search?q=${encodeURIComponent(query)}`);
    const results = data.results || [];
    renderSearchResults(results);
    $("stock-search-status").textContent = results.length
      ? `${results.length}개 종목을 찾았습니다.`
      : "검색 결과가 없습니다. 종목명 또는 티커를 확인하세요.";
  } catch (error) {
    $("stock-search-status").textContent = `검색 실패: ${error.message}`;
  }
}

function updateQuote(quote) {
  const stock = state.stocks.find((item) => item.symbol === quote.symbol);
  if (stock) Object.assign(stock, quote);

  document.querySelectorAll(".stock-card[data-stock-index]").forEach((card) => {
    const item = state.stocks[Number(card.dataset.stockIndex)];
    if (item?.symbol !== quote.symbol) return;
    const change = Number(quote.pct || 0);
    const changeElement = card.querySelector('[data-role="change"]');
    card.querySelector('[data-role="price"]').textContent = formatPrice(quote.price, quote.currency);
    changeElement.textContent = `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`;
    changeElement.className = `card-change ${change > 0 ? "up" : change < 0 ? "down" : "flat"}`;
  });

  if (quote.symbol === state.currentSymbol) updateDetailQuote(quote);
}

function updateDetailQuote(quote) {
  const change = Number(quote.pct || 0);
  const priceElement = $("detail-price");
  const changeElement = $("detail-change");
  priceElement.textContent = formatPrice(quote.price, quote.currency || state.currency);
  priceElement.className = `detail-price ${change > 0 ? "up" : change < 0 ? "down" : ""}`;
  changeElement.textContent =
    `${change >= 0 ? "+" : ""}${formatNumber(quote.change, 2)} ` +
    `(${change >= 0 ? "+" : ""}${change.toFixed(2)}%)`;
  changeElement.className = `detail-change ${change > 0 ? "up" : change < 0 ? "down" : "flat"}`;
  $("detail-timestamp").textContent = quote.ts ? `실시간 갱신 ${quote.ts}` : "실시간 갱신";
}

function connectLiveStream() {
  state.eventSource?.close();
  state.eventSource = new EventSource(`${API}/api/live/stream`);
  state.eventSource.onmessage = (event) => {
    try {
      JSON.parse(event.data).forEach(updateQuote);
      $("status-dot").classList.remove("offline");
      $("server-time").textContent = new Date().toLocaleTimeString("ko-KR", { hour12: false });
    } catch (error) {
      console.error("SSE 데이터 처리 실패", error);
    }
  };
  state.eventSource.onerror = () => $("status-dot").classList.add("offline");
}

function analysisKpi(label, value, subtext, color = "var(--text-primary)") {
  return `
    <div class="kpi-card">
      <span class="kpi-label">${escapeHtml(label)}</span>
      <strong class="kpi-value" style="color:${color}">${escapeHtml(value)}</strong>
      <span class="kpi-sub">${escapeHtml(subtext)}</span>
    </div>`;
}

function renderAnalysis(data) {
  const latest = data.latest || {};
  const rsi = Number(latest.rsi);
  const sentiment = Number(latest.sentiment);
  const risk = Number(latest.risk_score);
  const isNumber = (value) => value != null && value !== "" && Number.isFinite(Number(value));
  const isPositive = (value) => isNumber(value) && Number(value) > 0;
  const cards = [];

  if (isNumber(latest.rsi)) {
    cards.push(analysisKpi("RSI", formatNumber(latest.rsi, 1), data.trend_label || "추세", rsi > 70 ? "#ff4d6d" : rsi < 30 ? "#00e5a0" : "#f5c518"));
  }
  if (isPositive(latest.support)) {
    cards.push(analysisKpi("지지선", formatPrice(latest.support, data.currency), "최근 20일 저가", "#00e5a0"));
  }
  if (isPositive(latest.resistance)) {
    cards.push(analysisKpi("저항선", formatPrice(latest.resistance, data.currency), "최근 20일 고가", "#ff4d6d"));
  }
  if (isNumber(latest.sentiment)) {
    cards.push(analysisKpi("감성점수", formatNumber(latest.sentiment, 2), "뉴스 감성", sentiment > 0 ? "#00e5a0" : sentiment < 0 ? "#ff4d6d" : "#f5c518"));
  }
  if (isPositive(latest.usd_krw)) {
    cards.push(analysisKpi("USD/KRW", formatNumber(latest.usd_krw, 1), "환율", "#00d4ff"));
  }
  if (isPositive(latest.sox)) {
    cards.push(analysisKpi("SOX", formatNumber(latest.sox, 1), "반도체 지수", "#9b6fff"));
  }
  if (isNumber(latest.risk_score) && risk >= 1) {
    cards.push(analysisKpi("리스크", `${risk} / 10`, "변동성 기반", risk > 7 ? "#ff4d6d" : risk < 4 ? "#00e5a0" : "#f5c518"));
  }

  $("analysis-kpis").innerHTML = cards.join("");
  $("analysis-section").hidden = cards.length === 0;

  $("detail-trend").textContent = `${data.trend_emoji || "⚖️"} ${data.trend_label || "중립"}`;
  $("analysis-close").textContent = latest.last_close == null
    ? ""
    : `최근 분석 종가 ${formatPrice(latest.last_close, data.currency)} · ${latest.timestamp?.slice(0, 16) || ""}`;
  $("analysis-message").textContent = cards.length ? (data.message || "") : "";
  renderPredictions(latest, data.currency);
}

function renderPredictions(latest, currency) {
  const low = latest.pred_low;
  const high = latest.pred_high;
  const lstmValues = [latest.lstm_t1, latest.lstm_t4, latest.lstm_t7];
  const tabpfnValues = [latest.tabpfn_t1, latest.tabpfn_t4, latest.tabpfn_t7];
  const hasRange = low != null && high != null;
  const hasLstmDetails = lstmValues.some((value) => value != null);
  const hasTabpfnDetails = tabpfnValues.some((value) => value != null);

  if (!hasRange && !hasLstmDetails && !hasTabpfnDetails) {
    $("prediction-section").hidden = true;
    $("prediction-cards").innerHTML = "";
    $("model-prediction-details").innerHTML = "";
    return;
  }
  $("prediction-section").hidden = false;
  const middle = hasRange ? (Number(low) + Number(high)) / 2 : null;
  $("prediction-cards").innerHTML = hasRange ? `
    <div class="prediction-card">
      <span class="pred-day">LSTM 예측 하단</span>
      <strong class="pred-price down">${formatPrice(low, currency)}</strong>
      <span class="pred-range">T+1 · T+4 · T+7 최솟값</span>
    </div>
    <div class="prediction-card">
      <span class="pred-day">LSTM 예측 중앙</span>
      <strong class="pred-price">${formatPrice(middle, currency)}</strong>
      <span class="pred-range">예측 범위 중앙값</span>
    </div>
    <div class="prediction-card">
      <span class="pred-day">LSTM 예측 상단</span>
      <strong class="pred-price up">${formatPrice(high, currency)}</strong>
      <span class="pred-range">T+1 · T+4 · T+7 최댓값</span>
    </div>` : "";

  const renderModelRow = (label, values, className) => `
    <div class="model-prediction-row ${className}">
      <strong class="model-prediction-name">${label}</strong>
      ${values.map((value, index) => `
        <span class="model-prediction-item">
          <small>${["1일 뒤", "4일 뒤", "7일 뒤"][index]}</small>
          <b>${formatPrice(value, currency)}</b>
        </span>
      `).join("")}
    </div>`;

  const detailRows = [];
  if (hasLstmDetails) {
    detailRows.push(renderModelRow("LSTM", lstmValues, "lstm-model-row"));
  }
  if (hasTabpfnDetails) {
    detailRows.push(renderModelRow("TabPFN", tabpfnValues, "tabpfn-model-row"));
  }
  $("model-prediction-details").innerHTML = detailRows.join("");
}

function resetDetail() {
  $("analysis-section").hidden = false;
  $("prediction-section").hidden = false;
  $("analysis-kpis").innerHTML = Array.from({ length: 7 }, () =>
    '<div class="kpi-card"><div class="skeleton kpi-skeleton"></div></div>'
  ).join("");
  $("live-kpis").innerHTML = "";
  $("prediction-cards").innerHTML = '<div class="skeleton prediction-skeleton"></div>';
  $("model-prediction-details").innerHTML = "";
  $("analysis-message").textContent = "";
  $("analysis-close").textContent = "";
  setChartLoading(true);
  destroyCharts();
  closeAiPanel();
}

async function openDetail(stock) {
  if (!stock) return;
  state.currentName = stock.name;
  state.currentSymbol = stock.symbol;
  state.currentExchange = stock.exchange || "";
  state.currency = stock.currency || (stock.symbol?.endsWith(".KS") ? "KRW" : "USD");
  state.interval = "1d";
  state.requestId += 1;
  const requestId = state.requestId;

  showPage("detail-page");
  resetDetail();
  document.querySelectorAll(".interval-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.interval === "1d");
  });
  $("detail-name").textContent = stock.name;
  $("detail-symbol").textContent = stock.symbol || "—";
  updateDetailQuote(stock);

  loadCandles(requestId);
  const hasBatchAnalysis = stock.tracked !== false;
  $("ai-button").hidden = !hasBatchAnalysis;
  if (hasBatchAnalysis) {
    try {
      const data = await fetchJson(`${API}/api/stock/${encodeURIComponent(stock.name)}`);
      if (requestId !== state.requestId) return;
      state.currentSymbol = data.symbol || state.currentSymbol;
      state.currency = data.currency || state.currency;
      renderAnalysis(data);
    } catch (error) {
      if (requestId === state.requestId) {
        $("analysis-message").textContent = error.message;
        renderAnalysis({ latest: {}, currency: state.currency, message: error.message });
      }
    }
  } else {
    $("analysis-section").hidden = true;
    $("prediction-section").hidden = true;
    $("detail-trend").textContent = "🔎 검색 종목";
    $("analysis-close").textContent = "ML·LLM 배치 분석 데이터 없음";
  }

  clearInterval(state.candleTimer);
  state.candleTimer = setInterval(() => {
    if ($("detail-page").classList.contains("active")) loadCandles(state.requestId, false);
  }, 30000);
}

function setChartLoading(visible) {
  $("chart-loading").style.display = visible ? "flex" : "none";
}

function renderLiveKpis(last, meta) {
  const items = [
    ["MA5", formatPrice(last.ma5, state.currency)],
    ["MA20", formatPrice(last.ma20, state.currency)],
    ["MA60", formatPrice(last.ma60, state.currency)],
    ["BB 상단", formatPrice(last.bb_up, state.currency)],
    ["BB 하단", formatPrice(last.bb_low, state.currency)],
    ["RSI(14)", formatNumber(last.rsi, 1)],
    ["거래량", formatVolume(meta.volume || last.v)],
  ];
  $("live-kpis").innerHTML = items.map(([label, value]) => `
    <div class="live-kpi">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");
}

async function loadCandles(requestId = state.requestId, showLoading = true) {
  if (!state.currentSymbol) return;
  if (showLoading) setChartLoading(true);
  try {
    const params = new URLSearchParams({
      interval: state.interval,
      name: state.currentName || state.currentSymbol,
      currency: state.currency,
      exchange: state.currentExchange,
    });
    const data = await fetchJson(
      `${API}/api/live/candles/${encodeURIComponent(state.currentSymbol)}?${params}`
    );
    if (requestId !== state.requestId) return;
    const candles = data.candles || [];
    if (!candles.length) throw new Error("해당 구간의 가격 데이터가 없습니다.");
    const last = candles.at(-1);
    renderLiveKpis(last, data.meta || {});
    if (data.meta?.price != null) updateDetailQuote(data.meta);
    renderLiveCharts(candles);
  } catch (error) {
    if (requestId === state.requestId) showToast(`차트 로드 실패: ${error.message}`);
  } finally {
    if (requestId === state.requestId) setChartLoading(false);
  }
}

function destroyCharts() {
  state.priceChart?.destroy();
  state.rsiChart?.destroy();
  state.priceChart = null;
  state.rsiChart = null;
}

const CHART_COLORS = {
  price: "#4f8ef7",
  ma5: "#f5c518",
  ma20: "#9d6fff",
  ma60: "#ff8c42",
  upper: "rgba(255,77,109,.55)",
  lower: "rgba(0,229,160,.55)",
  grid: "rgba(255,255,255,.05)",
  tick: "#5f6b82",
};

function renderLiveCharts(candles) {
  if (typeof Chart === "undefined") {
    showToast("Chart.js를 불러오지 못했습니다.");
    return;
  }
  destroyCharts();
  const labels = candles.map((candle) => candle.t);
  const values = (key) => candles.map((candle) => candle[key]);
  const priceContext = $("live-price-chart").getContext("2d");
  const gradient = priceContext.createLinearGradient(0, 0, 0, 360);
  gradient.addColorStop(0, "rgba(79,142,247,.22)");
  gradient.addColorStop(1, "rgba(79,142,247,0)");

  state.priceChart = new Chart(priceContext, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "BB 상단", data: values("bb_up"), borderColor: CHART_COLORS.upper, borderWidth: 1, pointRadius: 0, fill: "+1", backgroundColor: "rgba(157,111,255,.05)", tension: 0.25, spanGaps: true },
        { label: "BB 하단", data: values("bb_low"), borderColor: CHART_COLORS.lower, borderWidth: 1, pointRadius: 0, fill: false, tension: 0.25, spanGaps: true },
        { label: "MA60", data: values("ma60"), borderColor: CHART_COLORS.ma60, borderWidth: 1.2, pointRadius: 0, tension: 0.25, spanGaps: true },
        { label: "MA20", data: values("ma20"), borderColor: CHART_COLORS.ma20, borderWidth: 1.2, pointRadius: 0, tension: 0.25, spanGaps: true },
        { label: "MA5", data: values("ma5"), borderColor: CHART_COLORS.ma5, borderWidth: 1.2, pointRadius: 0, tension: 0.25, spanGaps: true },
        { label: "종가", data: values("c"), borderColor: CHART_COLORS.price, backgroundColor: gradient, borderWidth: 2, pointRadius: 0, fill: true, tension: 0.25, spanGaps: true },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: CHART_COLORS.tick, boxWidth: 12, font: { size: 10 } } },
        tooltip: {
          backgroundColor: "rgba(8,13,26,.96)",
          callbacks: {
            label: (context) => ` ${context.dataset.label}: ${formatPrice(context.parsed.y)}`,
          },
        },
      },
      scales: {
        x: { grid: { color: CHART_COLORS.grid }, ticks: { color: CHART_COLORS.tick, maxTicksLimit: 9, maxRotation: 0 } },
        y: { position: "right", grid: { color: CHART_COLORS.grid }, ticks: { color: CHART_COLORS.tick, callback: (value) => formatPrice(value) } },
      },
    },
  });

  const rsiContext = $("live-rsi-chart").getContext("2d");
  state.rsiChart = new Chart(rsiContext, {
    type: "line",
    data: {
      labels,
      datasets: [{ label: "RSI", data: values("rsi"), borderColor: CHART_COLORS.ma5, borderWidth: 1.5, pointRadius: 0, tension: 0.25, spanGaps: true }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: {
          min: 0,
          max: 100,
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.tick, callback: (value) => [30, 50, 70].includes(value) ? value : null },
        },
      },
    },
    plugins: [{
      id: "rsiZones",
      beforeDraw(chart) {
        const { ctx, chartArea, scales } = chart;
        if (!chartArea) return;
        const y70 = scales.y.getPixelForValue(70);
        const y30 = scales.y.getPixelForValue(30);
        ctx.save();
        ctx.fillStyle = "rgba(255,77,109,.10)";
        ctx.fillRect(chartArea.left, chartArea.top, chartArea.width, y70 - chartArea.top);
        ctx.fillStyle = "rgba(0,229,160,.10)";
        ctx.fillRect(chartArea.left, y30, chartArea.width, chartArea.bottom - y30);
        ctx.restore();
      },
    }],
  });
}

function personaIcon(persona) {
  return { 주식전문가: "📊", 뉴스기업전문가: "📰", 최종결정자: "⚖️" }[persona] || "🤖";
}

async function openAiPanel() {
  if (!state.currentName) return;
  state.aiOpen = true;
  $("ai-panel").classList.add("open");
  $("ai-panel").setAttribute("aria-hidden", "false");
  $("ai-button").classList.add("active");
  $("ai-stock-name").textContent = state.currentName;
  const content = $("ai-panel-content");
  content.innerHTML = '<div class="ai-loading"><span class="spinner"></span><span>AI 분석 로드 중...</span></div>';

  try {
    const data = await fetchJson(`${API}/api/stock/${encodeURIComponent(state.currentName)}/ai`);
    if (!state.aiOpen || data.name !== state.currentName) return;
    const order = ["주식전문가", "뉴스기업전문가", "최종결정자"];
    const personas = order.map((name) => data.personas.find((item) => item.persona === name)).filter(Boolean);
    const final = data.final;
    const summary = final ? `
      <div class="final-decision-summary">
        <strong class="final-summary-score" style="color:${scoreColor(final.score)}">${final.score == null ? "오류" : `${Number(final.score) >= 0 ? "+" : ""}${Number(final.score).toFixed(2)}`}</strong>
        <div class="final-summary-info">
          <span class="final-summary-label">최종 판단</span>
          <strong class="final-summary-verdict" style="color:${scoreColor(final.score)}">${scoreLabel(final.score)}</strong>
        </div>
        <p class="final-summary-opinion">${escapeHtml(final.opinion || "")}</p>
      </div>` : "";

    const cards = personas.map((persona) => {
      const hasScore = persona.score != null;
      const score = hasScore ? Number(persona.score) : 0;
      const percent = ((score + 1) / 2) * 100;
      const gauge = hasScore ? `
        <div class="score-gauge">
          <span class="score-gauge-center"></span>
          <span class="score-gauge-fill" style="left:${score >= 0 ? 50 : percent}%;width:${Math.abs(percent - 50)}%;background:${score >= 0 ? "#00e5a0" : "#ff4d6d"}"></span>
        </div>` : "";
      return `
        <article class="persona-card">
          <div class="persona-header">
            <span class="persona-name"><span class="persona-icon">${personaIcon(persona.persona)}</span>${escapeHtml(persona.persona)}</span>
            <span class="persona-score-wrap">
              <strong class="persona-score" style="color:${scoreColor(persona.score)}">${hasScore ? `${score >= 0 ? "+" : ""}${score.toFixed(2)}` : "—"}</strong>
              <span class="persona-signal" style="color:${scoreColor(persona.score)}">${escapeHtml(persona.signal)}</span>
            </span>
          </div>
          ${gauge}
          <p class="persona-opinion">${escapeHtml(persona.opinion || "의견 없음")}</p>
        </article>`;
    }).join("");
    content.innerHTML = summary + `<div class="ai-personas-grid">${cards}</div>`;
  } catch (error) {
    content.innerHTML = `<div class="empty-state"><h3>AI 분석 없음</h3><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function closeAiPanel() {
  state.aiOpen = false;
  $("ai-panel").classList.remove("open");
  $("ai-panel").setAttribute("aria-hidden", "true");
  $("ai-button").classList.remove("active");
}

function goHome() {
  state.requestId += 1;
  state.currentName = null;
  state.currentSymbol = null;
  state.currentExchange = "";
  clearInterval(state.candleTimer);
  destroyCharts();
  closeAiPanel();
  showPage("main-page");
}

document.addEventListener("DOMContentLoaded", async () => {
  $("home-button").addEventListener("click", goHome);
  $("back-button").addEventListener("click", goHome);
  $("ai-button").addEventListener("click", () => state.aiOpen ? closeAiPanel() : openAiPanel());
  $("ai-close-button").addEventListener("click", closeAiPanel);
  $("stock-search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    searchStocks();
  });
  $("stock-search-input").addEventListener("input", (event) => {
    if (!event.target.value.trim()) {
      $("stock-search-status").textContent = "";
      clearSearchResults();
    }
  });
  $("interval-bar").addEventListener("click", (event) => {
    const button = event.target.closest(".interval-btn");
    if (!button || button.dataset.interval === state.interval) return;
    state.interval = button.dataset.interval;
    document.querySelectorAll(".interval-btn").forEach((item) => item.classList.toggle("active", item === button));
    loadCandles(state.requestId);
  });

  await Promise.all([loadStatus(), loadStocks()]);
  connectLiveStream();
  setInterval(loadStatus, 60000);
});

window.addEventListener("beforeunload", () => {
  state.eventSource?.close();
  clearInterval(state.candleTimer);
});
