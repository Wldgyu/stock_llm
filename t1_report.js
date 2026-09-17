const state={oosCharts:[]};
const escapeHtml=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const formatNumber=(v,n=2)=>v==null?'—':Number(v).toFixed(n);
const formatPrice=(v,c)=>v==null?'—':Number(v).toLocaleString('ko-KR',{maximumFractionDigits:2})+' '+c;


function clearOosCharts() {
  state.oosCharts.forEach(chart => chart.destroy());
  state.oosCharts = [];
}

function renderOosCharts(entries, models, currency, container) {
  entries.forEach(([key, label]) => {
    const metrics = models[key];
    const bt = metrics.strategy_backtest;
    const panel = document.createElement("section");
    panel.className = "validation-card";
    const t1 = metrics.test?.by_horizon?.["T+1"];
    panel.innerHTML = `<h4>${label} · T+1 테스트 예측과 가상 매매</h4>
      <p>전체 T+1 테스트: MAE ${formatPrice(t1?.mae, currency)} · 방향 정확도 ${formatNumber(t1?.direction_accuracy, 2)}%</p>`;
    container.appendChild(panel);
    const savedStrategy = bt?.strategies?.[key];
    if (savedStrategy) {
      panel.insertAdjacentHTML("beforeend", `<p>수익률 ${formatNumber(savedStrategy.return_pct)}% · Sharpe ${formatNumber(savedStrategy.sharpe)} · MDD ${formatNumber(savedStrategy.mdd_pct)}% · 주문 ${savedStrategy.orders}회</p>`);
      if (!bt?.close_prices) {
        panel.insertAdjacentHTML("beforeend", `<details open><summary>저장된 매매 기록</summary><table><tr><th>날짜</th><th>거래</th><th>체결가</th></tr>${(savedStrategy.trades || []).map(t=>`<tr><td>${escapeHtml(t.date)}</td><td>${escapeHtml(t.side)}</td><td>${formatPrice(t.price,currency)}</td></tr>`).join("")}</table></details>`);
      }
    }
    if (!bt?.close_prices || !bt?.dates || !metrics.oos_t1_predictions || typeof Chart === "undefined") {
      panel.insertAdjacentHTML("beforeend", "<p>차트용 종가 데이터가 없습니다. 업데이트 후 ML 분석을 새로 실행하면 표시됩니다.</p>");
      return;
    }
    const strategy = bt.strategies[key];
    if (!strategy) return;
    const dates = bt.dates;
    const trades = strategy.trades || [];
    const common = new Set(bt.common_signal_dates || dates);
    let holding = false;
    const cash = dates.map(date => {
      trades.filter(t => t.date === date && t.side !== "sell_close").forEach(t => { holding = t.side === "buy"; });
      return !holding;
    });
    panel.insertAdjacentHTML("beforeend", `<p>${escapeHtml(bt.start)} ~ ${escapeHtml(bt.end)} · 회색 배경: 시가 거래 후 현금 보유 · 삼각형: 비용 반영 체결가<br>점선은 전날까지의 정보로 예측한 당일 종가입니다. 예측 누락일은 선이 끊깁니다. 마지막 날은 종가 청산합니다.</p>`);
    const wrap = document.createElement("div");
    wrap.style.cssText = "height:340px;position:relative";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", `${label} T+1 실제 종가, 예측가 및 매매 시점`);
    wrap.appendChild(canvas); panel.appendChild(wrap);
    const markers = side => trades.filter(t => side === "buy" ? t.side === "buy" : t.side !== "buy").map(t => ({x:t.date, y:t.price}));
    const shadeCash = {id: `cash_${key}`, beforeDatasetsDraw(chart) {
      const {ctx, chartArea, scales} = chart;
      if (!chartArea) return;
      ctx.save(); ctx.fillStyle = "rgba(140,150,170,0.18)";
      const x = scales.x;
      cash.forEach((isCash, i) => {
        if (!isCash) return;
        const center=x.getPixelForValue(i);
        const left=i ? (center+x.getPixelForValue(i-1))/2 : chartArea.left;
        const right=i+1<dates.length ? (center+x.getPixelForValue(i+1))/2 : chartArea.right;
        ctx.fillRect(left,chartArea.top,right-left,chartArea.bottom-chartArea.top);
      }); ctx.restore();
    }};
    state.oosCharts.push(new Chart(canvas, {type:"line", data:{labels:dates,datasets:[
      {label:"실제 종가",data:bt.close_prices,borderColor:"#61a8ff",pointRadius:0},
      {label:"T+1 예측 종가",data:dates.map(d => common.has(d) ? (metrics.oos_t1_predictions[d] ?? null) : null),borderColor:"#f5c518",borderDash:[5,4],pointRadius:0,spanGaps:false},
      {label:"매수",data:markers("buy"),showLine:false,pointStyle:"triangle",pointRadius:7,backgroundColor:"#00e5a0",borderColor:"#00e5a0"},
      {label:"매도/종료 청산",data:markers("sell"),showLine:false,pointStyle:"triangle",pointRotation:180,pointRadius:7,backgroundColor:"#ff4d6d",borderColor:"#ff4d6d"}
    ]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{maxTicksLimit:8}},y:{ticks:{callback:v=>formatPrice(v,currency)}}}},plugins:[shadeCash]}));
    const details=document.createElement("details");
    details.innerHTML=`<summary>매매 기록 ${trades.length}건 보기</summary><div style="max-height:240px;overflow:auto"><table><thead><tr><th>날짜</th><th>거래</th><th>체결가(슬리피지 반영)</th></tr></thead><tbody>${trades.map(t=>`<tr><td>${escapeHtml(t.date)}</td><td>${t.side === "buy" ? "매수" : t.side === "sell_close" ? "종료 종가 청산" : "매도"}</td><td>${formatPrice(t.price,currency)}</td></tr>`).join("")}</tbody></table></div>`;
    panel.appendChild(details);
  });
}

const payload=JSON.parse(document.getElementById('report-data').textContent);
renderOosCharts(Object.entries(payload.models).map(([key])=>[key,key.toUpperCase()]),payload.models,payload.currency,document.getElementById('charts'));
