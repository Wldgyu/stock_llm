"""Render saved T+1 predictions and simulated trades without model training.

Usage: python ml_visualize_t1.py [--db PATH] [--name 삼성전자] [--output DIR]
"""
import argparse
import html
import json
import sqlite3
from pathlib import Path


def generate_reports(db_path, output, name=None):
    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f'DB 파일이 없습니다: {db_path}')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderer = Path(__file__).with_name('t1_report.js').read_text(encoding='utf-8')
    with sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('''SELECT a.* FROM analysis_log a
            JOIN (SELECT name,MAX(id) id FROM analysis_log GROUP BY name) b ON a.id=b.id
            WHERE (? IS NULL OR a.name=?) ORDER BY a.name''', (name,name)).fetchall()
    links = []
    for row in rows:
        row = dict(row)
        models = {}
        for key in ('lstm','tabpfn'):
            raw = row.get(key+'_metrics_json')
            if raw:
                try:
                    value = json.loads(raw)
                    if isinstance(value,dict): models[key] = value
                except (ValueError,TypeError):
                    print(f"경고: {row['name']} {key} 검증 JSON을 읽지 못했습니다.")
        payload = json.dumps({'models':models,'currency':'KRW' if row.get('symbol','').endswith(('.KS','.KQ')) else 'USD'},ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
        title = html.escape(str(row['name']))
        body = f'''<h1>{title} · T+1 예측 및 가상 투자</h1>
<p>분석 ID {row['id']} · 저장 시각 {html.escape(str(row.get('timestamp','')))} · 과거 테스트 예측이며 실제 주문 기록이 아닙니다.</p>
<p>차트는 모델 공통 평가 기간을 표시합니다. MAE·방향 정확도는 각 모델의 전체 T+1 테스트 기준입니다. 회색 구간은 현금 보유 상태이며, 예측 누락일에도 현금으로 전환합니다.</p>
<p>차트를 표시하려면 인터넷 연결이 필요합니다(Chart.js). 원본 DB는 수정하지 않습니다.</p>
<div id="charts"></div>'''
        if not models: body += '<p>저장된 모델 검증 결과가 없습니다.</p>'
        filename=f"t1_analysis_{int(row['id'])}.html"
        page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T+1 분석 리포트</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;background:#101827;color:#e5edf8}.validation-card{background:#192538;padding:24px;margin:24px 0;border-radius:12px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #40506a}summary{cursor:pointer}a{color:#80baff}</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>'''+body+f'<script type="application/json" id="report-data">{payload}</script><script>{renderer}</script></html>'
        path=output/filename; path.write_text(page,encoding='utf-8')
        links.append(f'<li><a href="{filename}">{title} · 분석 {row["id"]}</a></li>')
        print(f"리포트: {path}")
        if any(not m.get('strategy_backtest',{}).get('close_prices') for m in models.values()):
            print('  안내: 기존 기록에 실제 종가 배열이 없어 가격 차트 일부가 표시되지 않습니다. 새 ML 실행부터 저장됩니다.')
    index=output/'index.html'
    index.write_text('<!doctype html><meta charset="utf-8"><title>T+1 리포트 목록</title><h1>T+1 리포트</h1><ul>'+''.join(links)+'</ul>'+('해당 분석이 없습니다.' if not links else ''),encoding='utf-8')
    return index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',help='SQLite 경로 (생략하면 프로젝트 공통 DB 경로)')
    parser.add_argument('--name',help='종목명; 생략하면 모든 종목')
    parser.add_argument('--output',default='t1_reports',help='HTML 저장 폴더')
    args=parser.parse_args()
    if args.db:
        db=args.db
    else:
        from database import DB_PATH
        db=DB_PATH
    try:
        print(f'목록을 브라우저에서 여세요: {generate_reports(db,args.output,args.name)}')
    except (OSError,sqlite3.Error,ValueError) as exc:
        parser.exit(1,f'리포트 생성 실패: {exc}\n')

if __name__=='__main__': main()
