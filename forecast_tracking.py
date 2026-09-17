"""Evaluate stored, pre-session T+1 forecasts against confirmed daily closes."""
from datetime import datetime
import pandas as pd
import yfinance as yf
from trading_dates import calendar_for, completed_bars


def t1_history(connection, name, unique_targets=False):
    columns={r[1] for r in connection.execute('PRAGMA table_info(analysis_log)')}
    if 'target_t1' not in columns:
        return []
    for field,kind in [('t1_actual_close','REAL'),('t1_evaluated_at','TEXT')]:
        if field not in columns:
            connection.execute(f'ALTER TABLE analysis_log ADD COLUMN {field} {kind}')
    connection.commit()
    rows=[dict(r) for r in connection.execute('SELECT * FROM analysis_log WHERE name=? ORDER BY timestamp DESC, id DESC',(name,))]
    if not rows:return []
    symbol=rows[0]['symbol']
    calendar=calendar_for(symbol)
    now=pd.Timestamp.now(tz='UTC')
    pending=[]
    results=[]
    for row in rows:
        target=row.get('target_t1'); base=row.get('data_date')
        status='untracked'
        if target and base:
            schedule=calendar.schedule(start_date=target,end_date=target)
            if schedule.empty:
                status='invalid_date'
            else:
                # Existing timestamps are generated in the application's Seoul local timezone.
                saved=pd.Timestamp(row['timestamp']).tz_localize('Asia/Seoul').tz_convert('UTC')
                if saved>=schedule['market_open'].iloc[0]:
                    status='late'
                elif row.get('t1_actual_close') is not None:
                    status='complete'
                elif now<schedule['market_close'].iloc[0]:
                    status='pending'
                else:
                    status='awaiting_data';pending.append(row)
        row['_status']=status
        results.append(row)
    if unique_targets:
        selected = {}
        for row in results:
            if row['_status'] in ('late', 'untracked', 'invalid_date'):
                continue
            # Sorted newest first: never select retrospectively by prediction accuracy.
            selected.setdefault(row['target_t1'], row)
        results = sorted(selected.values(), key=lambda r: r['target_t1'], reverse=True)[:30]
    else:
        results = results[:30]
    selected_ids = {r['id'] for r in results}
    pending = [r for r in pending if r['id'] in selected_ids]
    if pending:
        try:
            start=min(r['data_date'] for r in pending)
            end=(pd.Timestamp(max(r['target_t1'] for r in pending))+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            frame=yf.Ticker(symbol).history(start=start,end=end,auto_adjust=True)
            frame=completed_bars(frame,symbol)
            closes={str(i.date()):float(v) for i,v in frame['Close'].items() if pd.notna(v)}
            for row in pending:
                actual=closes.get(row['target_t1']);base_close=closes.get(row['data_date'])
                if actual is None or not base_close or not row.get('last_close'):continue
                # Normalize later-adjusted history to the original forecast price basis.
                actual=actual/base_close*row['last_close']
                stamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                connection.execute('UPDATE analysis_log SET t1_actual_close=?,t1_evaluated_at=? WHERE id=? AND t1_actual_close IS NULL',(actual,stamp,row['id']))
                row['t1_actual_close']=actual;row['_status']='complete'
            connection.commit()
        except Exception:
            connection.rollback()
            for row in pending:
                row['_status']='fetch_failed'
    output=[]
    for row in results:
        actual=row.get('t1_actual_close') if row['_status']=='complete' else None
        models={}
        for key in ('lstm','tabpfn'):
            pred=row.get(key+'_t1');base=row.get('last_close')
            error=pred-actual if pred is not None and actual is not None else None
            direction=None
            if error is not None and base is not None:
                sign=lambda v:(v>0)-(v<0)
                direction=sign(pred-base)==sign(actual-base)
            models[key]={'prediction':pred,'error':error,'error_pct':error/actual*100 if error is not None and actual else None,'direction_correct':direction}
        output.append({'analysis_id':row['id'],'saved_at':row['timestamp'],'data_date':row.get('data_date'),'target_date':row.get('target_t1'),'status':row['_status'],'actual':actual,'reference_close':row.get('last_close'),'models':models})
    return output


def realized_summary(history, minimum_samples=20):
    """Model influence is a cautious evidence category, not a probability."""
    import math
    summary = {}
    for key in ('lstm', 'tabpfn'):
        samples = []
        for row in history:
            model = row['models'][key]
            values = [model.get('prediction'), row.get('actual'), row.get('reference_close')]
            if row['status'] != 'complete' or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                continue
            if row['actual'] <= 0:
                continue
            samples.append(row)
        count = len(samples)
        if not count:
            summary[key] = {'samples': 0, 'influence': '평가 불가: 보조 정보만 사용'}
            continue
        mae = sum(abs(r['models'][key]['error']) for r in samples) / count
        naive = sum(abs(r['reference_close'] - r['actual']) for r in samples) / count
        bias = sum(r['models'][key]['error_pct'] for r in samples) / count
        mape = sum(abs(r['models'][key]['error_pct']) for r in samples) / count
        direction = sum(r['models'][key]['direction_correct'] is True for r in samples) / count * 100
        improvement = (naive - mae) / naive * 100 if naive > 0 else None
        if count < minimum_samples:
            influence = '표본 부족: 비중 상향 금지'
        elif improvement is not None and improvement >= 10 and direction >= 50:
            influence = '제한적 상향 검토: 보조 근거로만 활용'
        elif mae > naive:
            influence = '비중 하향: naive보다 가격 오차 큼'
        else:
            influence = '낮은 비중 유지: 개선 근거 제한적'
        latest = max(samples, key=lambda r: r['target_date'])
        summary[key] = {'samples':count,'mae':mae,'naive_mae':naive,'mape_pct':mape,'bias_pct':bias,
            'direction_accuracy':direction,'improvement_pct':improvement,'influence':influence,
            'latest':{'target_date':latest['target_date'],'prediction':latest['models'][key]['prediction'],
                'actual':latest['actual'],'error_pct':latest['models'][key]['error_pct']}}
    return summary


def realized_summary_for_llm(connection, name):
    history = t1_history(connection, name, unique_targets=True)
    summary = realized_summary(history)
    # User preference: ML versus news, not LSTM versus TabPFN.
    ml_weight, news_weight = 40, 60
    model_results = [summary[key] for key in ('lstm', 'tabpfn')]
    if all(result['samples'] >= 20 for result in model_results):
        if all(result['influence'].startswith('제한적 상향') for result in model_results):
            ml_weight, news_weight = 50, 50
        elif all(result['mae'] > result['naive_mae'] for result in model_results):
            ml_weight, news_weight = 30, 70
    text = ['실제 사전 예측 성과(T+1): 목표일 개장 전 저장된 최신 1건만 사용, 최대 최근 30개 목표일.',
            '중복 실행·개장 후 저장은 제외. 평가 대기는 틀린 예측으로 세지 않음.',
            '비중 기준은 운영용 휴리스틱이며 검증된 확률/자동 학습이 아님. 최소 20개 완료 목표일, naive 대비 MAE 10% 이상 개선 및 방향 50% 이상일 때만 제한적 상향 검토.',
            f"평가 대기/조회 미완료 목표일: {sum(r['status'] != 'complete' for r in history)}개"]
    for key, result in summary.items():
        if not result['samples']:
            text.append(f'{key}: 평가 완료 0건, 평가 불가. 예측 비중 상향 금지.')
            continue
        last = result['latest']
        text.append(f"{key}: {result['samples']}개 목표일, MAE {result['mae']:.4f}, naive MAE {result['naive_mae']:.4f}, MAPE {result['mape_pct']:.2f}%, 평균 편향 {result['bias_pct']:+.2f}%, 방향 {result['direction_accuracy']:.2f}%. {result['influence']}.")
        text.append(f"  가장 최근 완료 {last['target_date']}: 예측 {last['prediction']:.4f}, 실제 {last['actual']:.4f}, 오차율 {last['error_pct']:+.2f}%.")
    text.append(f"판단 근거 참고 비중: ML 예측 {ml_weight}% : 뉴스 {news_weight}%. 기본 40:60. 충분한 실제 성과로 두 모델 모두 개선된 경우만 50:50, 두 모델 모두 naive 미달이면 30:70. 표본 부족·성과 혼재·평가 불가는 기본 유지. 한 번에 최대 10%포인트 범위이고 실행마다 누적 조정하지 않음. LSTM:TabPFN 비율이 아니며 전문가 점수를 이 비율로 산술 합산하지 말 것. 뉴스도 제목만 있거나 확인되지 않았으면 높은 신뢰를 부여하지 말고 전체 판단을 보류할 수 있음. 비율은 참고 우선순위이며 확률·신뢰도·자동매매 비중이 아님.")
    text.append('한 번 맞았다고 신뢰도를 올리지 말 것. 비중 상향은 매수 점수 상승이 아니라 모델 방향성 근거의 상대적 비중 조절이며, 매수·매도 모두에 적용. T+1 성과를 T+4/T+7 신뢰도로 확대하지 말 것.')
    return '\n'.join(text)
