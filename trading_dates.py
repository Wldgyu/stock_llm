"""Exchange-session dates for the tracked Korean and Nasdaq stocks."""
from functools import lru_cache
import pandas as pd
import pandas_market_calendars as mcal

@lru_cache(maxsize=4)
def calendar_for(symbol):
    return mcal.get_calendar('XKRX' if symbol.endswith(('.KS','.KQ')) else 'NASDAQ')

def local_date(value, calendar):
    value=pd.Timestamp(value)
    return (value.tz_convert(calendar.tz) if value.tzinfo else value).date()

def prediction_dates(index_value, symbol):
    calendar=calendar_for(symbol)
    base=local_date(index_value,calendar)
    schedule=calendar.schedule(start_date=pd.Timestamp(base)+pd.Timedelta(days=1),end_date=pd.Timestamp(base)+pd.Timedelta(days=60))
    if len(schedule)<7:
        raise ValueError('거래소 달력에 예측 목표일이 부족합니다.')
    return {'data_date':str(base), **{f'target_t{n}':str(schedule.index[n-1].date()) for n in (1,4,7)}}

def saved_before_open(timestamp, timestamp_utc, target_date, symbol, market_open=None):
    """Compare the saved instant with the target session open.

    Older records have no offset and were stored as Seoul local time.
    """
    try:
        if market_open is None:
            schedule = calendar_for(symbol).schedule(start_date=target_date, end_date=target_date)
            if schedule.empty:
                return None
            market_open = schedule['market_open'].iloc[0]
        saved = pd.Timestamp(timestamp_utc or timestamp)
        if pd.isna(saved):
            return None
        if saved.tzinfo is None:
            saved = saved.tz_localize('Asia/Seoul')
        return bool(saved.tz_convert('UTC') < market_open)
    except (TypeError, ValueError, OverflowError):
        return None


def forecast_validity(data_date, target_date, symbol, now=None):
    """Require the latest completed session and a forecast made before target open."""
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('now must include timezone')
    now = now.tz_convert('UTC')
    calendar = calendar_for(symbol)
    local_day = now.tz_convert(calendar.tz).date()
    schedule = calendar.schedule(start_date=pd.Timestamp(local_day) - pd.Timedelta(days=60),
                                 end_date=pd.Timestamp(local_day) + pd.Timedelta(days=30))
    completed = schedule[schedule['market_close'] <= now]
    if completed.empty:
        return False, '최근 확정 거래일을 확인할 수 없습니다.'
    latest = str(completed.index[-1].date())
    if str(data_date) != latest:
        return False, f'기준 일봉 {data_date}이 최신 확정 거래일 {latest}과 다릅니다.'
    future = schedule[schedule.index > completed.index[-1]]
    if future.empty or str(future.index[0].date()) != str(target_date):
        return False, 'T+1 목표일이 다음 거래일과 일치하지 않습니다.'
    if now >= future['market_open'].iloc[0]:
        return False, f'T+1 목표일 {target_date}의 장이 이미 시작됐습니다.'
    return True, '최신 확정 일봉 기준, T+1 개장 전'

def completed_bars(frame,symbol,now=None):
    if frame.empty:
        return frame
    calendar=calendar_for(symbol)
    dates=[local_date(value,calendar) for value in frame.index]
    schedule=calendar.schedule(start_date=min(dates),end_date=max(dates))
    closes={stamp.date():close for stamp,close in schedule['market_close'].items()}
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('now must include timezone')
    return frame.loc[[day in closes and closes[day]<=now for day in dates]].copy()
