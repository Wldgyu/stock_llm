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
