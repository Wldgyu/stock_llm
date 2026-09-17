"""Out-of-sample, long/cash comparison using signals known before the open."""
import numpy as np


def compare_strategies(frame, predictions, entry_threshold=0.005, fee_bps=10, slippage_bps=5):
    """predictions maps model names to {target trading date: T+1 price}.

    All strategies use the same continuous date range. Model signals are used
    only on the intersection of OOS dates; missing-signal days target cash.
    Signals use the previous close, execute at today's open, and value at close.
    Fractional shares, no leverage, no interest, no taxes. Last close liquidates.
    """
    if not predictions or not np.isfinite([entry_threshold, fee_bps, slippage_bps]).all():
        raise ValueError('Invalid backtest configuration')
    if entry_threshold < 0 or fee_bps < 0 or slippage_bps < 0 or slippage_bps >= 10000:
        raise ValueError('Invalid costs or threshold')
    keys = [str(d.date()) for d in frame.index]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate trading dates')
    common = set.intersection(*(set(p) for p in predictions.values()))
    positions = [i for i, d in enumerate(keys) if d in common and i > 0]
    if not positions:
        raise ValueError('No common out-of-sample dates')
    start, end = min(positions), max(positions)
    window = frame.iloc[start:end + 1]
    prices = window[['Open', 'Close']].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Invalid execution prices')
    fee, slip = fee_bps / 10000, slippage_bps / 10000
    outcomes = {}
    for name in [*predictions, 'rsi', 'buy_hold']:
        cash, shares = 1.0, 0.0
        equity, trades = [1.0], []
        for i in range(start, end + 1):
            day = keys[i]
            previous = frame.iloc[i - 1]
            row = frame.iloc[i]
            desired = shares > 0
            if name in predictions:
                if day not in common:
                    desired = False
                else:
                    predicted = float(predictions[name][day])
                    base = float(previous['Close'])
                    if not np.isfinite([predicted, base]).all() or base <= 0:
                        raise ValueError('Invalid prediction or reference close')
                    expected = predicted / base - 1
                    if expected > entry_threshold:
                        desired = True
                    elif expected <= 0:
                        desired = False
            elif name == 'rsi':
                rsi = float(previous['RSI'])
                if rsi < 35:
                    desired = True
                elif rsi > 65:
                    desired = False
            else:
                desired = True
            if desired and shares == 0:
                price = float(row['Open']) * (1 + slip)
                shares = cash / (price * (1 + fee))
                cash = 0.0
                trades.append({'date': day, 'side': 'buy', 'price': price})
            elif not desired and shares > 0:
                price = float(row['Open']) * (1 - slip)
                cash = shares * price * (1 - fee)
                shares = 0.0
                trades.append({'date': day, 'side': 'sell', 'price': price})
            if i == end and shares > 0:
                price = float(row['Close']) * (1 - slip)
                cash = shares * price * (1 - fee)
                shares = 0.0
                trades.append({'date': day, 'side': 'sell_close', 'price': price})
            equity.append(cash + shares * float(row['Close']))
        values = np.array(equity)
        returns = values[1:] / values[:-1] - 1
        std = returns.std(ddof=1) if len(returns) > 1 else 0.0
        outcomes[name] = {
            'return_pct': float((values[-1] - 1) * 100),
            'sharpe': float(np.sqrt(252) * returns.mean() / std) if std > 1e-12 else 0.0,
            'mdd_pct': float((values / np.maximum.accumulate(values) - 1).min() * 100),
            'orders': len(trades), 'trades': trades,
            'equity': [float(v) for v in equity],
        }
    return {
        'start': keys[start], 'end': keys[end], 'days': end - start + 1,
        'signal_days': len(positions), 'dates': keys[start:end + 1],
        'close_prices': [float(v) for v in window['Close']],
        'common_signal_dates': [keys[i] for i in positions],
        'entry_threshold_pct': entry_threshold * 100,
        'fee_bps_per_side': fee_bps, 'slippage_bps_per_side': slippage_bps,
        'rules': 'T+1; previous-close signal, next open fill; long/cash; missing signal exits; final close liquidation; fractional shares; zero risk-free rate; excludes taxes',
        'strategies': outcomes,
    }
