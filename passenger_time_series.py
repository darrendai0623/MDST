"""Simple ARIMA and weekly SARIMA analysis. Requires pandas, numpy, statsmodels, matplotlib.

Run: python3 passenger_time_series.py
Fixed model orders keep this example simple; no parameter search is performed.
Weekly SARIMA models seven-day seasonality, not annual or holiday effects.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring a desktop window.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

FORECAST_DAYS = 90
TEST_DAYS = 90
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'time_series_outputs'


def fit_model(history, name):
    """Scale counts to millions for numerical stability during fitting."""
    scaled = history / 1_000_000
    if name == 'ARIMA':
        result = ARIMA(scaled, order=(1, 1, 1)).fit(method_kwargs={'maxiter': 300})
    else:
        result = SARIMAX(scaled, order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 7)).fit(disp=False, maxiter=300)
    if not result.mle_retvals.get('converged', False):
        raise RuntimeError(f'{name} fit did not converge; review the model before using forecasts.')
    return result


def plot_results(passengers, backtest, forecast, metrics):
    """Save readable charts in passenger units of millions."""
    colors = {'ARIMA': '#2878B5', 'SARIMA': '#C46A16'}
    styles = {'ARIMA': '--', 'SARIMA': '-'}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})

    def format_axis(ax, start, end):
        ax.set_ylabel('Passengers (millions)')
        ax.grid(axis='y', alpha=0.2)
        ticks = pd.date_range(start, end, periods=5)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        ax.set_xlim(start, end)
        ax.legend(loc='upper left', fontsize=9)

    def save(fig, filename, note):
        fig.text(0.08, 0.025, note, fontsize=9, color='#555555')
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(passengers.index, passengers / 1e6, color='#A0A7AF',
            linewidth=0.6, label='Daily observations')
    ax.plot(passengers.index, passengers.rolling(30).mean() / 1e6,
            color='#2878B5', linewidth=1.8, label='30-day trailing average')
    ax.set_title('Daily passenger volume | 2019–2025', loc='left')
    format_axis(ax, passengers.index[0], passengers.index[-1])
    ax.set_ylim(bottom=0)
    save(fig, 'passenger_history.png', 'Source: passenger_volume_data.csv. Rolling average requires 30 daily observations.')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(backtest['Date'], backtest['actual'] / 1e6,
            color='#30343B', linewidth=1.5, label='Actual')
    for name in colors:
        mae = metrics.loc[metrics['model'] == name, 'MAE'].iloc[0]
        ax.plot(backtest['Date'], backtest[name] / 1e6, color=colors[name],
                linestyle=styles[name], label=f'{name} | MAE {mae:,.0f}')
    ax.set_title('90-day holdout | Actual vs. predicted passenger volume', loc='left')
    format_axis(ax, backtest['Date'].iloc[0], backtest['Date'].iloc[-1])
    save(fig, 'holdout_comparison.png', 'Each model predicts all 90 days from the same training cutoff; MAE is measured in passengers.')

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    recent = passengers.iloc[-90:]
    for ax, name in zip(axes, colors):
        ax.plot(recent.index, recent / 1e6, color='#30343B', linewidth=1.2, label='Observed')
        ax.plot(forecast['Date'], forecast[name] / 1e6, color=colors[name],
                linestyle=styles[name], label=f'{name} forecast')
        ax.fill_between(forecast['Date'], forecast[f'{name}_lower_95'] / 1e6,
                        forecast[f'{name}_upper_95'] / 1e6, color=colors[name],
                        alpha=0.18, label='95% prediction interval')
        ax.axvline(passengers.index[-1], color='#777777', linestyle=':', linewidth=1)
        ax.axhline(0, color='#777777', linewidth=0.6)
        ax.set_title(f'{name} | 90-day forecast', loc='left')
        format_axis(ax, recent.index[0], forecast['Date'].iloc[-1])
    save(fig, 'forecast_comparison.png',
         'Forecasts start after 31 Dec 2025. Model-based intervals are not clipped at zero; both panels use the same scale.')


def main():
    # Load and sort the daily passenger counts.
    data = pd.read_csv(BASE_DIR / 'passenger_volume_data.csv', parse_dates=['Date'])
    data = data.sort_values('Date').set_index('Date')
    passengers = pd.to_numeric(data['Numbers'], errors='raise')

    # Require complete daily data so seven rows always represent one week.
    if passengers.index.hasnans or passengers.index.has_duplicates:
        raise ValueError('Dates must be nonmissing and unique.')
    expected_dates = pd.date_range(passengers.index.min(), passengers.index.max(), freq='D')
    if not passengers.index.equals(expected_dates):
        raise ValueError('Data must contain one observation per day with no gaps.')
    if not np.isfinite(passengers).all() or (passengers < 0).any():
        raise ValueError('Passenger counts must be finite and nonnegative.')
    if len(passengers) < TEST_DAYS + 7:
        raise ValueError(f'At least {TEST_DAYS + 7} daily observations are required.')

    passengers = passengers.astype(float).asfreq('D')
    train = passengers.iloc[:-TEST_DAYS]
    test = passengers.iloc[-TEST_DAYS:]
    backtest = pd.DataFrame({'Date': test.index, 'actual': test.to_numpy()})
    future_dates = pd.date_range(passengers.index[-1] + pd.Timedelta(days=1),
                                 periods=FORECAST_DAYS, freq='D')
    forecast = pd.DataFrame({'Date': future_dates})
    metrics = []

    for name in ['ARIMA', 'SARIMA']:
        print(f'Fitting {name}...', flush=True)
        # Predict the entire holdout without fitting on any holdout observations.
        fitted = fit_model(train, name)
        predicted = np.asarray(fitted.forecast(TEST_DAYS)) * 1_000_000
        backtest[name] = predicted
        errors = test.to_numpy() - predicted
        metrics.append({'model': name, 'order': '(1,1,1)',
                        'seasonal_order': '(1,1,1,7)' if name == 'SARIMA' else 'none',
                        'test_days': TEST_DAYS,
                        'MAE': np.abs(errors).mean(),
                        'RMSE': np.sqrt(np.mean(errors ** 2))})

        # Refit on all observations for the future forecast and 95% intervals.
        full_fit = fit_model(passengers, name)
        future = full_fit.get_forecast(FORECAST_DAYS)
        forecast[name] = np.asarray(future.predicted_mean) * 1_000_000
        interval = np.asarray(future.conf_int(alpha=0.05)) * 1_000_000
        forecast[f'{name}_lower_95'] = interval[:, 0]
        forecast[f'{name}_upper_95'] = interval[:, 1]

    OUTPUT_DIR.mkdir(exist_ok=True)
    forecast.to_csv(OUTPUT_DIR / 'forecast.csv', index=False)
    backtest.to_csv(OUTPUT_DIR / 'backtest_predictions.csv', index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUTPUT_DIR / 'backtest_metrics.csv', index=False)
    plot_results(passengers, backtest, forecast, metrics)
    print(f'\nDaily observations: {len(passengers):,}')
    print(metrics[['model', 'MAE', 'RMSE']].round(0).to_string(index=False))
    print(f'Forecast: {future_dates[0].date()} through {future_dates[-1].date()}')
    print('Intervals are model-based and may include negative counts; values are not clipped.')
    print(f'Results saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
