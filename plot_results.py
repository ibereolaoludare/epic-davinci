"""
Generate charts from V7 walk-forward backtest results.

Outputs:
  chart_equity_curve.png       — Portfolio cumulative PnL over time
  chart_monthly_pnl.png        — Monthly PnL bar chart per pair + portfolio
  chart_regime_breakdown.png   — Win rate and PnL by regime per pair
  chart_drawdown.png           — Drawdown curve underwater plot
  chart_per_pair_equity.png    — Individual equity curves for each pair
  chart_trade_scatter.png      — Win/loss scatter by meta confidence
  chart_summary_dashboard.png  — All-in-one summary panel

Usage:
  python plot_results.py              # all pairs, all charts
  python plot_results.py USDJPY       # single pair
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
import sys
import warnings

warnings.filterwarnings('ignore')

from run_portfolio import PORTFOLIO_CONFIGS, run_pair_walk_forward, apply_portfolio_currency_matrix_cap

COLORS = {
    'USDJPY': '#e74c3c',
    'GBPJPY': '#3498db',
    'AUDUSD': '#f39c12',
    'XAUUSD': '#d4af37',
    'EURUSD': '#2980b9',
    'GBPUSD': '#27ae60',
    'Portfolio': '#9b59b6',
    'win': '#2ecc71',
    'loss': '#e74c3c',
}
REGIME_COLORS = {
    'risk_on': '#2ecc71',
    'risk_off': '#e74c3c',
    'intervention': '#f39c12',
    'range': '#3498db',
    'trend': '#9b59b6',
    'mixed': '#95a5a6',
}


def collect_trades(pairs):
    all_trades = []
    per_pair = {}
    for pair in pairs:
        cfg = PORTFOLIO_CONFIGS.get(pair)
        if cfg is None:
            continue
        print(f"  Running walk-forward for {pair}...")
        trades = run_pair_walk_forward(pair, cfg)
        per_pair[pair] = trades
        all_trades.extend(trades)
        
    all_trades.sort(key=lambda t: t['entry_time'])
    if len(pairs) > 1:
        all_trades = apply_portfolio_currency_matrix_cap(all_trades, max_usd_exposure=2, max_jpy_exposure=2)
        per_pair = {p: [t for t in all_trades if t['ticker'] == p] for p in pairs}
    return all_trades, per_pair


def plot_equity_curve(all_trades, savepath):
    fig, ax = plt.subplots(figsize=(14, 6))

    times = [pd.Timestamp(t['entry_time']) for t in all_trades]
    cum = np.cumsum([t['pips'] for t in all_trades])
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max

    ax.plot(times, cum, color=COLORS['Portfolio'], linewidth=2, label='Cumulative PnL')
    ax.fill_between(times, 0, cum, alpha=0.15, color=COLORS['Portfolio'])
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    final = cum[-1] if len(cum) > 0 else 0
    mdd = drawdown.min() if len(drawdown) > 0 else 0
    ax.set_title(f'Portfolio Equity Curve — {len(all_trades)} trades, '
                 f'+{final:.0f} pips, MaxDD: {mdd:.0f} pips',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative PnL (pips)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved {savepath}")


def plot_drawdown(all_trades, savepath):
    fig, ax = plt.subplots(figsize=(14, 4))

    times = [pd.Timestamp(t['entry_time']) for t in all_trades]
    cum = np.cumsum([t['pips'] for t in all_trades])
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max

    ax.fill_between(times, drawdown, 0, color=COLORS['loss'], alpha=0.4)
    ax.plot(times, drawdown, color=COLORS['loss'], linewidth=1.5)

    mdd = drawdown.min() if len(drawdown) > 0 else 0
    ax.set_title(f'Underwater Drawdown Curve — Max DD: {mdd:.0f} pips',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (pips)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved {savepath}")


def plot_monthly_pnl(all_trades, per_pair, savepath):
    pairs = list(per_pair.keys())

    fig, axes = plt.subplots(len(pairs) + 1, 1, figsize=(14, 4 * (len(pairs) + 1)),
                             sharex=True)

    # Collect all calendar months across the portfolio for universal x-axis alignment
    all_months = sorted(list(set([pd.Timestamp(t['entry_time']).strftime('%Y-%m') for t in all_trades])))

    # Per-pair monthly
    for i, pair in enumerate(pairs):
        ax = axes[i]
        trades = per_pair[pair]
        monthly = {}
        for t in trades:
            key = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
            monthly[key] = monthly.get(key, 0) + t['pips']

        vals = [monthly.get(m, 0.0) for m in all_months]
        colors = [COLORS['win'] if v > 0 else (COLORS['loss'] if v < 0 else '#bdc3c7') for v in vals]

        ax.bar(range(len(all_months)), vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_ylabel(f'{pair}\n(pips)', fontsize=10)
        ax.set_title(f'{pair} Monthly PnL', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

        # Annotate non-zero values
        for j, v in enumerate(vals):
            if abs(v) > 0.01:
                ax.text(j, v + (3 if v > 0 else -6), f'{v:.0f}',
                        ha='center', fontsize=7.5, fontweight='bold')

    # Portfolio monthly (bottom)
    ax = axes[-1]
    monthly = {}
    for t in all_trades:
        key = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly[key] = monthly.get(key, 0) + t['pips']

    vals = [monthly.get(m, 0.0) for m in all_months]
    colors = [COLORS['win'] if v > 0 else (COLORS['loss'] if v < 0 else '#bdc3c7') for v in vals]

    ax.bar(range(len(all_months)), vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_ylabel('Portfolio\n(pips)', fontsize=10)
    ax.set_title('Portfolio Combined Monthly PnL', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels(all_months, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    for j, v in enumerate(vals):
        if abs(v) > 0.01:
            ax.text(j, v + (5 if v > 0 else -8), f'{v:.0f}',
                    ha='center', fontsize=8, fontweight='bold')

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved {savepath}")


def plot_per_pair_equity(per_pair, savepath):
    fig, ax = plt.subplots(figsize=(14, 6))

    for pair, trades in per_pair.items():
        if not trades:
            continue
        times = [pd.Timestamp(t['entry_time']) for t in trades]
        cum = np.cumsum([t['pips'] for t in trades])
        ax.plot(times, cum, color=COLORS[pair], linewidth=2, label=pair)
        ax.fill_between(times, 0, cum, alpha=0.08, color=COLORS[pair])

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title('Per-Pair Equity Curves', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative PnL (pips)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved {savepath}")


def plot_regime_breakdown(per_pair, savepath):
    pairs = [p for p, t in per_pair.items() if t]
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Win rate by regime ---
    ax = axes[0]
    all_regimes = set()
    for pair in pairs:
        for t in per_pair[pair]:
            all_regimes.add(t['regime'])
    regimes = sorted(all_regimes)
    x = np.arange(len(regimes))
    width = 0.8 / n_pairs

    for i, pair in enumerate(pairs):
        trades = per_pair[pair]
        wrs = []
        for reg in regimes:
            reg_trades = [t for t in trades if t['regime'] == reg]
            if reg_trades:
                wrs.append(sum(t['win'] for t in reg_trades) / len(reg_trades) * 100)
            else:
                wrs.append(0)
        ax.bar(x + i * width - 0.4 + width/2, wrs, width, label=pair, color=COLORS[pair])

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=30, ha='right')
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Win Rate by Regime', fontsize=13, fontweight='bold')
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    # --- PnL by regime ---
    ax = axes[1]
    for i, pair in enumerate(pairs):
        trades = per_pair[pair]
        pnls = []
        for reg in regimes:
            reg_trades = [t for t in trades if t['regime'] == reg]
            pnls.append(sum(t['pips'] for t in reg_trades))
        colors = [COLORS['win'] if p > 0 else COLORS['loss'] for p in pnls]
        ax.barh(x + i * width - 0.4 + width/2, pnls, width, label=pair, color=COLORS[pair])

    ax.set_yticks(x)
    ax.set_yticklabels(regimes)
    ax.set_xlabel('PnL (pips)', fontsize=11)
    ax.set_title('PnL by Regime', fontsize=13, fontweight='bold')
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='x')

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved {savepath}")


def plot_trade_scatter(all_trades, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    pairs = ['USDJPY', 'EURUSD', 'GBPUSD']

    for i, pair in enumerate(pairs):
        ax = axes[i]
        trades = [t for t in all_trades if t['ticker'] == pair]
        if not trades:
            ax.set_title(f'{pair} (no trades)')
            continue

        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]

        ax.scatter([t['meta_conf'] for t in wins],
                   [t['pips'] for t in wins],
                   c=COLORS['win'], alpha=0.6, s=20, label=f'Win ({len(wins)})')
        ax.scatter([t['meta_conf'] for t in losses],
                   [t['pips'] for t in losses],
                   c=COLORS['loss'], alpha=0.6, s=20, label=f'Loss ({len(losses)})')

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(0.40, color='orange', linewidth=1, linestyle=':', label='Threshold 0.40')

        n = len(trades)
        wr = sum(t['win'] for t in trades) / n * 100
        pnl = sum(t['pips'] for t in trades)
        ax.set_title(f'{pair} — {n} trades, WR={wr:.1f}%, PnL={pnl:+.0f} pips',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Meta Confidence', fontsize=10)
        if i == 0:
            ax.set_ylabel('PnL per trade (pips)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Trade Outcome by Meta Model Confidence',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {savepath}")


def plot_summary_dashboard(all_trades, per_pair, savepath):
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Portfolio equity curve (top, spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    times = [pd.Timestamp(t['entry_time']) for t in all_trades]
    cum = np.cumsum([t['pips'] for t in all_trades])
    ax1.plot(times, cum, color=COLORS['Portfolio'], linewidth=2.5)
    ax1.fill_between(times, 0, cum, alpha=0.15, color=COLORS['Portfolio'])
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax1.set_title('Portfolio Equity Curve', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative PnL (pips)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax1.grid(True, alpha=0.3)

    # 2. Stats box (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    n = len(all_trades)
    wins = sum(t['win'] for t in all_trades)
    total_pnl = sum(t['pips'] for t in all_trades)
    wr = wins / n * 100 if n > 0 else 0
    gp = sum(t['pips'] for t in all_trades if t['pips'] > 0)
    gl = abs(sum(t['pips'] for t in all_trades if t['pips'] < 0))
    pf = gp / gl if gl > 0 else 0
    rmax = np.maximum.accumulate(cum)
    mdd = (rmax - cum).max() if len(cum) > 0 else 0
    exp = total_pnl / n if n > 0 else 0
    be_wr = np.mean([t.get('sl', PORTFOLIO_CONFIGS[t['ticker']].get('sl_mult', 1.0)) /
                     (t.get('tp', PORTFOLIO_CONFIGS[t['ticker']].get('tp_mult', 1.0)) +
                      t.get('sl', PORTFOLIO_CONFIGS[t['ticker']].get('sl_mult', 1.0)))
                     for t in all_trades])
    p_val = 1 - stats.binom.cdf(wins - 1, n, be_wr) if n > 0 else 1

    stats_text = (
        f"PORTFOLIO SUMMARY\n\n"
        f"  Total Trades:  {n}\n"
        f"  Win Rate:      {wr:.1f}%\n"
        f"  Break-Even WR: {be_wr*100:.1f}%\n"
        f"  Net PnL:       {total_pnl:+.0f} pips\n"
        f"  Profit Factor: {pf:.2f}\n"
        f"  Max Drawdown:  {mdd:.0f} pips\n"
        f"  Expectancy:    {exp:.2f} pips/trade\n"
        f"  p-value:       {p_val:.6f}\n"
        f"  Significance:  {'*** SIGNIFICANT' if p_val < 0.05 else 'not sig.'}\n"
    )
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
             fontsize=12, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    # 3. Monthly PnL (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    monthly = {}
    for t in all_trades:
        key = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly[key] = monthly.get(key, 0) + t['pips']
    months = sorted(monthly.keys())
    vals = [monthly[m] for m in months]
    colors = [COLORS['win'] if v > 0 else COLORS['loss'] for v in vals]
    ax3.bar(range(len(vals)), vals, color=colors, edgecolor='white')
    ax3.axhline(0, color='gray', linewidth=0.8)
    ax3.set_xticks(range(len(months)))
    ax3.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax3.set_title('Monthly Portfolio PnL', fontsize=13, fontweight='bold')
    ax3.set_ylabel('PnL (pips)')
    ax3.grid(True, alpha=0.2, axis='y')
    for j, v in enumerate(vals):
        ax3.text(j, v + (5 if v > 0 else -8), f'{v:.0f}',
                 ha='center', fontsize=7, fontweight='bold')

    # 4. Per-pair pie (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    pair_pnls = {}
    for pair, trades in per_pair.items():
        if trades:
            pair_pnls[pair] = sum(t['pips'] for t in trades)
    if pair_pnls:
        labels = list(pair_pnls.keys())
        sizes = [abs(v) for v in pair_pnls.values()]
        colors_pie = [COLORS[p] for p in labels]
        explode = [0.05] * len(labels)
        wedges, texts, autotexts = ax4.pie(
            sizes, labels=labels, colors=colors_pie, explode=explode,
            autopct=lambda pct: f'{pct:.0f}%', startangle=90,
            textprops={'fontsize': 10})
        ax4.set_title('PnL Contribution by Pair', fontsize=13, fontweight='bold')

    # 5. Drawdown (bottom left + middle)
    ax5 = fig.add_subplot(gs[2, :2])
    dd = cum - rmax
    ax5.fill_between(times, dd, 0, color=COLORS['loss'], alpha=0.4)
    ax5.plot(times, dd, color=COLORS['loss'], linewidth=1)
    ax5.set_title(f'Drawdown — Max: {mdd:.0f} pips', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Drawdown (pips)')
    ax5.set_xlabel('Date')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax5.grid(True, alpha=0.3)

    # 6. Per-pair stats table (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    table_data = []
    for pair in per_pair.keys():
        trades = per_pair.get(pair, [])
        if not trades:
            continue
        n_p = len(trades)
        wr_p = sum(t['win'] for t in trades) / n_p * 100
        pnl_p = sum(t['pips'] for t in trades)
        cfg = PORTFOLIO_CONFIGS.get(pair, {})
        be = np.mean([t.get('sl', cfg.get('sl_mult', 1.0)) / (t.get('tp', cfg.get('tp_mult', 1.0)) + t.get('sl', cfg.get('sl_mult', 1.0))) for t in trades]) * 100
        p = 1 - stats.binom.cdf(sum(t['win'] for t in trades) - 1, n_p, be / 100)
        rr_str = f"{cfg.get('tp_mult', 1.0)}:1" if 'modes' not in cfg else "Multi"
        table_data.append([pair, n_p, f'{wr_p:.1f}%', f'{pnl_p:+.0f}',
                           rr_str, f'{p:.4f}'])

    table = ax6.table(cellText=table_data,
                      colLabels=['Pair', 'Trades', 'WR', 'PnL', 'RR', 'p-val'],
                      cellLoc='center', loc='center',
                      colColours=['#bdc3c7'] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax6.set_title('Per-Pair Results', fontsize=13, fontweight='bold', pad=20)

    fig.suptitle('Epic DaVinci V7 — Portfolio Backtest Results',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {savepath}")


def main():
    pairs = sys.argv[1:] if len(sys.argv) > 1 else ['USDJPY', 'GBPJPY', 'AUDUSD', 'XAUUSD']
    valid = [p for p in pairs if p in PORTFOLIO_CONFIGS]
    if not valid:
        print(f"No valid pairs. Supported: {list(PORTFOLIO_CONFIGS.keys())}")
        return

    print(f"Collecting walk-forward trades for {valid}...")
    all_trades, per_pair = collect_trades(valid)

    if not all_trades:
        print("No trades generated. Check data files exist.")
        return

    print(f"\nGenerating charts ({len(all_trades)} total trades)...")

    if len(valid) > 1:
        plot_equity_curve(all_trades, 'chart_equity_curve.png')
        plot_drawdown(all_trades, 'chart_drawdown.png')
        plot_monthly_pnl(all_trades, per_pair, 'chart_monthly_pnl.png')
        plot_per_pair_equity(per_pair, 'chart_per_pair_equity.png')
        plot_regime_breakdown(per_pair, 'chart_regime_breakdown.png')
        plot_trade_scatter(all_trades, 'chart_trade_scatter.png')
        plot_summary_dashboard(all_trades, per_pair, 'chart_summary_dashboard.png')
    else:
        pair = valid[0]
        plot_equity_curve(all_trades, f'chart_equity_curve_{pair.lower()}.png')
        plot_drawdown(all_trades, f'chart_drawdown_{pair.lower()}.png')
        plot_monthly_pnl(all_trades, per_pair, f'chart_monthly_pnl_{pair.lower()}.png')
        plot_regime_breakdown(per_pair, f'chart_regime_breakdown_{pair.lower()}.png')
        plot_trade_scatter(all_trades, f'chart_trade_scatter_{pair.lower()}.png')
        plot_summary_dashboard(all_trades, per_pair,
                               f'chart_summary_dashboard_{pair.lower()}.png')

    print("\nDone! All charts saved.")


if __name__ == '__main__':
    main()
