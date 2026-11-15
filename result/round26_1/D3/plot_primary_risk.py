"""Plot all prespecified Both-Add seeds, without selecting a historical winner."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
returns = pd.read_csv(R/'daily_returns.csv.gz', parse_dates=['trade_date']).set_index('trade_date')
metrics = pd.read_csv(R/'all_results.csv').set_index('experiment_id')
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
checks = []
for column, arm in enumerate(['AG_RAW', 'AG_RES']):
    names = ['AG_A0'] + [f'{arm}_{seed}_A3_BOTH_ADD' for seed in ['17', '42', '73', 'EQ3']]
    for name in names:
        series = returns[name]
        assert series.notna().all() and (series > -1).all()
        wealth = pd.concat([pd.Series([1.0], index=pd.to_datetime(['2023-01-03'])), (1+series).cumprod()])
        assert wealth.index.is_unique and wealth.index.is_monotonic_increasing
        expected = float(metrics.loc[name, 'TerminalNAV'])/300000
        assert np.isclose(wealth.iloc[-1], expected, rtol=1e-10, atol=1e-10), name
        drawdown = wealth/wealth.cummax()-1
        assert np.isclose(drawdown.min(), float(metrics.loc[name, 'MaxDD']), atol=1e-10), name
        label = 'Matched no-factor' if name == 'AG_A0' else name.split('_')[2]
        options = {'color':'black', 'linewidth':2} if name == 'AG_A0' else {'linewidth':1.2}
        axes[0,column].plot(wealth.index, wealth, label=label, **options)
        axes[1,column].plot(drawdown.index, drawdown*100, label=label, **options)
        checks.append({'experiment_id':name, 'terminal_wealth':float(wealth.iloc[-1]), 'max_drawdown':float(drawdown.min()), 'matches_actual_result_table':True})
    axes[0,column].set_title(arm+' — all frozen Both-Add seeds')
    axes[0,column].set_yscale('log')
    axes[0,column].set_ylabel('Wealth / initial cash (log scale)')
    axes[1,column].set_ylabel('Drawdown (%)')
    for row in [0,1]:
        axes[row,column].grid(alpha=.25)
        axes[row,column].legend(fontsize=8)
fig.suptitle('Development walk-forward; signals end 2026-07-01, valuation ends 2026-07-24')
fig.tight_layout()
fig.savefig(R/'primary_NAV_drawdown.png', dpi=170)
fig.savefig(R/'primary_NAV_drawdown.svg')
svg = R/'primary_NAV_drawdown.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
inputs = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'daily_returns.csv.gz',R/'all_results.csv']}
(R/'primary_plot_audit.json').write_text(json.dumps({'inputs_sha256':inputs, 'scope':'S0; all frozen Both-Add seeds and equal3, no winner selection; individual and Remove/Replace results remain in full tables', 'checks':checks},indent=2))
