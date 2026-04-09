import numpy as np
import pandas as pd
from pathlib import Path
from strategy7.factors.labeling import add_labels
from strategy7.mining.custom import CustomFactorSpec
from strategy7.mining.runner import FactorMiningConfig, run_factor_mining

rng = np.random.default_rng(7)
codes = [f'{i:06d}' for i in range(1, 41)]
days = pd.bdate_range('2024-01-02', periods=10)
bars = []
for d in days:
    start = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=45)
    bars.extend([start + pd.Timedelta(minutes=15*k) for k in range(16)])

data = []
for c in codes:
    base = 10 + rng.normal(0, 0.5)
    rets = rng.normal(0.0002, 0.004, size=len(bars))
    px = base * np.cumprod(1.0 + rets)
    for t, p in zip(bars, px):
        data.append((t, pd.Timestamp(t).normalize(), c, p))

panel = pd.DataFrame(data, columns=['datetime','date','code','close'])
panel['open'] = panel['close'] * (1 - 0.0005)
panel['high'] = panel[['open','close']].max(axis=1) * 1.001
panel['low'] = panel[['open','close']].min(axis=1) * 0.999
panel['volume'] = 1_000 + rng.integers(0, 500, size=len(panel))
panel['amount'] = panel['close'] * panel['volume']
panel['barra_size_proxy'] = np.log(panel['amount'] + 1.0)
panel['industry_bucket'] = np.where(panel['code'].astype(int) % 2 == 0, 'ind_a', 'ind_b')

panel = add_labels(
    panel=panel,
    horizon=5,
    execution_scheme='vwap30_vwap30',
    price_table_daily=pd.DataFrame(),
    factor_freq='15min',
)

cfg = FactorMiningConfig(
    framework='custom',
    factor_freq='15min',
    horizon=5,
    train_start=pd.Timestamp('2024-01-02'),
    train_end=pd.Timestamp('2024-01-05'),
    valid_start=pd.Timestamp('2024-01-08'),
    valid_end=pd.Timestamp('2024-01-12'),
    population_size=8,
    generations=1,
    elite_size=2,
    top_n=3,
    min_cross_section=30,
    factor_store_root=str(Path('d:/PythonProject/Quant/TradeSystem/Strategy7/outputs/_tmp_freq_guard').resolve()),
    catalog_path=str(Path('d:/PythonProject/Quant/TradeSystem/Strategy7/outputs/_tmp_freq_guard/catalog.json').resolve()),
    save_format='csv',
)
summary = run_factor_mining(
    cfg=cfg,
    panel_with_label=panel,
    minute_df=None,
    custom_specs=[CustomFactorSpec(name='fac_close', expression='close', freq='15min')],
)
print('framework', summary.get('framework'))
print('freq', summary.get('factor_freq'))
print('selected_count', summary.get('selected_count'))
print('factor_table', summary.get('factor_table_path'))
