# scripts/test_env.py
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.supplysim_env import SupplySimEnv

env = SupplySimEnv(seed=0, T=20, gamma=0.8, log_kpis=False)
obs = env.reset()

done = False
while not done:
    # Example action: reroute one buyer+product to a different supplier (you’ll choose valid tuples)
    obs, reward, done, info = env.step(action=None)
    k = info["kpis"]
    print(
        f"t={k['t']:02d} txns={k['transactions']:3d} open={k['open_orders']:4d} "
        f"backlog_u={k['consumer_backlog_units']:9.1f} fill_cum={k['consumer_cumulative_fill_rate']:.3f} "
        f"shock={k['shock_exposure']:.2f} reward={reward:.1f}"
    )
