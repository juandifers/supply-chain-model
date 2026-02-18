import argparse
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache_supplysim"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.supplysim_env import SupplySimEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run SupplySim episode and plot timeline KPIs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--T", type=int, default=50, help="Number of timesteps.")
    parser.add_argument("--gamma", type=float, default=0.8, help="Same-supplier probability.")
    parser.add_argument("--shock-prob", type=float, default=0.001, help="Per exogenous product shock probability.")
    parser.add_argument("--init-inv", type=float, default=0.0, help="Initial inventory for all firm-product pairs.")
    parser.add_argument("--init-supply", type=float, default=100.0, help="Initial exogenous supply before schedules.")
    parser.add_argument("--init-demand", type=float, default=1.0, help="Initial demand for consumer products.")
    parser.add_argument("--stride", type=int, default=1, help="Print every N timesteps in ASCII timeline.")
    parser.add_argument("--bar-width", type=int, default=28, help="ASCII bar width.")
    parser.add_argument("--no-ascii", action="store_true", help="Disable ASCII timeline printout.")
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG output.")
    parser.add_argument("--out", type=str, default=None, help="PNG output path.")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional KPI CSV output path.")
    return parser.parse_args()


def _ascii_bar(value, max_value, width):
    if max_value <= 0 or width <= 0:
        return ""
    filled = int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def print_ascii_timeline(history_df, stride=1, bar_width=28):
    if len(history_df) == 0:
        print("No KPI data collected.")
        return

    stride = max(1, int(stride))
    max_open_orders = float(history_df["open_orders"].max())
    print("\nASCII timeline")
    print("t   txns  open_ord  backlog_u  fill_cum  shock  reroutes  open_orders_bar")
    for row in history_df.itertuples(index=False):
        if row.t % stride != 0 and row.t != history_df.iloc[-1]["t"]:
            continue
        bar = _ascii_bar(float(row.open_orders), max_open_orders, bar_width)
        print(
            f"{int(row.t):>2}  {int(row.transactions):>5}  {int(row.open_orders):>8}  "
            f"{float(row.consumer_backlog_units):>9.1f}  {float(row.consumer_cumulative_fill_rate):>8.3f}  "
            f"{float(row.shock_exposure):>5.2f}  {int(row.reroutes_applied):>8}  |{bar}|"
        )

    peak_idx = int(history_df["open_orders"].idxmax())
    peak_row = history_df.iloc[peak_idx]
    print("\nSummary")
    print(
        f"Peak open orders at t={int(peak_row['t'])}: {int(peak_row['open_orders'])} "
        f"(backlog_u={float(peak_row['consumer_backlog_units']):.1f})"
    )
    print(
        f"Final cumulative fill rate: {float(history_df.iloc[-1]['consumer_cumulative_fill_rate']):.3f} | "
        f"Final backlog_u: {float(history_df.iloc[-1]['consumer_backlog_units']):.1f}"
    )


def plot_timeline(history_df, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    ax0 = axes[0]
    ax0.plot(history_df["t"], history_df["transactions"], color="#1f77b4", linewidth=2, label="transactions")
    ax0.set_ylabel("transactions")
    ax0.grid(alpha=0.25)
    ax0r = ax0.twinx()
    ax0r.plot(history_df["t"], history_df["open_orders"], color="#ff7f0e", linewidth=2, label="open_orders")
    ax0r.set_ylabel("open_orders")
    ax0.set_title("Throughput vs Queue Pressure")
    h0, l0 = ax0.get_legend_handles_labels()
    h0r, l0r = ax0r.get_legend_handles_labels()
    ax0.legend(h0 + h0r, l0 + l0r, loc="upper left")

    ax1 = axes[1]
    ax1.plot(
        history_df["t"],
        history_df["consumer_demand_added_units"],
        color="#2ca02c",
        linewidth=1.8,
        label="consumer_demand_added_units",
    )
    ax1.plot(
        history_df["t"],
        history_df["consumer_fulfilled_units"],
        color="#17becf",
        linewidth=1.8,
        label="consumer_fulfilled_units",
    )
    ax1r = ax1.twinx()
    ax1r.plot(
        history_df["t"],
        history_df["consumer_backlog_units"],
        color="#d62728",
        linewidth=2.0,
        label="consumer_backlog_units",
    )
    ax1.set_ylabel("demand/fulfilled units")
    ax1r.set_ylabel("backlog units")
    ax1.grid(alpha=0.25)
    ax1.set_title("Consumer Service Dynamics")
    h1, l1 = ax1.get_legend_handles_labels()
    h1r, l1r = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h1r, l1 + l1r, loc="upper left")

    ax2 = axes[2]
    ax2.plot(
        history_df["t"],
        history_df["consumer_cumulative_fill_rate"],
        color="#9467bd",
        linewidth=2.2,
        label="consumer_cumulative_fill_rate",
    )
    ax2.plot(
        history_df["t"],
        history_df["shock_exposure"],
        color="#8c564b",
        linewidth=2.0,
        label="shock_exposure",
    )
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("ratio")
    ax2.set_xlabel("timestep")
    ax2.grid(alpha=0.25)
    ax2.set_title("Resilience Signals")

    if float(history_df["reroutes_applied"].sum()) > 0:
        ax2r = ax2.twinx()
        ax2r.bar(
            history_df["t"],
            history_df["reroutes_applied"],
            width=0.8,
            alpha=0.25,
            color="#7f7f7f",
            label="reroutes_applied",
        )
        ax2r.set_ylabel("reroutes")
        h2r, l2r = ax2r.get_legend_handles_labels()
    else:
        h2r, l2r = [], []

    if float(history_df["shock_exposure"].max()) > 0:
        shock_t = int(history_df.loc[history_df["shock_exposure"].idxmax(), "t"])
        for ax in axes:
            ax.axvline(shock_t, linestyle="--", linewidth=1.2, color="black", alpha=0.4)

    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2 + h2r, l2 + l2r, loc="upper left")

    fig.suptitle("SupplySim Timeline KPIs", fontsize=13)
    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_episode(args):
    env = SupplySimEnv(seed=args.seed, T=args.T, gamma=args.gamma, log_kpis=False)
    env.reset(
        init_inv=args.init_inv,
        init_supply=args.init_supply,
        init_demand=args.init_demand,
        use_demand_schedule=True,
        use_exog_schedule=True,
        shock_prob=args.shock_prob,
    )

    done = False
    while not done:
        _, _, done, _ = env.step(action=None, debug=False, log_kpis=False)
    return env.get_kpi_history()


def main():
    args = parse_args()
    history_df = run_episode(args)
    if len(history_df) == 0:
        print("No KPI rows produced; nothing to plot.")
        return

    if args.csv_out is not None:
        csv_path = args.csv_out
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        history_df.to_csv(csv_path, index=False)
        print(f"Saved KPI CSV to: {csv_path}")

    if not args.no_ascii:
        print_ascii_timeline(history_df, stride=args.stride, bar_width=args.bar_width)

    if not args.no_plot:
        out_path = args.out
        if out_path is None:
            out_path = os.path.join("artifacts", f"supplysim_timeline_seed{args.seed}.png")
        plot_timeline(history_df, out_path)
        print(f"Saved timeline plot to: {out_path}")


if __name__ == "__main__":
    main()
