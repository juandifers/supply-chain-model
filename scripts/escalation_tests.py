"""
Escalation paths to improve shock sensitivity:
1. Slower recovery (recovery_rate closer to 1.0) — shocks persist longer
2. Higher demand (check if demand is scalable)
3. Different supply levels near the transition zone
"""
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.calibrated_scenario import create_calibrated_env


def run_episode(default_supply, shock_prob=0.0, shock_magnitude=0.0,
                firm_shock_fraction=1.0, warmup_steps=10, T=60, seed=42,
                recovery_rate=1.25, init_inv=0):
    env, obs, shock_log = create_calibrated_env(
        seed=seed, default_supply=default_supply,
        shock_magnitude=shock_magnitude, shock_prob=shock_prob,
        firm_shock_fraction=firm_shock_fraction, warmup_steps=warmup_steps,
        recovery_rate=recovery_rate, T=T, init_inv=init_inv,
    )
    for t in range(T):
        obs, _, done, info = env.step({"reroute": [], "supply_multiplier": {}})
        if done:
            break
    kpi_df = env.get_kpi_history()
    return {
        "fill_rate": kpi_df["consumer_cumulative_fill_rate"].iloc[-1],
        "backlog_auc": kpi_df["consumer_backlog_units"].sum(),
        "num_shocks": len(shock_log),
    }


def escalation1_slow_recovery():
    """Test slower recovery rates to make shocks persist longer."""
    print("=" * 70)
    print("ESCALATION 1: Slower recovery (ds=500K, sp=0.15, sf=0.3, fsf=0.5)")
    print("=" * 70)
    print(f"{'recovery':>10} {'no-shock AUC':>14} {'shocked AUC':>14} {'delta%':>8} {'fill_ns':>8} {'fill_sh':>8}")
    print("-" * 66)

    for rr in [1.25, 1.10, 1.05, 1.02, 1.01]:
        r0 = run_episode(500_000, shock_prob=0.0, recovery_rate=rr)
        r1 = run_episode(500_000, shock_prob=0.15, shock_magnitude=0.7,
                        firm_shock_fraction=0.5, recovery_rate=rr)
        delta = (r1["backlog_auc"] - r0["backlog_auc"]) / max(r0["backlog_auc"], 1) * 100
        print(f"{rr:>10.2f} {r0['backlog_auc']:>14,.0f} {r1['backlog_auc']:>14,.0f} {delta:>8.1f}% {r0['fill_rate']:>8.4f} {r1['fill_rate']:>8.4f}")


def escalation1b_slow_recovery_400k():
    """Same but at ds=400K."""
    print("\n" + "=" * 70)
    print("ESCALATION 1b: Slower recovery (ds=400K, sp=0.15, sf=0.3, fsf=0.5)")
    print("=" * 70)
    print(f"{'recovery':>10} {'no-shock AUC':>14} {'shocked AUC':>14} {'delta%':>8} {'fill_ns':>8} {'fill_sh':>8}")
    print("-" * 66)

    for rr in [1.25, 1.10, 1.05, 1.02, 1.01]:
        r0 = run_episode(400_000, shock_prob=0.0, recovery_rate=rr)
        r1 = run_episode(400_000, shock_prob=0.15, shock_magnitude=0.7,
                        firm_shock_fraction=0.5, recovery_rate=rr)
        delta = (r1["backlog_auc"] - r0["backlog_auc"]) / max(r0["backlog_auc"], 1) * 100
        print(f"{rr:>10.2f} {r0['backlog_auc']:>14,.0f} {r1['backlog_auc']:>14,.0f} {delta:>8.1f}% {r0['fill_rate']:>8.4f} {r1['fill_rate']:>8.4f}")


def escalation2_combined():
    """Combine slow recovery with shock_prob sweep to find best config."""
    print("\n" + "=" * 70)
    print("ESCALATION 2: Best configs (rr=1.05, various ds)")
    print("=" * 70)

    for ds in [400_000, 450_000, 500_000]:
        print(f"\n--- ds={ds:,}, recovery_rate=1.05 ---")
        print(f"{'shock_prob':>10} {'sf':>5} {'fill_rate':>10} {'backlog_auc':>12} {'delta%':>8}")
        print("-" * 50)

        r0 = run_episode(ds, shock_prob=0.0, recovery_rate=1.05)
        baseline = r0["backlog_auc"]
        print(f"{'0.00':>10} {'1.0':>5} {r0['fill_rate']:>10.4f} {baseline:>12,.0f}  baseline")

        for sp in [0.05, 0.10, 0.15, 0.20]:
            r = run_episode(ds, shock_prob=sp, shock_magnitude=0.7,
                          firm_shock_fraction=0.5, recovery_rate=1.05)
            delta = (r["backlog_auc"] - baseline) / max(baseline, 1) * 100
            print(f"{sp:>10.2f} {'0.3':>5} {r['fill_rate']:>10.4f} {r['backlog_auc']:>12,.0f}  ({delta:+.1f}%)")

        # Severity sweep
        for sm in [0.1, 0.3, 0.5, 0.7]:
            r = run_episode(ds, shock_prob=0.15, shock_magnitude=sm,
                          firm_shock_fraction=0.5, recovery_rate=1.05)
            delta = (r["backlog_auc"] - baseline) / max(baseline, 1) * 100
            print(f"{'0.15':>10} {sf:>5.1f} {r['fill_rate']:>10.4f} {r['backlog_auc']:>12,.0f}  ({delta:+.1f}%)")


def escalation3_multi_seed_best():
    """Multi-seed at best config to verify consistency."""
    print("\n" + "=" * 70)
    print("ESCALATION 3: Multi-seed at best config")
    print("=" * 70)

    # Test the most promising configurations
    configs = [
        {"ds": 450_000, "rr": 1.05, "label": "ds=450K rr=1.05"},
        {"ds": 500_000, "rr": 1.05, "label": "ds=500K rr=1.05"},
        {"ds": 500_000, "rr": 1.02, "label": "ds=500K rr=1.02"},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['label']}, sp=0.15, sf=0.3, fsf=0.5 ---")
        print(f"{'seed':>5} {'no-shock':>14} {'shocked':>14} {'delta%':>8}")
        print("-" * 45)
        deltas = []
        for seed in range(8):
            r0 = run_episode(cfg["ds"], shock_prob=0.0, recovery_rate=cfg["rr"], seed=seed)
            r1 = run_episode(cfg["ds"], shock_prob=0.15, shock_magnitude=0.7,
                           firm_shock_fraction=0.5, recovery_rate=cfg["rr"], seed=seed)
            d = (r1["backlog_auc"] - r0["backlog_auc"]) / max(r0["backlog_auc"], 1) * 100
            deltas.append(d)
            print(f"{seed:>5} {r0['backlog_auc']:>14,.0f} {r1['backlog_auc']:>14,.0f} {d:>8.1f}%")
        print(f"{'mean':>5} {'':>14} {'':>14} {np.mean(deltas):>8.1f}%")
        print(f"{'min':>5} {'':>14} {'':>14} {np.min(deltas):>8.1f}%")


if __name__ == "__main__":
    escalation1_slow_recovery()
    escalation1b_slow_recovery_400k()
    escalation2_combined()
    escalation3_multi_seed_best()
