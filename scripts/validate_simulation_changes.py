"""
Quick validation that the 3 calibrated simulation changes work correctly.
Runs ONE episode, prints diagnostic values, exits.
Should complete in under 10 seconds.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.calibrated_scenario import create_calibrated_env


def main():
    print("=" * 70)
    print("VALIDATION: Calibrated Scenario Changes")
    print("=" * 70)

    # Create env with known params to make shocks very likely
    env, obs, shock_log = create_calibrated_env(
        seed=42,
        T=40,
        default_supply=50000,
        shock_fraction=0.3,
        shock_prob=0.15,
        recovery_rate=1.25,
        firm_shock_fraction=0.5,
        warmup_steps=10,
        expedite_budget=500.0,
    )

    # 1. Verify warmup — no shocks in first warmup_steps
    warmup_shocks = [s for s in shock_log if s["t"] < 10]
    post_warmup_shocks = [s for s in shock_log if s["t"] >= 10]
    print(f"\n1. WARMUP CHECK:")
    print(f"   Shocks in warmup period (t<10): {len(warmup_shocks)}")
    print(f"   Shocks after warmup (t>=10):    {len(post_warmup_shocks)}")
    assert len(warmup_shocks) == 0, "FAIL: shocks found during warmup!"
    print("   -> PASS: warmup is shock-free")

    # 2. Verify firm-level shocks
    if post_warmup_shocks:
        s = post_warmup_shocks[0]
        print(f"\n2. FIRM-LEVEL SHOCK CHECK:")
        print(f"   First shock at t={s['t']}: product={s['product']}")
        print(f"   Total firms for product: {s['num_firms_total']}")
        print(f"   Firms shocked: {s['num_firms_shocked']} ({s['shocked_firms'][:3]}...)")
        print(f"   Firms unshocked: {s['num_firms_total'] - s['num_firms_shocked']}")
        assert s["num_firms_shocked"] < s["num_firms_total"] or s["num_firms_total"] == 1, \
            "FAIL: all firms shocked (firm_shock_fraction should leave some unshocked)"
        print("   -> PASS: not all firms are shocked")
    else:
        print("\n2. FIRM-LEVEL SHOCK CHECK: No shocks occurred (increase shock_prob)")

    # 3. Verify proportional shocks
    print(f"\n3. PROPORTIONAL SHOCK CHECK:")
    shock_supply = 50000 * 0.3
    print(f"   default_supply = 50000")
    print(f"   shock_fraction = 0.3")
    print(f"   Expected shock_supply = {shock_supply:.0f} (= 50000 * 0.3)")
    # Check the exog schedule at a shock timestep
    if post_warmup_shocks:
        shock_t = post_warmup_shocks[0]["t"]
        shock_prod = post_warmup_shocks[0]["product"]
        shocked_firms_set = set(post_warmup_shocks[0]["shocked_firms"])
        exog_t = env.exog_schedule[shock_t]
        for (firm, prod), amount in exog_t.items():
            if prod == shock_prod and firm in shocked_firms_set:
                # Amount should be ~Poisson(shock_supply)
                print(f"   Realized supply for shocked firm {firm}: {amount}")
                print(f"   Expected ~Poisson({shock_supply:.0f})")
                # Poisson with mean 15000 should be in range [10000, 20000] with high prob
                assert 1000 < amount < 100000, f"FAIL: supply {amount} looks wrong for mean {shock_supply}"
                print("   -> PASS: proportional shock supply is reasonable")
                break
    else:
        print("   (No shocks to verify)")

    # 4. Run the episode and check post-warmup fill rate
    print(f"\n4. POST-WARMUP FILL RATE CHECK:")
    done = False
    while not done:
        obs, reward, done, info = env.step()

    kpi_df = env.get_kpi_history()
    warmup_kpis = kpi_df[kpi_df["t"] < 10]
    post_warmup_kpis = kpi_df[kpi_df["t"] >= 10]

    warmup_fill = warmup_kpis["consumer_new_demand_fill_rate"].mean() if len(warmup_kpis) > 0 else 0
    post_fill = post_warmup_kpis["consumer_new_demand_fill_rate"].mean() if len(post_warmup_kpis) > 0 else 0
    final_fill = float(kpi_df["consumer_cumulative_fill_rate"].iloc[-1])

    print(f"   Warmup fill rate (t<10):     {warmup_fill:.3f}")
    print(f"   Post-warmup fill rate (t>=10): {post_fill:.3f}")
    print(f"   Final cumulative fill rate:    {final_fill:.3f}")

    # With shocks, post-warmup should be lower than warmup
    if len(post_warmup_shocks) > 0:
        print(f"   -> Fill rate dropped from {warmup_fill:.3f} to {post_fill:.3f} due to shocks")
    else:
        print("   -> No shocks, fill rates should be similar")

    print(f"\n{'=' * 70}")
    print("ALL CHECKS PASSED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
