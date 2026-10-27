import numpy as np
import matplotlib.pyplot as plt

from black_scholes import (
    run_hedging_simulation,
    run_event_driven_hedging_simulation,
)


# ============================================================
# EXPERIMENT 7: EVENT-DRIVEN HEDGING ACROSS MULTIPLE SCENARIOS
# ============================================================

TRANSACTION_COST_RATE = 0.001      # 0.10% = 10 bps
N_PATHS = 2000
SEED = 42

DELTA_THRESHOLDS = [
    0.01,
    0.02,
    0.05,
    0.10,
]

SCENARIOS = {
    "ATM 1Y": {
        "S": 100,
        "K": 100,
        "T": 1.00,
        "pricing_sigma": 0.20,
        "realised_sigma": 0.20,
    },
    "ITM 1Y": {
        "S": 100,
        "K": 90,
        "T": 1.00,
        "pricing_sigma": 0.20,
        "realised_sigma": 0.20,
    },
    "OTM 1Y": {
        "S": 100,
        "K": 110,
        "T": 1.00,
        "pricing_sigma": 0.20,
        "realised_sigma": 0.20,
    },
    "ATM 3M": {
        "S": 100,
        "K": 100,
        "T": 0.25,
        "pricing_sigma": 0.20,
        "realised_sigma": 0.20,
    },
    "ATM 1Y High Vol": {
        "S": 100,
        "K": 100,
        "T": 1.00,
        "pricing_sigma": 0.40,
        "realised_sigma": 0.40,
    },
}


def summarize(pnls, costs, trades):
    return {
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "mean_cost": np.mean(costs),
        "mean_trades": np.mean(trades),
        "p5": np.percentile(pnls, 5),
        "p95": np.percentile(pnls, 95),
    }


results = {}


for scenario_name, params in SCENARIOS.items():

    print("\n" + "=" * 72)
    print(scenario_name)
    print("=" * 72)

    daily_steps = max(1, round(252 * params["T"]))

    # --------------------------------------------------------
    # DAILY FIXED-FREQUENCY BENCHMARK
    # --------------------------------------------------------

    daily_pnls, daily_costs = run_hedging_simulation(
        S=params["S"],
        K=params["K"],
        T=params["T"],
        r=0.05,
        pricing_sigma=params["pricing_sigma"],
        realised_sigma=params["realised_sigma"],
        steps=daily_steps,
        n_paths=N_PATHS,
        seed=SEED,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )

    daily_summary = {
        "mean_pnl": np.mean(daily_pnls),
        "std_pnl": np.std(daily_pnls),
        "mean_cost": np.mean(daily_costs),
        "mean_trades": daily_steps - 1,
        "p5": np.percentile(daily_pnls, 5),
        "p95": np.percentile(daily_pnls, 95),
    }

    results[scenario_name] = {
        "Daily": daily_summary
    }

    print("\nDaily fixed hedge")
    print(f"Mean P&L: £{daily_summary['mean_pnl']:.4f}")
    print(f"Standard deviation: £{daily_summary['std_pnl']:.4f}")
    print(f"Mean transaction cost: £{daily_summary['mean_cost']:.4f}")
    print(f"Approx. scheduled rebalances: {daily_summary['mean_trades']:.0f}")
    print(f"5th percentile: £{daily_summary['p5']:.4f}")
    print(f"95th percentile: £{daily_summary['p95']:.4f}")

    # --------------------------------------------------------
    # EVENT-DRIVEN THRESHOLDS
    # --------------------------------------------------------

    for threshold in DELTA_THRESHOLDS:

        pnls, costs, trades = run_event_driven_hedging_simulation(
            S=params["S"],
            K=params["K"],
            T=params["T"],
            r=0.05,
            pricing_sigma=params["pricing_sigma"],
            realised_sigma=params["realised_sigma"],
            delta_threshold=threshold,
            monitoring_steps=daily_steps,
            n_paths=N_PATHS,
            seed=SEED,
            transaction_cost_rate=TRANSACTION_COST_RATE,
        )

        summary = summarize(
            pnls,
            costs,
            trades,
        )

        label = f"Δ threshold {threshold:.2f}"
        results[scenario_name][label] = summary

        print(f"\n{label}")
        print(f"Mean P&L: £{summary['mean_pnl']:.4f}")
        print(f"Standard deviation: £{summary['std_pnl']:.4f}")
        print(f"Mean transaction cost: £{summary['mean_cost']:.4f}")
        print(f"Mean number of rebalances: {summary['mean_trades']:.2f}")
        print(f"5th percentile: £{summary['p5']:.4f}")
        print(f"95th percentile: £{summary['p95']:.4f}")


# ============================================================
# PLOT 1: RISK VS COST FOR EVERY SCENARIO
# ============================================================

plt.figure(figsize=(11, 7))

for scenario_name, strategies in results.items():

    x = []
    y = []

    for strategy_name, summary in strategies.items():
        x.append(summary["std_pnl"])
        y.append(summary["mean_cost"])

    plt.plot(
        x,
        y,
        marker="o",
        label=scenario_name,
    )

plt.xlabel("Standard Deviation of P&L (£)")
plt.ylabel("Mean Transaction Cost (£)")
plt.title("Event-Driven Hedging: Risk-Cost Trade-off Across Scenarios")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2: THRESHOLD VS AVERAGE NUMBER OF REBALANCES
# ============================================================

plt.figure(figsize=(11, 7))

for scenario_name in SCENARIOS:

    x = []
    y = []

    for threshold in DELTA_THRESHOLDS:
        label = f"Δ threshold {threshold:.2f}"

        x.append(threshold)
        y.append(
            results[scenario_name][label]["mean_trades"]
        )

    plt.plot(
        x,
        y,
        marker="o",
        label=scenario_name,
    )

plt.xlabel("Delta Threshold")
plt.ylabel("Mean Number of Rebalances")
plt.title("Delta Threshold vs Trading Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 3: THRESHOLD VS P&L RISK
# ============================================================

plt.figure(figsize=(11, 7))

for scenario_name in SCENARIOS:

    x = []
    y = []

    for threshold in DELTA_THRESHOLDS:
        label = f"Δ threshold {threshold:.2f}"

        x.append(threshold)
        y.append(
            results[scenario_name][label]["std_pnl"]
        )

    plt.plot(
        x,
        y,
        marker="o",
        label=scenario_name,
    )

plt.xlabel("Delta Threshold")
plt.ylabel("Standard Deviation of P&L (£)")
plt.title("Delta Threshold vs Hedging Risk")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 4: THRESHOLD VS TRANSACTION COST
# ============================================================

plt.figure(figsize=(11, 7))

for scenario_name in SCENARIOS:

    x = []
    y = []

    for threshold in DELTA_THRESHOLDS:
        label = f"Δ threshold {threshold:.2f}"

        x.append(threshold)
        y.append(
            results[scenario_name][label]["mean_cost"]
        )

    plt.plot(
        x,
        y,
        marker="o",
        label=scenario_name,
    )

plt.xlabel("Delta Threshold")
plt.ylabel("Mean Transaction Cost (£)")
plt.title("Delta Threshold vs Transaction Cost")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
