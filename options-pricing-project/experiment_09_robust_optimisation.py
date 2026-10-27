import csv
import numpy as np
import matplotlib.pyplot as plt

from black_scholes import (
    run_hedging_simulation,
    run_event_driven_hedging_simulation,
)

R = 0.05
TRANSACTION_COST_RATE = 0.001

TARGET_RISK_INCREASE = 0.10
TRAINING_RISK_LIMIT = 0.08

THRESHOLDS = np.arange(0.005, 0.0801, 0.005)

TRAINING_SEEDS = [11, 23, 42, 87, 123]
PATHS_PER_TRAINING_SEED = 600

VALIDATION_SEEDS = [2026, 2027, 2028]
PATHS_PER_VALIDATION_SEED = 1500

SCENARIOS = {
    "ATM 1Y": {
        "S": 100, "K": 100, "T": 1.00,
        "pricing_sigma": 0.20, "realised_sigma": 0.20,
    },
    "ITM 1Y": {
        "S": 100, "K": 90, "T": 1.00,
        "pricing_sigma": 0.20, "realised_sigma": 0.20,
    },
    "OTM 1Y": {
        "S": 100, "K": 110, "T": 1.00,
        "pricing_sigma": 0.20, "realised_sigma": 0.20,
    },
    "ATM 3M": {
        "S": 100, "K": 100, "T": 0.25,
        "pricing_sigma": 0.20, "realised_sigma": 0.20,
    },
    "ATM 1Y High Vol": {
        "S": 100, "K": 100, "T": 1.00,
        "pricing_sigma": 0.40, "realised_sigma": 0.40,
    },
}


def daily_benchmark(params, n_paths, seed):
    steps = max(1, round(252 * params["T"]))

    pnls, costs = run_hedging_simulation(
        S=params["S"],
        K=params["K"],
        T=params["T"],
        r=R,
        pricing_sigma=params["pricing_sigma"],
        realised_sigma=params["realised_sigma"],
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )

    return {
        "std": float(np.std(pnls)),
        "mean_cost": float(np.mean(costs)),
        "mean_pnl": float(np.mean(pnls)),
        "p5": float(np.percentile(pnls, 5)),
        "p95": float(np.percentile(pnls, 95)),
    }


def threshold_strategy(params, threshold, n_paths, seed):
    monitoring_steps = max(1, round(252 * params["T"]))

    pnls, costs, trades = run_event_driven_hedging_simulation(
        S=params["S"],
        K=params["K"],
        T=params["T"],
        r=R,
        pricing_sigma=params["pricing_sigma"],
        realised_sigma=params["realised_sigma"],
        delta_threshold=threshold,
        monitoring_steps=monitoring_steps,
        n_paths=n_paths,
        seed=seed,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )

    return {
        "std": float(np.std(pnls)),
        "mean_cost": float(np.mean(costs)),
        "mean_trades": float(np.mean(trades)),
        "mean_pnl": float(np.mean(pnls)),
        "p5": float(np.percentile(pnls, 5)),
        "p95": float(np.percentile(pnls, 95)),
    }


selected_thresholds = {}
training_summary = {}

for scenario_name, params in SCENARIOS.items():
    print("\n" + "=" * 82)
    print(f"ROBUST OPTIMISATION: {scenario_name}")
    print("=" * 82)

    daily_by_seed = {
        seed: daily_benchmark(params, PATHS_PER_TRAINING_SEED, seed)
        for seed in TRAINING_SEEDS
    }

    mean_daily_cost = np.mean(
        [x["mean_cost"] for x in daily_by_seed.values()]
    )
    mean_daily_sd = np.mean(
        [x["std"] for x in daily_by_seed.values()]
    )

    print(f"\nAverage daily benchmark SD: £{mean_daily_sd:.4f}")
    print(f"Average daily transaction cost: £{mean_daily_cost:.4f}")

    threshold_results = {}

    for threshold in THRESHOLDS:
        relative_risks = []
        costs = []
        trades = []
        pnls = []

        for seed in TRAINING_SEEDS:
            result = threshold_strategy(
                params,
                float(threshold),
                PATHS_PER_TRAINING_SEED,
                seed,
            )

            daily = daily_by_seed[seed]

            relative_risks.append(
                result["std"] / daily["std"] - 1
            )
            costs.append(result["mean_cost"])
            trades.append(result["mean_trades"])
            pnls.append(result["mean_pnl"])

        mean_risk_increase = float(np.mean(relative_risks))
        worst_risk_increase = float(np.max(relative_risks))
        mean_cost = float(np.mean(costs))
        mean_trades = float(np.mean(trades))
        mean_pnl = float(np.mean(pnls))

        feasible = (
            mean_risk_increase <= TRAINING_RISK_LIMIT
            and worst_risk_increase <= TARGET_RISK_INCREASE
        )

        threshold_results[float(threshold)] = {
            "mean_risk_increase": mean_risk_increase,
            "worst_risk_increase": worst_risk_increase,
            "mean_cost": mean_cost,
            "mean_trades": mean_trades,
            "mean_pnl": mean_pnl,
            "feasible": feasible,
        }

        status = "FEASIBLE" if feasible else "NO"

        print(
            f"Threshold {threshold:.3f} | "
            f"Avg risk +{mean_risk_increase * 100:5.2f}% | "
            f"Worst +{worst_risk_increase * 100:5.2f}% | "
            f"Cost £{mean_cost:.4f} | "
            f"Trades {mean_trades:6.1f} | "
            f"{status}"
        )

    feasible_thresholds = [
        (threshold, result)
        for threshold, result in threshold_results.items()
        if result["feasible"]
    ]

    if not feasible_thresholds:
        selected_thresholds[scenario_name] = None
        print("\nNo threshold satisfied the robust risk constraint.")
    else:
        best_threshold, best_result = min(
            feasible_thresholds,
            key=lambda item: item[1]["mean_cost"],
        )

        selected_thresholds[scenario_name] = best_threshold

        cost_saving = (
            (mean_daily_cost - best_result["mean_cost"])
            / mean_daily_cost
            * 100
        )

        print("\nROBUST SELECTED THRESHOLD")
        print(f"Threshold: {best_threshold:.3f}")
        print(
            f"Average risk increase: "
            f"{best_result['mean_risk_increase'] * 100:.2f}%"
        )
        print(
            f"Worst-seed risk increase: "
            f"{best_result['worst_risk_increase'] * 100:.2f}%"
        )
        print(
            f"Average transaction cost: "
            f"£{best_result['mean_cost']:.4f}"
        )
        print(
            f"Average rebalances: "
            f"{best_result['mean_trades']:.1f}"
        )
        print(
            f"Training cost saving vs daily: "
            f"{cost_saving:.2f}%"
        )

    training_summary[scenario_name] = {
        "daily_mean_cost": mean_daily_cost,
        "daily_mean_sd": mean_daily_sd,
        "thresholds": threshold_results,
    }


validation_summary = {}

for scenario_name, params in SCENARIOS.items():
    threshold = selected_thresholds[scenario_name]

    print("\n" + "=" * 82)
    print(f"ROBUST VALIDATION: {scenario_name}")
    print("=" * 82)

    if threshold is None:
        print("Skipped: no robust threshold selected.")
        continue

    seed_results = []

    for seed in VALIDATION_SEEDS:
        daily = daily_benchmark(
            params,
            PATHS_PER_VALIDATION_SEED,
            seed,
        )

        strategy = threshold_strategy(
            params,
            threshold,
            PATHS_PER_VALIDATION_SEED,
            seed,
        )

        relative_risk = strategy["std"] / daily["std"] - 1
        passed = relative_risk <= TARGET_RISK_INCREASE

        cost_saving = (
            (daily["mean_cost"] - strategy["mean_cost"])
            / daily["mean_cost"]
            * 100
        )

        seed_results.append({
            "seed": seed,
            "daily_sd": daily["std"],
            "strategy_sd": strategy["std"],
            "risk_increase": relative_risk,
            "daily_cost": daily["mean_cost"],
            "strategy_cost": strategy["mean_cost"],
            "cost_saving": cost_saving,
            "mean_trades": strategy["mean_trades"],
            "mean_pnl": strategy["mean_pnl"],
            "p5": strategy["p5"],
            "p95": strategy["p95"],
            "passed": passed,
        })

        status = "PASS" if passed else "FAIL"

        print(f"\nSeed {seed}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Daily SD: £{daily['std']:.4f}")
        print(f"Strategy SD: £{strategy['std']:.4f}")
        print(
            f"Risk increase: "
            f"{relative_risk * 100:.2f}% [{status}]"
        )
        print(
            f"Transaction cost saving: "
            f"{cost_saving:.2f}%"
        )
        print(
            f"Mean rebalances: "
            f"{strategy['mean_trades']:.1f}"
        )

    average_risk = float(
        np.mean([x["risk_increase"] for x in seed_results])
    )
    worst_risk = float(
        np.max([x["risk_increase"] for x in seed_results])
    )
    average_saving = float(
        np.mean([x["cost_saving"] for x in seed_results])
    )
    average_trades = float(
        np.mean([x["mean_trades"] for x in seed_results])
    )
    all_passed = all(x["passed"] for x in seed_results)

    validation_summary[scenario_name] = {
        "threshold": threshold,
        "average_risk_increase": average_risk,
        "worst_risk_increase": worst_risk,
        "average_cost_saving": average_saving,
        "average_trades": average_trades,
        "all_passed": all_passed,
        "seed_results": seed_results,
    }

    print("\nVALIDATION SUMMARY")
    print(
        f"Average risk increase: "
        f"{average_risk * 100:.2f}%"
    )
    print(
        f"Worst validation risk increase: "
        f"{worst_risk * 100:.2f}%"
    )
    print(
        f"Average transaction cost saving: "
        f"{average_saving:.2f}%"
    )
    print(f"Average rebalances: {average_trades:.1f}")
    print(
        "Result: "
        + ("ROBUST PASS" if all_passed else "NOT FULLY ROBUST")
    )


csv_filename = "experiment_09_robust_validation.csv"

with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Scenario",
        "Selected Threshold",
        "Validation Seed",
        "Daily SD",
        "Strategy SD",
        "Risk Increase %",
        "Passed 10% Limit",
        "Daily Cost",
        "Strategy Cost",
        "Cost Saving %",
        "Mean Rebalances",
        "Mean P&L",
        "5th Percentile",
        "95th Percentile",
    ])

    for scenario_name, data in validation_summary.items():
        for result in data["seed_results"]:
            writer.writerow([
                scenario_name,
                data["threshold"],
                result["seed"],
                result["daily_sd"],
                result["strategy_sd"],
                result["risk_increase"] * 100,
                result["passed"],
                result["daily_cost"],
                result["strategy_cost"],
                result["cost_saving"],
                result["mean_trades"],
                result["mean_pnl"],
                result["p5"],
                result["p95"],
            ])

print(f"\nSaved robust validation results to {csv_filename}")


scenario_names = list(validation_summary.keys())

threshold_values = [
    validation_summary[name]["threshold"]
    for name in scenario_names
]

cost_savings = [
    validation_summary[name]["average_cost_saving"]
    for name in scenario_names
]

risk_increases = [
    validation_summary[name]["average_risk_increase"] * 100
    for name in scenario_names
]

plt.figure(figsize=(10, 6))
plt.bar(scenario_names, threshold_values)
plt.ylabel("Robust Delta Threshold")
plt.title("Robust Event-Driven Hedge Threshold by Option Scenario")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(scenario_names, cost_savings)
plt.ylabel("Average Transaction Cost Saving (%)")
plt.title("Out-of-Sample Cost Saving from Robust Event-Driven Hedging")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(scenario_names, risk_increases)
plt.axhline(
    TARGET_RISK_INCREASE * 100,
    linestyle="--",
    label="10% risk limit",
)
plt.ylabel("Average P&L Risk Increase vs Daily (%)")
plt.title("Out-of-Sample Risk Increase from Robust Event-Driven Hedging")
plt.xticks(rotation=25, ha="right")
plt.legend()
plt.tight_layout()
plt.show()
