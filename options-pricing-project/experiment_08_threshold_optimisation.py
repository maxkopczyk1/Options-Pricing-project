import csv
import numpy as np
import matplotlib.pyplot as plt

from black_scholes import (
    run_hedging_simulation,
    run_event_driven_hedging_simulation,
)


# ============================================================
# EXPERIMENT 8: OPTIMISE EVENT-DRIVEN DELTA-HEDGE THRESHOLDS
# ============================================================

# We define "optimal" as:
#
#   the LOWEST-COST delta threshold whose P&L standard deviation
#   stays within a chosen percentage of the daily-hedging benchmark.
#
# Example:
#   10% risk tolerance means
#
#       strategy P&L SD <= 1.10 * daily hedge P&L SD
#
# We optimise on one Monte Carlo sample ("training") and then test
# the selected threshold on an independent seed ("validation").


TRANSACTION_COST_RATE = 0.001      # 0.10% = 10 basis points
R = 0.05

# Training sample used to select thresholds.
TRAIN_PATHS = 1500
TRAIN_SEED = 42

# Larger independent sample used only after the threshold is selected.
VALIDATION_PATHS = 5000
VALIDATION_SEED = 2026

# Fine threshold grid: 0.005, 0.010, ..., 0.100.
THRESHOLDS = np.arange(0.005, 0.1001, 0.005)

# How much extra P&L volatility we tolerate relative to daily hedging.
RISK_TOLERANCES = [
    0.05,
    0.10,
    0.20,
    0.30,
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


def summarise(pnls, transaction_costs, rebalance_counts=None):
    result = {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "mean_cost": float(np.mean(transaction_costs)),
        "p5": float(np.percentile(pnls, 5)),
        "p95": float(np.percentile(pnls, 95)),
    }

    if rebalance_counts is not None:
        result["mean_rebalances"] = float(
            np.mean(rebalance_counts)
        )

    return result


def run_daily_benchmark(params, n_paths, seed):
    monitoring_steps = max(
        1,
        round(252 * params["T"])
    )

    pnls, costs = run_hedging_simulation(
        S=params["S"],
        K=params["K"],
        T=params["T"],
        r=R,
        pricing_sigma=params["pricing_sigma"],
        realised_sigma=params["realised_sigma"],
        steps=monitoring_steps,
        n_paths=n_paths,
        seed=seed,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )

    summary = summarise(
        pnls,
        costs,
    )

    summary["scheduled_rebalances"] = (
        monitoring_steps - 1
    )

    return summary


def run_threshold(
    params,
    threshold,
    n_paths,
    seed,
):
    monitoring_steps = max(
        1,
        round(252 * params["T"])
    )

    pnls, costs, trades = (
        run_event_driven_hedging_simulation(
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
    )

    return summarise(
        pnls,
        costs,
        trades,
    )


# ============================================================
# 1. TRAINING / OPTIMISATION
# ============================================================

training_results = {}
selected_thresholds = {}


for scenario_name, params in SCENARIOS.items():

    print("\n" + "=" * 78)
    print(f"OPTIMISING: {scenario_name}")
    print("=" * 78)

    daily = run_daily_benchmark(
        params=params,
        n_paths=TRAIN_PATHS,
        seed=TRAIN_SEED,
    )

    print(
        f"\nDaily benchmark SD: "
        f"£{daily['std_pnl']:.4f}"
    )
    print(
        f"Daily benchmark mean cost: "
        f"£{daily['mean_cost']:.4f}"
    )

    threshold_results = {}

    for threshold in THRESHOLDS:

        result = run_threshold(
            params=params,
            threshold=float(threshold),
            n_paths=TRAIN_PATHS,
            seed=TRAIN_SEED,
        )

        threshold_results[
            float(threshold)
        ] = result

        print(
            f"Threshold {threshold:.3f} | "
            f"SD £{result['std_pnl']:.4f} | "
            f"Cost £{result['mean_cost']:.4f} | "
            f"Trades {result['mean_rebalances']:.1f}"
        )

    training_results[scenario_name] = {
        "daily": daily,
        "thresholds": threshold_results,
    }

    selected_thresholds[scenario_name] = {}

    print("\nSELECTED THRESHOLDS")

    for tolerance in RISK_TOLERANCES:

        max_allowed_sd = (
            daily["std_pnl"]
            * (1 + tolerance)
        )

        feasible = [
            (threshold, result)
            for threshold, result
            in threshold_results.items()
            if result["std_pnl"] <= max_allowed_sd
        ]

        if not feasible:
            selected_thresholds[
                scenario_name
            ][tolerance] = None

            print(
                f"{tolerance:.0%} tolerance: "
                f"no threshold met the constraint"
            )

            continue

        # Among all feasible thresholds, choose the one with
        # the lowest expected transaction cost.
        best_threshold, best_result = min(
            feasible,
            key=lambda item: item[1]["mean_cost"],
        )

        selected_thresholds[
            scenario_name
        ][tolerance] = best_threshold

        print(
            f"{tolerance:.0%} tolerance: "
            f"threshold {best_threshold:.3f} | "
            f"SD £{best_result['std_pnl']:.4f} | "
            f"Cost £{best_result['mean_cost']:.4f} | "
            f"Trades {best_result['mean_rebalances']:.1f}"
        )


# ============================================================
# 2. OUT-OF-SAMPLE VALIDATION
# ============================================================

validation_results = {}


for scenario_name, params in SCENARIOS.items():

    print("\n" + "=" * 78)
    print(f"VALIDATING: {scenario_name}")
    print("=" * 78)

    validation_daily = run_daily_benchmark(
        params=params,
        n_paths=VALIDATION_PATHS,
        seed=VALIDATION_SEED,
    )

    validation_results[scenario_name] = {
        "daily": validation_daily,
        "strategies": {},
    }

    print(
        f"\nValidation daily SD: "
        f"£{validation_daily['std_pnl']:.4f}"
    )

    print(
        f"Validation daily cost: "
        f"£{validation_daily['mean_cost']:.4f}"
    )

    for tolerance in RISK_TOLERANCES:

        threshold = selected_thresholds[
            scenario_name
        ][tolerance]

        if threshold is None:
            continue

        result = run_threshold(
            params=params,
            threshold=threshold,
            n_paths=VALIDATION_PATHS,
            seed=VALIDATION_SEED,
        )

        allowed_sd = (
            validation_daily["std_pnl"]
            * (1 + tolerance)
        )

        constraint_passed = (
            result["std_pnl"] <= allowed_sd
        )

        validation_results[
            scenario_name
        ]["strategies"][tolerance] = {
            "threshold": threshold,
            "result": result,
            "allowed_sd": allowed_sd,
            "constraint_passed": constraint_passed,
        }

        status = (
            "PASS"
            if constraint_passed
            else "FAIL"
        )

        print(
            f"\n{tolerance:.0%} tolerance"
        )

        print(
            f"Chosen threshold: "
            f"{threshold:.3f}"
        )

        print(
            f"Validation SD: "
            f"£{result['std_pnl']:.4f}"
        )

        print(
            f"Allowed SD: "
            f"£{allowed_sd:.4f} [{status}]"
        )

        print(
            f"Mean transaction cost: "
            f"£{result['mean_cost']:.4f}"
        )

        print(
            f"Mean rebalances: "
            f"{result['mean_rebalances']:.1f}"
        )

        print(
            f"Mean P&L: "
            f"£{result['mean_pnl']:.4f}"
        )

        print(
            f"5th percentile: "
            f"£{result['p5']:.4f}"
        )

        print(
            f"95th percentile: "
            f"£{result['p95']:.4f}"
        )


# ============================================================
# 3. SAVE VALIDATION SUMMARY TO CSV
# ============================================================

csv_filename = (
    "experiment_08_optimisation_results.csv"
)

with open(
    csv_filename,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Scenario",
        "Risk Tolerance",
        "Selected Threshold",
        "Daily SD",
        "Strategy SD",
        "Allowed SD",
        "Constraint Passed",
        "Daily Mean Cost",
        "Strategy Mean Cost",
        "Mean Rebalances",
        "Mean P&L",
        "5th Percentile",
        "95th Percentile",
    ])

    for scenario_name, scenario_data in (
        validation_results.items()
    ):

        daily = scenario_data["daily"]

        for tolerance, strategy_data in (
            scenario_data["strategies"].items()
        ):

            result = strategy_data["result"]

            writer.writerow([
                scenario_name,
                tolerance,
                strategy_data["threshold"],
                daily["std_pnl"],
                result["std_pnl"],
                strategy_data["allowed_sd"],
                strategy_data[
                    "constraint_passed"
                ],
                daily["mean_cost"],
                result["mean_cost"],
                result["mean_rebalances"],
                result["mean_pnl"],
                result["p5"],
                result["p95"],
            ])


print(
    f"\nSaved validation results to "
    f"{csv_filename}"
)


# ============================================================
# 4. PLOT OPTIMAL THRESHOLD BY SCENARIO
#    USING THE 10% RISK-TOLERANCE RULE
# ============================================================

TARGET_TOLERANCE = 0.10

scenario_labels = []
optimal_threshold_values = []

for scenario_name in SCENARIOS:

    threshold = selected_thresholds[
        scenario_name
    ][TARGET_TOLERANCE]

    if threshold is not None:
        scenario_labels.append(
            scenario_name
        )

        optimal_threshold_values.append(
            threshold
        )


plt.figure(figsize=(10, 6))

plt.bar(
    scenario_labels,
    optimal_threshold_values,
)

plt.ylabel("Optimal Delta Threshold")
plt.title(
    "Optimal Event-Driven Hedge Threshold "
    "(10% Risk Tolerance)"
)

plt.xticks(
    rotation=25,
    ha="right",
)

plt.tight_layout()
plt.show()


# ============================================================
# 5. PLOT COST SAVING VS DAILY HEDGING
#    USING THE 10% RISK-TOLERANCE RULE
# ============================================================

cost_savings_percent = []
cost_saving_labels = []

for scenario_name in SCENARIOS:

    data = validation_results[
        scenario_name
    ]

    if TARGET_TOLERANCE not in data[
        "strategies"
    ]:
        continue

    daily_cost = data[
        "daily"
    ]["mean_cost"]

    strategy_cost = data[
        "strategies"
    ][TARGET_TOLERANCE][
        "result"
    ]["mean_cost"]

    saving = (
        (daily_cost - strategy_cost)
        / daily_cost
        * 100
    )

    cost_saving_labels.append(
        scenario_name
    )

    cost_savings_percent.append(
        saving
    )


plt.figure(figsize=(10, 6))

plt.bar(
    cost_saving_labels,
    cost_savings_percent,
)

plt.ylabel(
    "Transaction Cost Saving vs Daily (%)"
)

plt.title(
    "Cost Saving from Optimised "
    "Event-Driven Hedging"
)

plt.xticks(
    rotation=25,
    ha="right",
)

plt.tight_layout()
plt.show()
