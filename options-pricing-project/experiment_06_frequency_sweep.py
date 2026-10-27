import numpy as np
import matplotlib.pyplot as plt

from black_scholes import run_hedging_simulation


# ============================================================
# EXPERIMENT 6: HEDGING FREQUENCY SWEEP
# ============================================================

# Keep pricing and realised volatility equal so we isolate
# the trade-off between hedging accuracy and transaction costs.
pricing_sigma = 0.20
realised_sigma = 0.20

# 0.10% = 10 basis points per stock trade.
transaction_cost_rate = 0.001

# Number of hedge adjustments per year.
hedging_frequencies = {
    "Monthly": 12,
    "Fortnightly": 26,
    "Weekly": 52,
    "2x Weekly": 104,
    "3x Weekly": 156,
    "Daily": 252,
}

results = {}

for label, steps in hedging_frequencies.items():
    pnls, transaction_costs = run_hedging_simulation(
        S=100,
        K=100,
        T=1,
        r=0.05,
        pricing_sigma=pricing_sigma,
        realised_sigma=realised_sigma,
        steps=steps,
        n_paths=3000,
        seed=42,
        transaction_cost_rate=transaction_cost_rate,
    )

    results[label] = {
        "steps": steps,
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "mean_transaction_cost": np.mean(transaction_costs),
        "p5": np.percentile(pnls, 5),
        "p95": np.percentile(pnls, 95),
    }


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "HEDGING FREQUENCY SWEEP "
    f"({transaction_cost_rate:.2%} transaction cost per trade)"
)

for label, result in results.items():
    print(f"\n{label} ({result['steps']} hedges/year)")
    print(f"Mean P&L: £{result['mean_pnl']:.4f}")
    print(f"Standard deviation: £{result['std_pnl']:.4f}")
    print(
        f"Mean transaction cost: "
        f"£{result['mean_transaction_cost']:.4f}"
    )
    print(f"5th percentile: £{result['p5']:.4f}")
    print(f"95th percentile: £{result['p95']:.4f}")


# ============================================================
# PREPARE DATA FOR PLOTS
# ============================================================

labels = list(results.keys())
steps = [results[label]["steps"] for label in labels]

mean_pnls = [
    results[label]["mean_pnl"]
    for label in labels
]

std_pnls = [
    results[label]["std_pnl"]
    for label in labels
]

mean_costs = [
    results[label]["mean_transaction_cost"]
    for label in labels
]


# ============================================================
# PLOT 1: HEDGING FREQUENCY VS P&L RISK
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    steps,
    std_pnls,
    marker="o"
)

plt.xlabel("Hedges per Year")
plt.ylabel("Standard Deviation of P&L (£)")
plt.title("Hedging Frequency vs P&L Risk")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2: HEDGING FREQUENCY VS TRANSACTION COST
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    steps,
    mean_costs,
    marker="o"
)

plt.xlabel("Hedges per Year")
plt.ylabel("Mean Transaction Cost (£)")
plt.title("Hedging Frequency vs Transaction Cost")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 3: HEDGING FREQUENCY VS MEAN P&L
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    steps,
    mean_pnls,
    marker="o"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Hedges per Year")
plt.ylabel("Mean P&L (£)")
plt.title("Hedging Frequency vs Mean P&L")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 4: RISK-COST FRONTIER
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    std_pnls,
    mean_costs
)

for i, label in enumerate(labels):
    plt.annotate(
        label,
        (std_pnls[i], mean_costs[i]),
        xytext=(6, 5),
        textcoords="offset points"
    )

plt.xlabel("Standard Deviation of P&L (£)")
plt.ylabel("Mean Transaction Cost (£)")
plt.title("Risk-Cost Frontier Across Hedging Frequencies")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
