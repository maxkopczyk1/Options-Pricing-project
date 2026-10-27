import numpy as np
import matplotlib.pyplot as plt

from black_scholes import run_hedging_simulation


# ============================================================
# EXPERIMENT 5: TRANSACTION COSTS VS HEDGING FREQUENCY
# ============================================================

# We keep pricing and realised volatility equal at 20%.
# This isolates the effect of trading costs.

hedging_frequencies = {
    "Monthly": 12,
    "Weekly": 52,
    "Daily": 252,
}

# 0.10% = 10 basis points per share trade.
transaction_cost_rate = 0.001

results = {}

for frequency, steps in hedging_frequencies.items():
    pnls, transaction_costs = run_hedging_simulation(
        S=100,
        K=100,
        T=1,
        r=0.05,
        pricing_sigma=0.20,
        realised_sigma=0.20,
        steps=steps,
        n_paths=3000,
        seed=42,
        transaction_cost_rate=transaction_cost_rate,
    )

    results[frequency] = {
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "p5": np.percentile(pnls, 5),
        "p95": np.percentile(pnls, 95),
        "mean_transaction_cost": np.mean(transaction_costs),
    }


print(
    "TRANSACTION COST EXPERIMENT "
    f"({transaction_cost_rate:.2%} per trade)"
)

for frequency, result in results.items():
    print(f"\n{frequency} hedging")
    print(f"Mean P&L: £{result['mean_pnl']:.4f}")
    print(f"Standard deviation: £{result['std_pnl']:.4f}")
    print(f"Mean transaction cost: £{result['mean_transaction_cost']:.4f}")
    print(f"5th percentile: £{result['p5']:.4f}")
    print(f"95th percentile: £{result['p95']:.4f}")


# ------------------------------------------------------------
# Plot 1: Mean P&L
# ------------------------------------------------------------

frequencies = list(results.keys())

mean_pnls = [
    results[f]["mean_pnl"]
    for f in frequencies
]

plt.figure(figsize=(8, 5))
plt.bar(frequencies, mean_pnls)
plt.axhline(0, linestyle="--")

plt.xlabel("Hedging Frequency")
plt.ylabel("Mean P&L (£)")
plt.title("Transaction Costs: Mean P&L by Hedging Frequency")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 2: Hedging risk
# ------------------------------------------------------------

std_pnls = [
    results[f]["std_pnl"]
    for f in frequencies
]

plt.figure(figsize=(8, 5))
plt.bar(frequencies, std_pnls)

plt.xlabel("Hedging Frequency")
plt.ylabel("Standard Deviation of P&L (£)")
plt.title("Transaction Costs: Hedging Risk by Frequency")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 3: Average trading cost
# ------------------------------------------------------------

mean_costs = [
    results[f]["mean_transaction_cost"]
    for f in frequencies
]

plt.figure(figsize=(8, 5))
plt.bar(frequencies, mean_costs)

plt.xlabel("Hedging Frequency")
plt.ylabel("Mean Transaction Cost (£)")
plt.title("Average Transaction Cost by Hedging Frequency")
plt.tight_layout()
plt.show()
