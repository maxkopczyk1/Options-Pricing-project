import numpy as np
import matplotlib.pyplot as plt
from black_scholes import run_hedging_simulation

hedging_frequencies = {
    "Monthly": 12,
    "Weekly": 52,
    "Daily": 252,
}

results = {}

for frequency, steps in hedging_frequencies.items():
    pnls = run_hedging_simulation(
        S=100,
        K=100,
        T=1,
        r=0.05,
        pricing_sigma=0.20,
        realised_sigma=0.20,
        steps=steps,
        n_paths=1000,
        seed=42,
    )

    results[frequency] = {
        "mean": np.mean(pnls),
        "std": np.std(pnls),
        "p5": np.percentile(pnls, 5),
        "p95": np.percentile(pnls, 95),
    }

for frequency, result in results.items():
    print(f"\n{frequency} hedging")
    print(f"Mean P&L: £{result['mean']:.4f}")
    print(f"Standard deviation: £{result['std']:.4f}")
    print(f"5th percentile: £{result['p5']:.4f}")
    print(f"95th percentile: £{result['p95']:.4f}")

frequencies = list(results.keys())
standard_deviations = [results[f]["std"] for f in frequencies]

plt.figure(figsize=(8, 5))
plt.bar(frequencies, standard_deviations)
plt.xlabel("Hedging Frequency")
plt.ylabel("Standard Deviation of P&L (£)")
plt.title("Hedging Frequency vs Hedging Risk")
plt.tight_layout()
plt.show()
