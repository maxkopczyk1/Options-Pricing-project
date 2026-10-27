import numpy as np
import matplotlib.pyplot as plt
from black_scholes import run_hedging_simulation

pricing_sigma = 0.20
realised_volatilities = [0.10, 0.20, 0.30, 0.40]

results = {}

for realised_sigma in realised_volatilities:
    pnls = run_hedging_simulation(
        S=100,
        K=100,
        T=1,
        r=0.05,
        pricing_sigma=pricing_sigma,
        realised_sigma=realised_sigma,
        steps=252,
        n_paths=2000,
        seed=42,
    )

    results[realised_sigma] = {
        "mean": np.mean(pnls),
        "std": np.std(pnls),
        "p5": np.percentile(pnls, 5),
        "p95": np.percentile(pnls, 95),
    }

for realised_sigma, result in results.items():
    print(f"\nRealised volatility: {realised_sigma:.0%}")
    print(f"Mean P&L: £{result['mean']:.4f}")
    print(f"Standard deviation: £{result['std']:.4f}")
    print(f"5th percentile: £{result['p5']:.4f}")
    print(f"95th percentile: £{result['p95']:.4f}")

vol_labels = [f"{vol:.0%}" for vol in realised_volatilities]
mean_pnls = [results[vol]["mean"] for vol in realised_volatilities]

plt.figure(figsize=(8, 5))
plt.plot(vol_labels, mean_pnls, marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Realised Volatility")
plt.ylabel("Mean Hedging P&L (£)")
plt.title("Realised Volatility vs Delta-Hedged P&L")
plt.tight_layout()
plt.show()
