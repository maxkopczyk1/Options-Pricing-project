import numpy as np
import matplotlib.pyplot as plt
from black_scholes import run_hedging_simulation

pnls = run_hedging_simulation(
    S=100,
    K=100,
    T=1,
    r=0.05,
    pricing_sigma=0.20,
    realised_sigma=0.20,
    steps=252,
    n_paths=1000,
    seed=42,
)

print("DAILY HEDGING BASELINE")
print(f"Mean P&L: £{np.mean(pnls):.4f}")
print(f"Standard deviation: £{np.std(pnls):.4f}")
print(f"Minimum P&L: £{np.min(pnls):.4f}")
print(f"Maximum P&L: £{np.max(pnls):.4f}")
print(f"5th percentile: £{np.percentile(pnls, 5):.4f}")
print(f"95th percentile: £{np.percentile(pnls, 95):.4f}")

plt.figure(figsize=(10, 6))
plt.hist(pnls, bins=50)
plt.axvline(np.mean(pnls), linestyle="--", label=f"Mean = £{np.mean(pnls):.3f}")
plt.xlabel("Hedging P&L (£)")
plt.ylabel("Frequency")
plt.title("Distribution of Daily Delta-Hedging P&L")
plt.legend()
plt.tight_layout()
plt.show()
