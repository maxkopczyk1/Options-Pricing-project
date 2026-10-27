from black_scholes import black_scholes_call, monte_carlo_option_price

S = 100
K = 100
T = 1
r = 0.05
sigma = 0.20

bs_call = black_scholes_call(S, K, T, r, sigma)

simulation_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

print("BLACK-SCHOLES VS MONTE CARLO\n")
print(f"Black-Scholes call price: £{bs_call:.4f}\n")

for n in simulation_sizes:
    mc_call = monte_carlo_option_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        simulations=n,
        option_type="call",
        seed=42,
    )

    error = mc_call - bs_call

    print(
        f"Simulations: {n:>7,} | "
        f"MC price: £{mc_call:.4f} | "
        f"Error: £{error:.4f}"
    )
