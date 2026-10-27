import math
import numpy as np


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def black_scholes_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_delta_value = norm_cdf(d1)
    put_delta_value = norm_cdf(d1) - 1
    gamma = norm_pdf(d1) / (S * sigma * np.sqrt(T))
    vega = (S * norm_pdf(d1) * np.sqrt(T)) / 100

    call_theta = (
        -(S * norm_pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm_cdf(d2)
    ) / 365

    put_theta = (
        -(S * norm_pdf(d1) * sigma) / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm_cdf(-d2)
    ) / 365

    return {
        "call_delta": call_delta_value,
        "put_delta": put_delta_value,
        "gamma": gamma,
        "vega": vega,
        "call_theta": call_theta,
        "put_theta": put_theta,
    }


def call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1)


def monte_carlo_option_price(
    S, K, T, r, sigma, simulations=100_000, option_type="call", seed=42
):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(simulations)

    ST = S * np.exp(
        (r - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * Z
    )

    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.exp(-r * T) * np.mean(payoff)


def simulate_stock_paths(
    S, T, r, sigma, steps=252, paths=20, seed=42
):
    rng = np.random.default_rng(seed)
    dt = T / steps

    prices = np.zeros((steps + 1, paths))
    prices[0] = S

    for t in range(1, steps + 1):
        Z = rng.standard_normal(paths)
        prices[t] = prices[t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

    return prices


def hedge_single_path(
    stock_path,
    K,
    T,
    r,
    pricing_sigma,
    transaction_cost_rate=0.0,
):
    steps = len(stock_path) - 1
    dt = T / steps
    S0 = stock_path[0]

    option_premium = black_scholes_call(S0, K, T, r, pricing_sigma)

    delta = call_delta(S0, K, T, r, pricing_sigma)
    shares = delta

    initial_transaction_cost = transaction_cost_rate * abs(shares) * S0
    cash = option_premium - shares * S0 - initial_transaction_cost

    total_transaction_cost = initial_transaction_cost

    deltas = [delta]
    cash_history = [cash]

    for i in range(1, steps):
        cash *= np.exp(r * dt)

        S = stock_path[i]
        remaining_T = T - i * dt

        new_delta = call_delta(
            S,
            K,
            remaining_T,
            r,
            pricing_sigma,
        )

        shares_to_trade = new_delta - shares
        trade_value = abs(shares_to_trade) * S
        transaction_cost = transaction_cost_rate * trade_value

        cash -= shares_to_trade * S + transaction_cost

        total_transaction_cost += transaction_cost
        shares = new_delta

        deltas.append(new_delta)
        cash_history.append(cash)

    cash *= np.exp(r * dt)

    ST = stock_path[-1]

    final_transaction_cost = transaction_cost_rate * abs(shares) * ST
    total_transaction_cost += final_transaction_cost

    hedge_value = cash + shares * ST - final_transaction_cost
    option_payoff = max(ST - K, 0)
    pnl = hedge_value - option_payoff

    return {
        "option_premium": option_premium,
        "final_stock_price": ST,
        "option_payoff": option_payoff,
        "hedge_value": hedge_value,
        "pnl": pnl,
        "total_transaction_cost": total_transaction_cost,
        "deltas": deltas,
        "cash_history": cash_history,
    }


def run_hedging_simulation(
    S=100,
    K=100,
    T=1,
    r=0.05,
    pricing_sigma=0.20,
    realised_sigma=None,
    steps=252,
    n_paths=1000,
    seed=42,
    transaction_cost_rate=0.0,
):
    if realised_sigma is None:
        realised_sigma = pricing_sigma

    paths = simulate_stock_paths(
        S=S,
        T=T,
        r=r,
        sigma=realised_sigma,
        steps=steps,
        paths=n_paths,
        seed=seed,
    )

    pnls = np.empty(n_paths)
    transaction_costs = np.empty(n_paths)

    for i in range(n_paths):
        result = hedge_single_path(
            stock_path=paths[:, i],
            K=K,
            T=T,
            r=r,
            pricing_sigma=pricing_sigma,
            transaction_cost_rate=transaction_cost_rate,
        )

        pnls[i] = result["pnl"]
        transaction_costs[i] = result["total_transaction_cost"]

    return pnls, transaction_costs


def hedge_single_path_event_driven(
    stock_path,
    K,
    T,
    r,
    pricing_sigma,
    delta_threshold,
    transaction_cost_rate=0.0,
):
    """
    Event-driven delta hedge for one short European call.

    The hedge is checked at every monitoring step, but shares are only traded
    when the model delta differs from the current stock position by at least
    delta_threshold.

    Example:
        delta_threshold=0.05 means rebalance only when the desired delta
        differs from the current hedge by 0.05 or more.
    """
    steps = len(stock_path) - 1
    dt = T / steps
    S0 = stock_path[0]

    option_premium = black_scholes_call(
        S0, K, T, r, pricing_sigma
    )

    initial_delta = call_delta(
        S0, K, T, r, pricing_sigma
    )

    shares = initial_delta

    initial_transaction_cost = (
        transaction_cost_rate * abs(shares) * S0
    )

    cash = (
        option_premium
        - shares * S0
        - initial_transaction_cost
    )

    total_transaction_cost = initial_transaction_cost
    rebalance_count = 0

    for i in range(1, steps):
        cash *= np.exp(r * dt)

        S = stock_path[i]
        remaining_T = T - i * dt

        target_delta = call_delta(
            S,
            K,
            remaining_T,
            r,
            pricing_sigma,
        )

        delta_gap = target_delta - shares

        # Only trade if the hedge has drifted far enough away
        # from the model delta.
        if abs(delta_gap) >= delta_threshold:
            shares_to_trade = delta_gap

            trade_value = abs(shares_to_trade) * S
            transaction_cost = (
                transaction_cost_rate * trade_value
            )

            cash -= (
                shares_to_trade * S
                + transaction_cost
            )

            total_transaction_cost += transaction_cost
            shares = target_delta
            rebalance_count += 1

    cash *= np.exp(r * dt)

    ST = stock_path[-1]

    # Liquidate remaining shares at expiry.
    final_transaction_cost = (
        transaction_cost_rate * abs(shares) * ST
    )

    total_transaction_cost += final_transaction_cost

    hedge_value = (
        cash
        + shares * ST
        - final_transaction_cost
    )

    option_payoff = max(ST - K, 0)
    pnl = hedge_value - option_payoff

    return {
        "option_premium": option_premium,
        "final_stock_price": ST,
        "option_payoff": option_payoff,
        "hedge_value": hedge_value,
        "pnl": pnl,
        "total_transaction_cost": total_transaction_cost,
        "rebalance_count": rebalance_count,
    }


def run_event_driven_hedging_simulation(
    S=100,
    K=100,
    T=1,
    r=0.05,
    pricing_sigma=0.20,
    realised_sigma=None,
    delta_threshold=0.05,
    monitoring_steps=None,
    n_paths=1000,
    seed=42,
    transaction_cost_rate=0.0,
):
    """
    Run event-driven delta hedging across many simulated stock paths.

    monitoring_steps:
        Number of times per option life that the hedge is checked.
        If None, uses approximately one check per trading day:
        round(252 * T), with a minimum of 1.

    The hedge is monitored at every step but only traded when the change
    required in delta exceeds delta_threshold.
    """
    if realised_sigma is None:
        realised_sigma = pricing_sigma

    if monitoring_steps is None:
        monitoring_steps = max(1, round(252 * T))

    paths = simulate_stock_paths(
        S=S,
        T=T,
        r=r,
        sigma=realised_sigma,
        steps=monitoring_steps,
        paths=n_paths,
        seed=seed,
    )

    pnls = np.empty(n_paths)
    transaction_costs = np.empty(n_paths)
    rebalance_counts = np.empty(n_paths)

    for i in range(n_paths):
        result = hedge_single_path_event_driven(
            stock_path=paths[:, i],
            K=K,
            T=T,
            r=r,
            pricing_sigma=pricing_sigma,
            delta_threshold=delta_threshold,
            transaction_cost_rate=transaction_cost_rate,
        )

        pnls[i] = result["pnl"]
        transaction_costs[i] = result["total_transaction_cost"]
        rebalance_counts[i] = result["rebalance_count"]

    return pnls, transaction_costs, rebalance_counts
