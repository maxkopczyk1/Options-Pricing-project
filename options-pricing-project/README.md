# Options Pricing & Dynamic Hedging Project

## Structure

- `black_scholes.py` — all reusable pricing, simulation and hedging logic
- `experiment_01_pricing_validation.py` — Black-Scholes vs Monte Carlo
- `experiment_02_daily_hedging.py` — daily hedging P&L baseline
- `experiment_03_hedging_frequency.py` — monthly vs weekly vs daily hedging
- `experiment_04_volatility_mismatch.py` — pricing volatility vs realised volatility
....

## Install

```bash
pip install numpy matplotlib
```

## Run an experiment

```bash
python experiment_03_hedging_frequency.py
```
Dynamic Option Hedging Optimisation
A Python project investigating Black-Scholes option pricing, Monte Carlo simulation, dynamic delta hedging, transaction costs and event-driven hedge optimisation.
The central question is:
Can a delta-hedged option position be managed with fewer trades and lower transaction costs without materially increasing hedging risk?

The project develops from a standard Black-Scholes implementation into an event-driven hedging strategy that is optimised across option scenarios, validated across independent Monte Carlo samples, and finally tested on historical underlying price paths.
Key Result
The final historical-underlying demo tests found that the frozen event-driven hedge rules achieved approximately:
- 60–72% fewer hedge adjustments
- 9–19% lower transaction costs
- no more than 2.86% higher P&L standard deviation
relative to daily delta hedging across the tested scenarios.
Scenario	Delta threshold	Risk change	Cost saving	Fewer rebalances
AMZN ATM 1Y	0.015	+1.23%	9.91%	60.56%
AMZN ITM 1Y	0.015	+2.49%	10.86%	67.60%
AMZN OTM 1Y	0.025	+2.86%	19.04%	71.53%
AMZN ATM 3M	0.035	-0.06%	10.44%	61.02%
TSLA ATM 1Y High Vol	0.015	+0.46%	8.60%	61.06%


These are historical-underlying demo options rather than actual historical listed-option trades.
Project Structure
options-pricing-project/
│
├── black_scholes.py
│
├── experiment_01_pricing_validation.py
├── experiment_02_daily_hedging.py
├── experiment_03_hedging_frequency.py
├── experiment_04_volatility_mismatch.py
├── experiment_05_transaction_costs.py
├── experiment_06_frequency_sweep.py
├── experiment_07_event_driven_scenarios.py
├── experiment_08_threshold_optimisation.py
├── experiment_09_robust_optimisation.py
├── experiment_10_historical_demo_backtest.py
│
├── experiment_08_optimisation_results.csv
├── experiment_09_robust_validation.csv
├── experiment_10_historical_demo_detail.csv
├── experiment_10_historical_demo_summary.csv
│
└── PROJECT_WRITEUP.md
black_scholes.py contains the reusable pricing, simulation and hedging logic.
Each experiment is kept separate so that the core model remains clean.
Methodology
1. Black-Scholes pricing
European calls and puts are priced analytically using Black-Scholes.
The project also calculates:
- delta
- gamma
- vega
- theta
2. Monte Carlo validation
Geometric Brownian motion is used to simulate terminal stock prices and estimate option values.
The Monte Carlo price is compared with the analytical Black-Scholes price as an implementation check.
3. Dynamic delta hedging
A short call position is hedged by holding the Black-Scholes delta in the underlying stock.
The hedge is rebalanced over the life of the option.
Residual P&L measures the hedging error.
4. Hedging-frequency analysis
Monthly, weekly and daily hedging are compared.
More frequent hedging reduces discrete hedging error but requires more trading.
5. Volatility mismatch
The option is priced and hedged using one volatility while the underlying is simulated using another.
This demonstrates that a delta-hedged short option remains exposed to realised volatility.
6. Transaction costs
A proportional transaction cost is charged whenever stock is traded.
This creates a trade-off:
more frequent hedging
        ↓
lower hedging risk
        +
higher transaction cost
7. Event-driven hedging
Instead of hedging automatically at fixed times, the strategy only trades when:
\[
|\Delta_{target}-\Delta_{held}|
\geq
\delta^*
\]
where \(\delta^*\) is the hedge threshold.
This avoids small, low-value hedge adjustments.
8. Threshold optimisation
The threshold is chosen by solving:
\[
\min_{\delta^*}
E[\text{transaction cost}]
\]
subject to a constraint on P&L risk relative to daily hedging.
9. Robust optimisation
To reduce simulation overfitting:
- thresholds are trained across multiple Monte Carlo seeds
- a safety margin is used
- the worst training seed must remain within the final risk limit
- the chosen threshold is frozen
- validation is performed on unseen seeds
10. Historical-underlying demo backtest
The frozen thresholds are then tested on real AMZN and TSLA historical price paths.
Trailing realised volatility from the preceding 60 trading days is used as the volatility input available at option inception.
The strategy is not re-optimised on the historical paths.
Selected Results
Hedging frequency
Frequency	P&L SD
Monthly	£1.9401
Weekly	£0.9504
Daily	£0.4323


Daily hedging reduced P&L dispersion substantially relative to monthly hedging.
Volatility mismatch
Option priced and hedged at 20% volatility:
Realised volatility	Mean P&L
10%	£3.8350
20%	-£0.0017
30%	-£3.9405
40%	-£7.8566


A delta-hedged short option remained strongly exposed to the difference between pricing and realised volatility.
Transaction-cost trade-off
At a 0.10% proportional transaction cost:
Frequency	Mean P&L	P&L SD	Mean cost
Monthly	-£0.2684	£1.9204	£0.2317
Weekly	-£0.3553	£0.9446	£0.3482
Daily	-£0.6353	£0.4881	£0.6205


Installation
Core dependencies:
pip install numpy matplotlib
Historical demo experiment:
pip install pandas yfinance
Running the Experiments
Run individual experiments directly.
For example:
python experiment_03_hedging_frequency.py
or:
python experiment_09_robust_optimisation.py
For the historical demo backtest:
python experiment_10_historical_demo_backtest.py
Experiment 10 expects the robust validation CSV created by Experiment 9 to be present in the same directory.
Main Conclusion
The project finds that hedging every day is not necessarily the most efficient policy once transaction costs are introduced.
A calibrated event-driven delta band can substantially reduce trading activity and costs while preserving very similar hedge-risk characteristics.
The result also shows that the appropriate hedge threshold depends on the option scenario rather than being universal.
Limitations
This is an educational quantitative-finance project, not a production trading strategy.
Important simplifications include:
- Black-Scholes assumptions
- constant interest rates
- simplified proportional transaction costs
- no explicit bid-ask spread or market impact
- no volatility surface
- no stochastic volatility
- no jumps
- historical demo options rather than actual historical option-chain trades
The historical experiment uses real underlying prices but theoretical European calls.
Future Work
Possible extensions include:
- historical listed-option data
- implied-volatility surfaces
- bid-ask spreads
- stochastic-volatility models
- dynamic gamma-based hedge thresholds
- expected-shortfall constraints
- multi-option portfolios
Technologies
- Python
- NumPy
- Matplotlib
- pandas
- yfinance
- Monte Carlo simulation
- Black-Scholes option pricing
- constrained optimisation