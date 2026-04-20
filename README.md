# Quant Trading System — Intraday FX Strategy

A systematic, fully backtested intraday FX trading strategy with realistic execution modeling and portfolio-level risk management.

This project develops and validates a fully systematic intraday forex trading strategy across EURUSD, GBPUSD, and USDJPY, including a complete backtesting, execution cost modeling, and portfolio simulation pipeline.

---

## Key Results

**Realistic Execution:**
* Return: **+136.71%**
* Sharpe Ratio: **1.60**
* Max Drawdown: **-5.36%**

**Worst-Case Execution (2× spread + slippage):**

* Return: **+62.97%**
* Sharpe Ratio: **0.867**
* Max Drawdown: **-8.74%**

These results are generated using a deterministic, event-driven backtesting engine with realistic spread and stochastic slippage modeling.

---

## Performance

### Equity Curve

![Equity Curve](results/equity_curve.png)

### Drawdown

![Drawdown](results/drawdown.png)

---

## Validation (Out-of-Sample)

### Walk-Forward Performance

Walk-forward validation tests the strategy on sequential out-of-sample periods using parameters fixed from prior data.

Performance remains consistently positive across these out-of-sample segments, supporting the robustness of the strategy and reducing the likelihood of overfitting.

![Walk Forward](results/walkforward.png)

---

## Parameter Robustness

The heatmap below shows strategy performance (Sharpe ratio) across nearby TP parameter combinations.

![Robustness](results/robustness_sharpe.png)

---

## Strategy Overview

The strategy combines three core components:

* **Multi-timeframe trend alignment** (1H + 4H)
* **Momentum confirmation** (15m break of structure)
* **Pullback entries** using price imbalance zones (5m)

Entries are executed via limit orders, with structure-based stops and a two-stage profit-taking model.

---

## Portfolio Construction

* Trades are executed across three pairs using **shared capital**
* Position sizing scales with portfolio equity
* A **5.5% open-risk cap** limits total exposure

This produces more realistic portfolio behavior than independent per-pair backtests.

---

## Sysytem Features

* Deterministic backtesting engine
* Realistic execution modeling (spread + stochastic slippage)
* Portfolio-level simulation with shared capital
* Walk-forward validation and robustness testing

---

## Project Structure

* `/src` — core backtesting and simulation code
* `/docs` — research papers
* `/results` — charts and outputs

---

## Papers

* **Condensed paper (5–8 min read):**
  [Condensed Paper](docs/quant_trading_condensed.docx)
  
* **Full research paper (technical):**
  [Full Research Paper](docs/full_paper.pdf)

---

## Limitations

* Performs best in trending markets
* Performance declines in low-volatility or choppy conditions
* Sensitive to execution costs

---

## Takeaway

This project demonstrates how combining structural market logic with disciplined execution and rigorous validation can produce a repeatable trading framework with measurable edge.
