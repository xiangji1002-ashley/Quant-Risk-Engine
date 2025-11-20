# Quant Risk Engine: Monte Carlo VaR Simulation

## 📖 Overview
This project implements a **Monte Carlo Simulation Engine** to calculate **Value at Risk (VaR)** and **Expected Shortfall (ES)** for a hypothetical portfolio. It demonstrates the transition from traditional Excel-based modeling to a robust **Python-based quantitative framework**.

## 🚀 Key Features
* **Stochastic Modeling:** Simulates asset price paths using Geometric Brownian Motion (GBM).
* **Risk Metrics:** Calculates 95% and 99% VaR & CVaR (Conditional VaR).
* **Visualization:** Generates distribution plots of potential future portfolio values.
* **Vectorization:** Utilizes `NumPy` for optimized matrix computations (replacing slow iterative loops).

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Libraries:** `NumPy`, `SciPy`, `Matplotlib`, `Pandas`
* **Concepts:** Stochastic Calculus, Statistical Analysis, Financial Risk Management

## 📊 How It Works (Logic)
1.  **Input Parameters:** Takes current asset price, volatility ($\sigma$), drift ($\mu$), and time horizon ($T$).
2.  **Simulation:** Generates 10,000+ random paths based on the discretized GBM formula:
    $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
3.  **Aggregation:** Computes the terminal portfolio value distribution.
4.  **Risk Calculation:** Identifies the tail loss percentiles to determine VaR.

## 📸 Sample Output

## 👨‍💻 Author
**Kehan Wang**
* M.S. in Information Systems, NYU
* M.S. in Financial Engineering, NUS
* [LinkedIn Profile]linkedin.com/in/kehan-wang-75b2b4361 | [Email](mailto:xiangji1002@gmail.com)
