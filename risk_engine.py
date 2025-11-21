import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class RiskEngine:
    """
    A Monte Carlo Simulation Engine for Financial Risk Management.
    Calculates Value at Risk (VaR) and Expected Shortfall (ES).
    """

    def __init__(self, initial_price: float, volatility: float, drift: float, time_horizon: int = 252):
        """
        Initialize the Risk Engine parameters.
        :param initial_price: Current price of the portfolio/asset (S0)
        :param volatility: Annualized volatility (sigma)
        :param drift: Annualized expected return (mu)
        :param time_horizon: Time horizon in days (T)
        """
        self.S0 = initial_price
        self.sigma = volatility
        self.mu = drift
        self.T = time_horizon / 252.0  # Convert days to years
        self.dt = 1 / 252.0            # Time step
        self.simulations = None

    def run_monte_carlo(self, num_simulations: int = 10000, seed: Optional[int] = 42) -> np.ndarray:
        """
        Executes Geometric Brownian Motion (GBM) simulation.
        Formula: dS_t = mu * S_t * dt + sigma * S_t * dW_t
        """
        if seed:
            np.random.seed(seed)

        # Generate random Brownian Motion paths
        # Z ~ N(0, 1)
        Z = np.random.normal(0, 1, (int(self.T * 252), num_simulations)) # Assuming T is integer days for simplicity adjustment in real implementation
        
        # Vectorized Simulation
        # St = S0 * exp((mu - 0.5 * sigma^2)t + sigma * Wt)
        # Here we simulate daily returns and accumulate them
        
        # Pre-compute constants
        drift_term = (self.mu - 0.5 * self.sigma**2) * self.dt
        shock_term = self.sigma * np.sqrt(self.dt) * np.random.normal(0, 1, (int(self.T * 252), num_simulations))
        
        # Calculate price paths
        # Use log returns for numerical stability
        log_returns = drift_term + shock_term
        
        # Cumulative sum of log returns
        cumulative_log_returns = np.cumsum(log_returns, axis=0)
        
        # Apply to initial price
        self.simulations = self.S0 * np.exp(cumulative_log_returns)
        
        return self.simulations

    def calculate_metrics(self, confidence_level: float = 0.95) -> dict:
        """
        Calculates VaR and ES based on terminal prices.
        """
        if self.simulations is None:
            raise ValueError("Run simulation first using run_monte_carlo()")

        # Get terminal prices (last row)
        terminal_prices = self.simulations[-1, :]
        
        # Calculate PnL (Profit and Loss)
        pnl = terminal_prices - self.S0
        
        # Calculate VaR (Percentile)
        # VaR is the loss threshold. If we look at the bottom (1-alpha) quantile.
        var_threshold = np.percentile(pnl, (1 - confidence_level) * 100)
        
        # Calculate Expected Shortfall (ES) / CVaR
        # Average of losses exceeding VaR
        es_value = pnl[pnl <= var_threshold].mean()

        return {
            "Confidence Level": f"{confidence_level*100}%",
            "VaR": round(abs(var_threshold), 4), # Absolute value for reporting "Risk amount"
            "Expected Shortfall (ES)": round(abs(es_value), 4),
            "Mean Terminal Price": round(terminal_prices.mean(), 2)
        }

if __name__ == "__main__":
    # --- Demo Usage ---
    print("Initializing Risk Engine...")
    
    # Scenario: Portfolio worth $1,000,000, 20% Volatility, 5% Drift, 1 Year horizon
    engine = RiskEngine(initial_price=1000000, volatility=0.20, drift=0.05, time_horizon=252)
    
    print("Running 10,000 Monte Carlo Simulations...")
    paths = engine.run_monte_carlo(num_simulations=10000)
    
    print("Calculating Risk Metrics...")
    metrics = engine.calculate_metrics(confidence_level=0.99)
    
    print("\n--- RISK REPORT ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    print("\n[Success] Simulation Complete.")
