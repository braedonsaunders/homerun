"""
Bregman Projection for optimal arbitrage trade calculation.

The Bregman divergence respects the market maker's cost function structure,
providing the information-theoretically optimal way to remove arbitrage.

For LMSR (Logarithmic Market Scoring Rule), the Bregman divergence is the
KL divergence, measuring information-theoretic distance between probability
distributions.

Key insight from research:
- Maximum guaranteed profit from any trade equals D(μ*||θ)
- Where μ* is the Bregman projection of θ onto the arbitrage-free manifold M
- The projection μ* tells you the arbitrage-free price vector
- The divergence D(μ*||θ) tells you the maximum extractable profit
- The gradient ∇D tells you the trading direction
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    from scipy.optimize import minimize  # noqa: F401
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class ProjectionResult:
    """Result of Bregman projection."""
    projected_prices: np.ndarray
    arbitrage_profit: float
    optimal_positions: list[dict]
    converged: bool
    iterations: int


class BregmanProjector:
    """
    Compute Bregman projections for LMSR-based prediction markets.

    For LMSR, the convex function R(μ) is negative entropy:
    R(μ) = Σ μ_i × ln(μ_i)

    The Bregman divergence becomes KL divergence:
    D(μ||θ) = Σ μ_i × ln(μ_i / θ_i)
    """

    def __init__(self, epsilon: float = 1e-10, liquidity_param: float = 100.0):
        """
        Args:
            epsilon: Small constant to prevent log(0)
            liquidity_param: LMSR liquidity parameter b
        """
        self.epsilon = epsilon
        self.b = liquidity_param

    def kl_divergence(self, mu: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute KL divergence D(μ||θ).

        This represents the maximum extractable arbitrage profit.
        """
        mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
        theta = np.clip(theta, self.epsilon, 1 - self.epsilon)
        return float(np.sum(mu * np.log(mu / theta)))

    def kl_gradient(self, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Gradient of KL divergence with respect to μ."""
        mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
        theta = np.clip(theta, self.epsilon, 1 - self.epsilon)
        return np.log(mu / theta) + 1

    def project_to_simplex(self, prices: np.ndarray) -> np.ndarray:
        """
        Project prices onto probability simplex (sum = 1, all >= 0).

        For KL divergence with simplex constraint, the solution is normalization.
        """
        prices = np.clip(prices, self.epsilon, None)
        return prices / np.sum(prices)

    def project_binary_market(
        self,
        yes_price: float,
        no_price: float
    ) -> Tuple[float, float, float]:
        """
        Project binary market prices to arbitrage-free state.

        For binary market: YES + NO must equal 1.

        Returns:
            (projected_yes, projected_no, arbitrage_profit)
        """
        prices = np.array([yes_price, no_price])
        prices = np.clip(prices, self.epsilon, 1 - self.epsilon)

        # Project to simplex
        projected = self.project_to_simplex(prices)

        # Arbitrage profit is KL divergence
        profit = self.kl_divergence(projected, prices)

        return float(projected[0]), float(projected[1]), profit

    def project_multi_outcome(
        self,
        prices: list[float],
        constraint_type: str = "sum_to_one"
    ) -> ProjectionResult:
        """
        Project multi-outcome market prices to arbitrage-free state.

        Args:
            prices: List of outcome prices
            constraint_type: Type of constraint
                - "sum_to_one": All prices sum to 1 (mutually exclusive)
                - "at_most_one": At most one outcome true (with none possible)

        Returns:
            ProjectionResult with optimal prices and profit
        """
        theta = np.array(prices)
        theta = np.clip(theta, self.epsilon, 1 - self.epsilon)
        n = len(theta)

        if not SCIPY_AVAILABLE:
            # Fallback to simple normalization
            mu = self.project_to_simplex(theta)
            profit = self.kl_divergence(mu, theta)
            return ProjectionResult(
                projected_prices=mu,
                arbitrage_profit=profit,
                optimal_positions=self._compute_positions(mu, theta),
                converged=True,
                iterations=1
            )

        # Objective: minimize KL divergence D(μ||θ)
        def objective(mu):
            mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
            return np.sum(mu * np.log(mu / theta))

        def gradient(mu):
            mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
            return np.log(mu / theta) + 1

        # Initial guess: normalized prices
        mu0 = theta / np.sum(theta)

        # Constraints based on type
        if constraint_type == "sum_to_one":
            constraints = [
                {"type": "eq", "fun": lambda mu: np.sum(mu) - 1}
            ]
        elif constraint_type == "at_most_one":
            constraints = [
                {"type": "ineq", "fun": lambda mu: 1 - np.sum(mu)}  # sum <= 1
            ]
        else:
            constraints = []

        # Bounds: probabilities in [epsilon, 1-epsilon]
        bounds = [(self.epsilon, 1 - self.epsilon)] * n

        result = minimize(
            objective,
            mu0,
            method="SLSQP",
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        mu_star = result.x if result.success else mu0
        profit = self.kl_divergence(mu_star, theta)

        return ProjectionResult(
            projected_prices=mu_star,
            arbitrage_profit=profit,
            optimal_positions=self._compute_positions(mu_star, theta),
            converged=result.success,
            iterations=result.nit if hasattr(result, 'nit') else 0
        )

    def project_with_constraints(
        self,
        prices: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[np.ndarray] = None
    ) -> ProjectionResult:
        """
        Project prices onto polytope defined by linear constraints.

        Solves: argmin_μ D(μ||θ) s.t. A_eq @ μ = b_eq, A_ineq @ μ >= b_ineq

        This is the general form needed for combinatorial arbitrage with
        logical dependencies between markets.

        Args:
            prices: Current market prices θ
            A_eq: Equality constraint matrix
            b_eq: Equality constraint bounds
            A_ineq: Inequality constraint matrix (Aμ >= b)
            b_ineq: Inequality constraint bounds

        Returns:
            ProjectionResult with optimal prices and profit
        """
        if not SCIPY_AVAILABLE:
            mu = self.project_to_simplex(prices)
            return ProjectionResult(
                projected_prices=mu,
                arbitrage_profit=self.kl_divergence(mu, prices),
                optimal_positions=self._compute_positions(mu, prices),
                converged=False,
                iterations=0
            )

        theta = np.clip(prices, self.epsilon, 1 - self.epsilon)
        n = len(theta)

        def objective(mu):
            mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
            return np.sum(mu * np.log(mu / theta))

        def gradient(mu):
            mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
            return np.log(mu / theta) + 1

        # Build constraints
        constraints = []

        if A_eq is not None and b_eq is not None:
            for i in range(len(b_eq)):
                constraints.append({
                    "type": "eq",
                    "fun": lambda mu, i=i: A_eq[i] @ mu - b_eq[i]
                })

        if A_ineq is not None and b_ineq is not None:
            for i in range(len(b_ineq)):
                constraints.append({
                    "type": "ineq",
                    "fun": lambda mu, i=i: A_ineq[i] @ mu - b_ineq[i]
                })

        bounds = [(self.epsilon, 1 - self.epsilon)] * n
        mu0 = theta / np.sum(theta)

        result = minimize(
            objective,
            mu0,
            method="SLSQP",
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        mu_star = result.x if result.success else mu0
        profit = self.kl_divergence(mu_star, theta)

        return ProjectionResult(
            projected_prices=mu_star,
            arbitrage_profit=profit,
            optimal_positions=self._compute_positions(mu_star, theta),
            converged=result.success,
            iterations=result.nit if hasattr(result, 'nit') else 0
        )

    def _compute_positions(
        self,
        projected: np.ndarray,
        original: np.ndarray
    ) -> list[dict]:
        """
        Compute optimal trading positions from projection.

        The direction of trade is from original to projected prices.
        For LMSR, optimal shares = b × ln(μ* / θ)
        """
        positions = []
        direction = projected - original

        for i in range(len(projected)):
            if abs(direction[i]) < self.epsilon:
                continue

            # For LMSR: shares = b × ln(projected / original)
            ratio = projected[i] / max(original[i], self.epsilon)
            shares = self.b * np.log(ratio) if ratio > 0 else 0

            positions.append({
                "index": i,
                "action": "BUY" if shares > 0 else "SELL",
                "shares": abs(float(shares)),
                "price_change": float(direction[i]),
                "from_price": float(original[i]),
                "to_price": float(projected[i])
            })

        return positions

    def compute_optimal_trade(
        self,
        current_prices: list[float],
        token_ids: list[str],
        constraint_type: str = "sum_to_one"
    ) -> dict:
        """
        Compute the optimal arbitrage trade for a market.

        This is the main entry point for using Bregman projection
        in the trading system.

        Args:
            current_prices: Current market prices
            token_ids: Token IDs corresponding to each price
            constraint_type: Market constraint type

        Returns:
            dict with optimal trade details ready for execution
        """
        result = self.project_multi_outcome(current_prices, constraint_type)

        # Build executable positions
        executable = []
        for pos in result.optimal_positions:
            if pos["shares"] > 0.01:  # Filter dust
                executable.append({
                    "token_id": token_ids[pos["index"]],
                    "action": pos["action"],
                    "shares": pos["shares"],
                    "price": pos["from_price"]
                })

        return {
            "arbitrage_profit": result.arbitrage_profit,
            "projected_prices": result.projected_prices.tolist(),
            "converged": result.converged,
            "positions": executable,
            "total_cost": float(np.sum(current_prices)),
            "projected_cost": float(np.sum(result.projected_prices))
        }


# Singleton instance
bregman_projector = BregmanProjector()
