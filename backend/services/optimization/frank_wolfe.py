"""
Frank-Wolfe algorithm for Bregman projection onto marginal polytopes.

Computing Bregman projection directly is intractable when the marginal
polytope has exponentially many vertices (e.g., 2^63 for NCAA tournament).

Frank-Wolfe reduces this to a sequence of linear programs, building the
polytope iteratively through integer programming oracle calls.

Research showed 50-150 iterations sufficient for markets with thousands
of conditions. After 45 games settled in tournament, 30-minute projections
became feasible, and FWMM outperformed LCMM by 38% median improvement.

Key insight: Instead of enumerating all vertices, we find them on-demand
via the oracle call: z_t = argmin_{z∈Z} ∇F(μ_t)·z
"""

import numpy as np
from typing import Callable, Optional, List
from dataclasses import dataclass, field
import time

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class FrankWolfeResult:
    """Result of Frank-Wolfe optimization."""
    optimal_prices: np.ndarray
    arbitrage_profit: float
    iterations: int
    converged: bool
    active_vertices: int
    gap_history: List[float]
    solve_time_ms: float


class FrankWolfeSolver:
    """
    Frank-Wolfe algorithm with integer programming oracle.

    Solves: min_μ F(μ) s.t. μ ∈ M = conv(Z)

    Where Z is defined by integer constraints (valid market outcomes)
    and F is the Bregman divergence (KL divergence for LMSR).

    The algorithm builds M iteratively:
    1. Start with small set of vertices Z₀
    2. For each iteration:
       a. Solve convex optimization over conv(Z_{t-1})
       b. Find new descent vertex via IP oracle
       c. Add to active set
       d. Check convergence gap
    """

    def __init__(
        self,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        initial_contraction: float = 0.1,
        ip_timeout_ms: float = 30000.0
    ):
        """
        Args:
            max_iterations: Maximum Frank-Wolfe iterations
            convergence_threshold: Stop when gap < threshold
            initial_contraction: Initial barrier parameter ε
            ip_timeout_ms: Timeout for each IP oracle call
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.epsilon = initial_contraction
        self.ip_timeout_ms = ip_timeout_ms

    def solve(
        self,
        prices: np.ndarray,
        ip_oracle: Callable[[np.ndarray], np.ndarray],
        epsilon: float = 1e-10
    ) -> FrankWolfeResult:
        """
        Run Frank-Wolfe optimization.

        Args:
            prices: Current market prices θ
            ip_oracle: Function that solves min_{z∈Z} c·z
                      Takes gradient vector, returns optimal vertex
            epsilon: Small constant to prevent log(0)

        Returns:
            FrankWolfeResult with optimal projection
        """
        start_time = time.perf_counter()
        n = len(prices)
        theta = np.clip(prices, epsilon, 1 - epsilon)

        # KL divergence objective and gradient
        def objective(mu):
            mu = np.clip(mu, epsilon, 1 - epsilon)
            return np.sum(mu * np.log(mu / theta))

        def gradient(mu):
            mu = np.clip(mu, epsilon, 1 - epsilon)
            return np.log(mu / theta) + 1

        # Initialize with first vertex from oracle
        z0 = ip_oracle(np.zeros(n))
        if z0 is None:
            z0 = np.ones(n) / n
        active_set = [z0.astype(float)]

        # Current iterate
        mu = z0.copy().astype(float)

        # Barrier parameter for controlled gradient growth
        barrier_epsilon = self.epsilon
        interior_point = np.ones(n) / n

        gap_history = []

        for t in range(self.max_iterations):
            # Contracted iterate for gradient evaluation (Barrier FW)
            mu_contracted = (1 - barrier_epsilon) * mu + barrier_epsilon * interior_point

            # Compute gradient
            grad = gradient(mu_contracted)

            # Oracle call: find vertex minimizing gradient
            z_new = ip_oracle(grad)
            if z_new is None:
                break

            # Compute Frank-Wolfe gap
            gap = float(np.dot(grad, mu - z_new))
            gap_history.append(gap)

            # Check convergence
            if gap < self.convergence_threshold:
                return FrankWolfeResult(
                    optimal_prices=mu,
                    arbitrage_profit=objective(mu),
                    iterations=t + 1,
                    converged=True,
                    active_vertices=len(active_set),
                    gap_history=gap_history,
                    solve_time_ms=(time.perf_counter() - start_time) * 1000
                )

            # Add new vertex if novel
            is_novel = True
            for z in active_set:
                if np.allclose(z, z_new, atol=1e-8):
                    is_novel = False
                    break
            if is_novel:
                active_set.append(z_new.astype(float))

            # Solve subproblem over convex hull of active set
            mu = self._solve_subproblem(active_set, theta, objective, epsilon)

            # Adaptive epsilon reduction (Barrier Frank-Wolfe)
            g_u = float(np.dot(gradient(interior_point), interior_point - z_new))
            if g_u < 0 and gap / (-4 * g_u) < barrier_epsilon:
                barrier_epsilon = min(gap / (-4 * g_u), barrier_epsilon / 2)

        return FrankWolfeResult(
            optimal_prices=mu,
            arbitrage_profit=objective(mu),
            iterations=self.max_iterations,
            converged=False,
            active_vertices=len(active_set),
            gap_history=gap_history,
            solve_time_ms=(time.perf_counter() - start_time) * 1000
        )

    def _solve_subproblem(
        self,
        active_set: List[np.ndarray],
        theta: np.ndarray,
        objective: Callable[[np.ndarray], float],
        epsilon: float
    ) -> np.ndarray:
        """
        Solve convex optimization over convex hull of active vertices.

        min_λ F(Σ λ_i z_i) s.t. Σλ_i = 1, λ_i ≥ 0
        """
        from scipy.optimize import minimize

        k = len(active_set)
        if k == 1:
            return active_set[0]

        Z = np.array(active_set)  # k x n matrix

        def objective_lambda(lam):
            lam = np.clip(lam, epsilon, None)
            lam = lam / np.sum(lam)  # Normalize
            mu = Z.T @ lam
            return objective(mu)

        # Initial: uniform weights
        lam0 = np.ones(k) / k

        # Constraints: sum to 1, non-negative
        constraints = [{"type": "eq", "fun": lambda lam: np.sum(lam) - 1}]
        bounds = [(epsilon, 1)] * k

        result = minimize(
            objective_lambda,
            lam0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 100}
        )

        if result.success:
            lam = np.clip(result.x, epsilon, None)
            lam = lam / np.sum(lam)
            return Z.T @ lam

        return Z.T @ lam0


class IPOracle:
    """
    Integer Programming oracle for Frank-Wolfe.

    Solves: min_{z∈Z} c·z where Z = {z ∈ {0,1}^n : A^T z ≥ b}

    This is called at each Frank-Wolfe iteration to find the next
    descent direction.
    """

    def __init__(
        self,
        constraint_matrix: np.ndarray,
        constraint_bounds: np.ndarray,
        is_equality: Optional[np.ndarray] = None,
        solver: str = "auto"
    ):
        """
        Args:
            constraint_matrix: A in Az ≥ b
            constraint_bounds: b in Az ≥ b
            is_equality: Boolean array indicating equality constraints
            solver: "cvxpy", "scipy", or "auto"
        """
        self.A = constraint_matrix
        self.b = constraint_bounds
        self.is_eq = is_equality if is_equality is not None else np.zeros(len(constraint_bounds), dtype=bool)

        if solver == "auto":
            self.solver = "cvxpy" if CVXPY_AVAILABLE else "scipy"
        else:
            self.solver = solver

    def __call__(self, c: np.ndarray) -> Optional[np.ndarray]:
        """
        Find vertex minimizing c·z subject to constraints.

        Args:
            c: Gradient vector (cost coefficients)

        Returns:
            Optimal binary vector z, or None on failure
        """
        if self.solver == "cvxpy" and CVXPY_AVAILABLE:
            return self._solve_cvxpy(c)
        else:
            return self._solve_fallback(c)

    def _solve_cvxpy(self, c: np.ndarray) -> Optional[np.ndarray]:
        """Solve using CVXPY."""
        n = len(c)
        z = cp.Variable(n, boolean=True)

        constraints = []
        for i in range(len(self.b)):
            if self.is_eq[i]:
                constraints.append(self.A[i] @ z == self.b[i])
            else:
                constraints.append(self.A[i] @ z >= self.b[i])

        objective = cp.Minimize(c @ z)
        problem = cp.Problem(objective, constraints)

        try:
            # Try Gurobi first, then GLPK
            try:
                problem.solve(solver=cp.GUROBI, verbose=False)
            except:
                try:
                    problem.solve(solver=cp.GLPK_MI, verbose=False)
                except:
                    problem.solve(verbose=False)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return np.round(z.value).astype(float)
        except:
            pass

        return self._solve_fallback(c)

    def _solve_fallback(self, c: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback: enumerate for small problems, greedy for large.
        """
        n = len(c)

        if n <= 15:
            # Enumerate all 2^n combinations
            best_cost = float('inf')
            best_z = None

            for i in range(2**n):
                z = np.array([(i >> j) & 1 for j in range(n)], dtype=float)

                # Check constraints
                feasible = True
                for k in range(len(self.b)):
                    val = self.A[k] @ z
                    if self.is_eq[k]:
                        if abs(val - self.b[k]) > 1e-6:
                            feasible = False
                            break
                    else:
                        if val < self.b[k] - 1e-6:
                            feasible = False
                            break

                if feasible:
                    cost = c @ z
                    if cost < best_cost:
                        best_cost = cost
                        best_z = z

            return best_z

        else:
            # Greedy: select lowest cost items that satisfy constraints
            # This is a heuristic and may not find optimal
            sorted_idx = np.argsort(c)
            z = np.zeros(n)

            for idx in sorted_idx:
                z[idx] = 1
                # Check if still feasible
                feasible = True
                for k in range(len(self.b)):
                    val = self.A[k] @ z
                    if self.is_eq[k]:
                        if val > self.b[k]:
                            feasible = False
                            break
                    # For inequality, we want Az >= b, so no issue adding
                if not feasible:
                    z[idx] = 0

            return z


def create_binary_market_oracle(n_outcomes: int) -> IPOracle:
    """
    Create IP oracle for a simple n-outcome market.

    Constraint: exactly one outcome must be true (sum = 1)
    """
    # Single equality constraint: sum(z) = 1
    A = np.ones((1, n_outcomes))
    b = np.array([1.0])
    is_eq = np.array([True])

    return IPOracle(A, b, is_eq)


def create_cross_market_oracle(
    n_a: int,
    n_b: int,
    dependencies: List[tuple]
) -> IPOracle:
    """
    Create IP oracle for cross-market arbitrage.

    Args:
        n_a: Outcomes in market A
        n_b: Outcomes in market B
        dependencies: List of (idx_a, idx_b, type) tuples
                     type: "implies" or "excludes"
    """
    n_total = n_a + n_b
    constraints = []
    bounds = []
    is_eq = []

    # Exactly one in A
    row_a = np.zeros(n_total)
    row_a[:n_a] = 1
    constraints.append(row_a)
    bounds.append(1)
    is_eq.append(True)

    # Exactly one in B
    row_b = np.zeros(n_total)
    row_b[n_a:] = 1
    constraints.append(row_b)
    bounds.append(1)
    is_eq.append(True)

    # Add dependency constraints
    for idx_a, idx_b, dep_type in dependencies:
        row = np.zeros(n_total)
        if dep_type == "implies":
            # z_a <= z_b => -z_a + z_b >= 0
            row[idx_a] = -1
            row[n_a + idx_b] = 1
            bounds.append(0)
        else:  # excludes
            # z_a + z_b <= 1 => -z_a - z_b >= -1
            row[idx_a] = -1
            row[n_a + idx_b] = -1
            bounds.append(-1)
        constraints.append(row)
        is_eq.append(False)

    return IPOracle(
        np.array(constraints),
        np.array(bounds),
        np.array(is_eq)
    )


# Singleton solver
frank_wolfe_solver = FrankWolfeSolver()
