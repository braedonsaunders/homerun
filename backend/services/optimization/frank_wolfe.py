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

Improvements from Kroer et al. Part 2:
- Algorithm 3 (InitFW): Proper initialization via IP feasibility checks
- Barrier Frank-Wolfe with adaptive contraction (ε → 0)
- Profit guarantee stopping: D(μ̂||θ) - g(μ̂) with α-extraction (α=0.9)
- Settlement-aware optimization with partial outcome extension
- Best iterate tracking for forced interruption recovery
"""

import numpy as np
from typing import Callable, Optional, List
from dataclasses import dataclass
import time

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class InitFWResult:
    """Result of Algorithm 3: InitFW initialization."""

    active_vertices: List[np.ndarray]
    interior_point: np.ndarray
    settled_securities: dict[int, int]  # index -> 0 or 1
    unsettled_indices: List[int]
    init_time_ms: float


@dataclass
class FrankWolfeResult:
    """Result of Frank-Wolfe optimization."""

    optimal_prices: np.ndarray
    arbitrage_profit: float
    guaranteed_profit: float  # D(μ̂||θ) - g(μ̂), always >= 0
    iterations: int
    converged: bool
    active_vertices: int
    gap_history: List[float]
    profit_history: List[float]  # Guaranteed profit at each iteration
    epsilon_history: List[float]  # Contraction parameter over time
    best_iterate_idx: int  # Iteration with max guaranteed profit
    solve_time_ms: float
    alpha_extraction_met: bool  # Whether α-extraction stopping was reached
    settled_securities: dict[int, int]  # From InitFW


class FrankWolfeSolver:
    """
    Barrier Frank-Wolfe with InitFW initialization and profit guarantees.

    Solves: min_μ F(μ) s.t. μ ∈ M = conv(Z)

    Where Z is defined by integer constraints (valid market outcomes)
    and F is the Bregman divergence (KL divergence for LMSR).

    Three stopping conditions (Proposition 4.1):
    1. α-extraction: g(μₜ) ≤ (1-α) × D(μₜ||θ), captures α% of profit
    2. Near-arbitrage-free: D(μₜ||θ) ≤ εD, skip small opportunities
    3. Forced interruption: Return best iterate t* with max guaranteed profit
    """

    def __init__(
        self,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        initial_contraction: float = 0.1,
        ip_timeout_ms: float = 30000.0,
        alpha_extraction: float = 0.9,
        min_profit_threshold: float = 0.05,
    ):
        """
        Args:
            max_iterations: Maximum Frank-Wolfe iterations
            convergence_threshold: Stop when gap < threshold
            initial_contraction: Initial barrier parameter ε
            ip_timeout_ms: Timeout for each IP oracle call
            alpha_extraction: Fraction of profit to capture before stopping (0.9 = 90%)
            min_profit_threshold: Minimum D(μ||θ) to bother trading ($0.05 from research)
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.epsilon = initial_contraction
        self.ip_timeout_ms = ip_timeout_ms
        self.alpha = alpha_extraction
        self.min_profit_threshold = min_profit_threshold

    def init_fw(
        self,
        n_securities: int,
        ip_oracle: "IPOracle",
        settled: Optional[dict[int, int]] = None,
        epsilon: float = 1e-10,
    ) -> InitFWResult:
        """
        Algorithm 3: InitFW - Proper initialization for Frank-Wolfe.

        For each unsettled security i:
        1. Ask IP: "Can z_i = 1?" → if yes, add vertex to Z_0
        2. Ask IP: "Can z_i = 0?" → if yes, add vertex to Z_0
        3. If only one is feasible → security is logically settled

        Constructs interior point u = average(Z_0), guaranteeing
        all unsettled coordinates are strictly in (0, 1).

        Args:
            n_securities: Total number of securities
            ip_oracle: IP oracle with market constraints
            settled: Already-settled securities {index: 0 or 1}
            epsilon: Small constant for numerical stability

        Returns:
            InitFWResult with vertices, interior point, and extended settlements
        """
        start_time = time.perf_counter()

        if settled is None:
            settled = {}

        extended_settled = dict(settled)
        active_vertices: List[np.ndarray] = []
        unsettled = [i for i in range(n_securities) if i not in settled]

        for i in unsettled:
            # Question 1: Can z_i = 1?
            z_one = ip_oracle.check_feasibility(i, 1)

            # Question 2: Can z_i = 0?
            z_zero = ip_oracle.check_feasibility(i, 0)

            if z_one is not None and z_zero is not None:
                # Case 1: Both feasible - security is genuinely uncertain
                active_vertices.append(z_one.astype(float))
                active_vertices.append(z_zero.astype(float))
            elif z_one is not None and z_zero is None:
                # Case 2: Only z_i=1 feasible - must resolve to 1
                extended_settled[i] = 1
                active_vertices.append(z_one.astype(float))
            elif z_zero is not None and z_one is None:
                # Case 3: Only z_i=0 feasible - must resolve to 0
                extended_settled[i] = 0
                active_vertices.append(z_zero.astype(float))
            # else: neither feasible (shouldn't happen with valid constraints)

        # Deduplicate vertices
        unique_vertices: List[np.ndarray] = []
        for v in active_vertices:
            is_dup = False
            for uv in unique_vertices:
                if np.allclose(v, uv, atol=1e-8):
                    is_dup = True
                    break
            if not is_dup:
                unique_vertices.append(v)

        # Construct interior point as average of all vertices
        if unique_vertices:
            interior_point = np.mean(unique_vertices, axis=0)
        else:
            # Fallback: uniform over unsettled
            interior_point = np.full(n_securities, 0.5)
            for idx, val in extended_settled.items():
                interior_point[idx] = val

        # Verify interior point: all unsettled coords must be in (0, 1)
        remaining_unsettled = [
            i for i in range(n_securities) if i not in extended_settled
        ]
        for i in remaining_unsettled:
            if interior_point[i] <= epsilon or interior_point[i] >= 1 - epsilon:
                interior_point[i] = 0.5  # Safety clamp

        init_time = (time.perf_counter() - start_time) * 1000

        return InitFWResult(
            active_vertices=unique_vertices,
            interior_point=interior_point,
            settled_securities=extended_settled,
            unsettled_indices=remaining_unsettled,
            init_time_ms=init_time,
        )

    def solve(
        self,
        prices: np.ndarray,
        ip_oracle: Callable[[np.ndarray], np.ndarray],
        epsilon: float = 1e-10,
        settled: Optional[dict[int, int]] = None,
        time_limit_ms: Optional[float] = None,
    ) -> FrankWolfeResult:
        """
        Run Barrier Frank-Wolfe with profit guarantees.

        Uses InitFW for proper initialization, adaptive contraction to
        prevent gradient explosion, and profit-guarantee stopping conditions.

        Args:
            prices: Current market prices θ
            ip_oracle: IP oracle (IPOracle instance or callable)
            epsilon: Small constant to prevent log(0)
            settled: Already-settled securities for InitFW
            time_limit_ms: Optional time limit for forced interruption

        Returns:
            FrankWolfeResult with optimal projection and profit guarantees
        """
        start_time = time.perf_counter()
        n = len(prices)
        theta = np.clip(prices, epsilon, 1 - epsilon)

        # ---- Phase 1: InitFW Initialization ----
        init_result = None
        if isinstance(ip_oracle, IPOracle):
            init_result = self.init_fw(n, ip_oracle, settled, epsilon)
            active_set = list(init_result.active_vertices)
            interior_point = init_result.interior_point.copy()
            settled_securities = init_result.settled_securities
        else:
            # Legacy callable oracle - use simple initialization
            z0 = ip_oracle(np.zeros(n))
            if z0 is None:
                z0 = np.ones(n) / n
            active_set = [z0.astype(float)]
            interior_point = np.ones(n) / n
            settled_securities = settled or {}

        if not active_set:
            # No valid vertices found
            return FrankWolfeResult(
                optimal_prices=theta,
                arbitrage_profit=0.0,
                guaranteed_profit=0.0,
                iterations=0,
                converged=True,
                active_vertices=0,
                gap_history=[],
                profit_history=[],
                epsilon_history=[],
                best_iterate_idx=0,
                solve_time_ms=(time.perf_counter() - start_time) * 1000,
                alpha_extraction_met=False,
                settled_securities=settled_securities,
            )

        # ---- KL divergence objective and gradient ----
        def objective(mu):
            mu = np.clip(mu, epsilon, 1 - epsilon)
            return float(np.sum(mu * np.log(mu / theta)))

        def gradient(mu):
            mu = np.clip(mu, epsilon, 1 - epsilon)
            return np.log(mu / theta) + 1

        # ---- Phase 2: Barrier Frank-Wolfe with Adaptive Contraction ----
        mu = (
            np.mean(active_set, axis=0) if len(active_set) > 1 else active_set[0].copy()
        )
        barrier_epsilon = self.epsilon

        gap_history: List[float] = []
        profit_history: List[float] = []
        epsilon_history: List[float] = []
        best_iterate_idx = 0
        best_guaranteed_profit = -float("inf")

        oracle_fn = (
            ip_oracle
            if callable(ip_oracle) and not isinstance(ip_oracle, IPOracle)
            else ip_oracle
        )

        for t in range(self.max_iterations):
            # Check time limit for forced interruption
            if time_limit_ms is not None:
                elapsed = (time.perf_counter() - start_time) * 1000
                if elapsed > time_limit_ms:
                    break

            # Contract iterate toward interior point (Barrier FW)
            # M' = (1 - ε)M + εu keeps all coords away from 0/1
            mu_contracted = (
                1 - barrier_epsilon
            ) * mu + barrier_epsilon * interior_point

            # Compute gradient on contracted iterate
            grad = gradient(mu_contracted)

            # Oracle call: find descent vertex
            z_new = oracle_fn(grad)
            if z_new is None:
                break

            # ---- Frank-Wolfe gap ----
            gap = float(np.dot(grad, mu - z_new))
            gap_history.append(gap)
            epsilon_history.append(barrier_epsilon)

            # ---- Profit guarantee (Proposition 4.1) ----
            # Guaranteed profit = D(μ̂||θ) - g(μ̂)
            divergence = objective(mu)
            guaranteed_profit = divergence - gap
            profit_history.append(guaranteed_profit)

            # Track best iterate for forced interruption
            if guaranteed_profit > best_guaranteed_profit:
                best_guaranteed_profit = guaranteed_profit
                best_iterate_idx = t
                best_mu = mu.copy()

            # ---- Stopping Condition 1: α-extraction ----
            # g(μₜ) ≤ (1-α) × D(μₜ||θ) means we capture α% of profit
            if divergence > epsilon and gap <= (1 - self.alpha) * divergence:
                return FrankWolfeResult(
                    optimal_prices=mu,
                    arbitrage_profit=divergence,
                    guaranteed_profit=guaranteed_profit,
                    iterations=t + 1,
                    converged=True,
                    active_vertices=len(active_set),
                    gap_history=gap_history,
                    profit_history=profit_history,
                    epsilon_history=epsilon_history,
                    best_iterate_idx=best_iterate_idx,
                    solve_time_ms=(time.perf_counter() - start_time) * 1000,
                    alpha_extraction_met=True,
                    settled_securities=settled_securities,
                )

            # ---- Stopping Condition 2: Near-arbitrage-free ----
            # D(μₜ||θ) ≤ εD means not enough profit to bother
            if divergence <= self.min_profit_threshold:
                return FrankWolfeResult(
                    optimal_prices=mu,
                    arbitrage_profit=divergence,
                    guaranteed_profit=max(0, guaranteed_profit),
                    iterations=t + 1,
                    converged=True,
                    active_vertices=len(active_set),
                    gap_history=gap_history,
                    profit_history=profit_history,
                    epsilon_history=epsilon_history,
                    best_iterate_idx=best_iterate_idx,
                    solve_time_ms=(time.perf_counter() - start_time) * 1000,
                    alpha_extraction_met=False,
                    settled_securities=settled_securities,
                )

            # ---- Standard gap convergence ----
            if gap < self.convergence_threshold:
                return FrankWolfeResult(
                    optimal_prices=mu,
                    arbitrage_profit=divergence,
                    guaranteed_profit=max(0, guaranteed_profit),
                    iterations=t + 1,
                    converged=True,
                    active_vertices=len(active_set),
                    gap_history=gap_history,
                    profit_history=profit_history,
                    epsilon_history=epsilon_history,
                    best_iterate_idx=best_iterate_idx,
                    solve_time_ms=(time.perf_counter() - start_time) * 1000,
                    alpha_extraction_met=False,
                    settled_securities=settled_securities,
                )

            # ---- Add new vertex if novel ----
            is_novel = True
            for z in active_set:
                if np.allclose(z, z_new, atol=1e-8):
                    is_novel = False
                    break
            if is_novel:
                active_set.append(z_new.astype(float))

            # ---- Solve subproblem over convex hull ----
            mu = self._solve_subproblem(active_set, theta, objective, epsilon)

            # ---- Adaptive epsilon reduction ----
            # From Kroer et al.: ε shrinks when gap/(−4g_u) < ε
            g_u = float(np.dot(gradient(interior_point), interior_point - z_new))
            if g_u < 0:
                ratio = gap / (-4 * g_u)
                if ratio < barrier_epsilon:
                    barrier_epsilon = min(ratio, barrier_epsilon / 2)

        # ---- Stopping Condition 3: Forced interruption ----
        # Return best iterate found across all iterations
        final_divergence = objective(mu)
        final_gap = gap_history[-1] if gap_history else 0
        final_profit = final_divergence - final_gap

        # Use best iterate if it was better than the final one
        if best_guaranteed_profit > final_profit and "best_mu" in dir():
            mu = best_mu
            final_divergence = objective(mu)
            final_profit = best_guaranteed_profit

        return FrankWolfeResult(
            optimal_prices=mu,
            arbitrage_profit=final_divergence,
            guaranteed_profit=max(0, final_profit),
            iterations=self.max_iterations,
            converged=False,
            active_vertices=len(active_set),
            gap_history=gap_history,
            profit_history=profit_history,
            epsilon_history=epsilon_history,
            best_iterate_idx=best_iterate_idx,
            solve_time_ms=(time.perf_counter() - start_time) * 1000,
            alpha_extraction_met=False,
            settled_securities=settled_securities,
        )

    def should_execute_trade(
        self, result: FrankWolfeResult, execution_cost: float = 0.02
    ) -> dict:
        """
        Apply Proposition 4.1 to determine if a trade should execute.

        Args:
            result: FrankWolfeResult from solve()
            execution_cost: Estimated execution cost (gas + slippage, default $0.02)

        Returns:
            dict with trade decision and reasoning
        """
        d = result.arbitrage_profit
        g = result.gap_history[-1] if result.gap_history else float("inf")
        guaranteed = result.guaranteed_profit

        # Decision matrix from article Part III
        if d < self.min_profit_threshold:
            return {
                "should_trade": False,
                "reason": f"Near-arbitrage-free: D={d:.6f} < threshold {self.min_profit_threshold}",
                "guaranteed_profit": guaranteed,
                "capture_ratio": 0,
            }

        if guaranteed <= 0:
            return {
                "should_trade": False,
                "reason": f"Gap exceeds divergence: D={d:.6f}, g={g:.6f}",
                "guaranteed_profit": guaranteed,
                "capture_ratio": 0,
            }

        if guaranteed < execution_cost:
            return {
                "should_trade": False,
                "reason": f"Profit ${guaranteed:.4f} below execution cost ${execution_cost:.4f}",
                "guaranteed_profit": guaranteed,
                "capture_ratio": guaranteed / d if d > 0 else 0,
            }

        capture_ratio = guaranteed / d if d > 0 else 0
        return {
            "should_trade": True,
            "reason": f"Guaranteed profit ${guaranteed:.4f} ({capture_ratio:.0%} of max ${d:.4f})",
            "guaranteed_profit": guaranteed,
            "capture_ratio": capture_ratio,
            "alpha_met": result.alpha_extraction_met,
        }

    def _solve_subproblem(
        self,
        active_set: List[np.ndarray],
        theta: np.ndarray,
        objective: Callable[[np.ndarray], float],
        epsilon: float,
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
            options={"maxiter": 100},
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
    descent direction. Also supports feasibility checks for InitFW.
    """

    def __init__(
        self,
        constraint_matrix: np.ndarray,
        constraint_bounds: np.ndarray,
        is_equality: Optional[np.ndarray] = None,
        solver: str = "auto",
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
        self.is_eq = (
            is_equality
            if is_equality is not None
            else np.zeros(len(constraint_bounds), dtype=bool)
        )

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

    def check_feasibility(self, index: int, value: int) -> Optional[np.ndarray]:
        """
        InitFW feasibility check: find any valid z where z[index] = value.

        This is the core of Algorithm 3. For each security, we ask:
        "Does a valid outcome exist where this security = value?"

        Args:
            index: Security index
            value: Required value (0 or 1)

        Returns:
            A feasible binary vector, or None if infeasible
        """
        n = self.A.shape[1]

        if self.solver == "cvxpy" and CVXPY_AVAILABLE:
            return self._check_feasibility_cvxpy(n, index, value)
        else:
            return self._check_feasibility_fallback(n, index, value)

    def _check_feasibility_cvxpy(
        self, n: int, index: int, value: int
    ) -> Optional[np.ndarray]:
        """Check feasibility using CVXPY."""
        z = cp.Variable(n, boolean=True)

        constraints = []
        for i in range(len(self.b)):
            if self.is_eq[i]:
                constraints.append(self.A[i] @ z == self.b[i])
            else:
                constraints.append(self.A[i] @ z >= self.b[i])

        # Fix z[index] = value
        constraints.append(z[index] == value)

        # Feasibility: minimize 0 (just check if constraints satisfiable)
        objective = cp.Minimize(0)
        problem = cp.Problem(objective, constraints)

        try:
            try:
                problem.solve(solver=cp.GUROBI, verbose=False)
            except Exception:
                try:
                    problem.solve(solver=cp.GLPK_MI, verbose=False)
                except Exception:
                    problem.solve(verbose=False)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return np.round(z.value).astype(float)
        except Exception:
            pass

        return self._check_feasibility_fallback(n, index, value)

    def _check_feasibility_fallback(
        self, n: int, index: int, value: int
    ) -> Optional[np.ndarray]:
        """Fallback feasibility check via enumeration or greedy."""
        if n <= 15:
            for i in range(2**n):
                z = np.array([(i >> j) & 1 for j in range(n)], dtype=float)
                if z[index] != value:
                    continue

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
                    return z

        else:
            # Greedy: construct a feasible solution with z[index] = value
            z = np.zeros(n)
            z[index] = value

            for j in range(n):
                if j == index:
                    continue
                # Try setting z[j] = 1 and check if still feasible
                z[j] = 1
                feasible = True
                for k in range(len(self.b)):
                    val = self.A[k] @ z
                    if self.is_eq[k]:
                        if val > self.b[k] + 1e-6:
                            feasible = False
                    # For inequality we need sum >= b eventually
                if not feasible:
                    z[j] = 0

            # Verify full feasibility
            for k in range(len(self.b)):
                val = self.A[k] @ z
                if self.is_eq[k]:
                    if abs(val - self.b[k]) > 1e-6:
                        return None
                else:
                    if val < self.b[k] - 1e-6:
                        return None
            return z

        return None

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
            except Exception:
                try:
                    problem.solve(solver=cp.GLPK_MI, verbose=False)
                except Exception:
                    problem.solve(verbose=False)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return np.round(z.value).astype(float)
        except Exception:
            pass

        return self._solve_fallback(c)

    def _solve_fallback(self, c: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback: enumerate for small problems, greedy for large.
        """
        n = len(c)

        if n <= 15:
            # Enumerate all 2^n combinations
            best_cost = float("inf")
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
    n_a: int, n_b: int, dependencies: List[tuple]
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

    return IPOracle(np.array(constraints), np.array(bounds), np.array(is_eq))


# Singleton solver with research-paper defaults
frank_wolfe_solver = FrankWolfeSolver(
    alpha_extraction=0.9,
    min_profit_threshold=0.05,
)
