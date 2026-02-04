"""
Integer Programming based arbitrage detection.

For markets with logical dependencies, instead of checking 2^n combinations
(exponentially many), we describe valid outcomes with linear constraints:

Z = {z ∈ {0,1}^I : A^T × z ≥ b}

Then check if current prices lie within the marginal polytope M = conv(Z).
If prices are outside M, arbitrage exists.

Research findings:
- NCAA 2010 tournament: 63 games, 2^63 possible outcomes (impossible to enumerate)
- 3 linear constraints can replace 16,384 brute force checks
- 17,218 conditions examined, 41% showed single-market arbitrage
- 13 confirmed dependent market pairs with exploitable arbitrage
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Try to import optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class DependencyType(str, Enum):
    """Types of logical dependencies between market outcomes."""
    IMPLIES = "implies"      # A implies B: if A true, B must be true
    EXCLUDES = "excludes"    # A excludes B: both cannot be true
    CUMULATIVE = "cumulative"  # A true means all "by date" variants true


@dataclass
class Dependency:
    """Represents a logical dependency between two outcomes."""
    market_a_idx: int
    outcome_a_idx: int
    market_b_idx: int
    outcome_b_idx: int
    dep_type: DependencyType
    reason: str = ""


@dataclass
class ArbitrageResult:
    """Result of arbitrage detection via constraint solving."""
    arbitrage_found: bool
    profit: float
    optimal_outcome: Optional[np.ndarray]
    total_cost: float
    positions: list[dict]
    solver_status: str


class ConstraintSolver:
    """
    Solves arbitrage detection as integer programming problem.

    For n conditions with dependency constraints, instead of checking
    2^n combinations, we describe valid outcomes with linear constraints.

    The marginal polytope M = conv(Z) contains all arbitrage-free prices.
    If current prices are outside M, arbitrage exists.
    """

    def __init__(self, solver: str = "auto"):
        """
        Args:
            solver: Solver to use ("cvxpy", "scipy", or "auto")
        """
        if solver == "auto":
            if CVXPY_AVAILABLE:
                self.solver = "cvxpy"
            elif SCIPY_AVAILABLE:
                self.solver = "scipy"
            else:
                self.solver = "fallback"
        else:
            self.solver = solver

    def detect_arbitrage(
        self,
        prices: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_bounds: np.ndarray,
        is_equality: Optional[np.ndarray] = None
    ) -> ArbitrageResult:
        """
        Check if prices admit arbitrage given constraints.

        Solves: min c·z s.t. Az ≥ b, z ∈ {0,1}^n

        If optimal value < sum(prices), arbitrage exists because we can
        buy a complete coverage (one outcome must occur) for less than
        the current prices imply.

        Args:
            prices: Current market prices [n_conditions]
            constraint_matrix: A in Az ≥ b [n_constraints, n_conditions]
            constraint_bounds: b in Az ≥ b [n_constraints]
            is_equality: Boolean array indicating equality constraints

        Returns:
            ArbitrageResult with arbitrage details
        """
        if self.solver == "cvxpy" and CVXPY_AVAILABLE:
            return self._solve_cvxpy(prices, constraint_matrix, constraint_bounds, is_equality)
        elif self.solver == "scipy" and SCIPY_AVAILABLE:
            return self._solve_scipy(prices, constraint_matrix, constraint_bounds, is_equality)
        else:
            return self._solve_fallback(prices, constraint_matrix, constraint_bounds)

    def _solve_cvxpy(
        self,
        prices: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        is_eq: Optional[np.ndarray]
    ) -> ArbitrageResult:
        """Solve using CVXPY (supports Gurobi, GLPK, etc.)."""
        n = len(prices)

        z = cp.Variable(n, boolean=True)

        constraints = []
        if is_eq is not None:
            for i in range(len(b)):
                if is_eq[i]:
                    constraints.append(A[i] @ z == b[i])
                else:
                    constraints.append(A[i] @ z >= b[i])
        else:
            constraints.append(A @ z >= b)

        # Objective: find minimum cost to cover all outcomes
        objective = cp.Minimize(prices @ z)

        problem = cp.Problem(objective, constraints)

        try:
            # Try Gurobi first, fall back to GLPK
            try:
                problem.solve(solver=cp.GUROBI, verbose=False)
            except:
                try:
                    problem.solve(solver=cp.GLPK_MI, verbose=False)
                except:
                    problem.solve(verbose=False)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                optimal_value = float(problem.value)
                z_star = np.round(z.value).astype(int)

                # Arbitrage exists if we can buy coverage for < $1
                if optimal_value < 1.0 - 1e-6:
                    return ArbitrageResult(
                        arbitrage_found=True,
                        profit=1.0 - optimal_value,
                        optimal_outcome=z_star,
                        total_cost=optimal_value,
                        positions=self._build_positions(z_star, prices),
                        solver_status=problem.status
                    )

            return ArbitrageResult(
                arbitrage_found=False,
                profit=0.0,
                optimal_outcome=None,
                total_cost=float(np.sum(prices)),
                positions=[],
                solver_status=problem.status
            )

        except Exception as e:
            return ArbitrageResult(
                arbitrage_found=False,
                profit=0.0,
                optimal_outcome=None,
                total_cost=float(np.sum(prices)),
                positions=[],
                solver_status=f"error: {str(e)}"
            )

    def _solve_scipy(
        self,
        prices: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        is_eq: Optional[np.ndarray]
    ) -> ArbitrageResult:
        """Solve using SciPy's MILP solver."""
        n = len(prices)

        try:
            # SciPy MILP expects: minimize c @ x
            # subject to: A_ub @ x <= b_ub (upper bound constraints)
            # We have A @ z >= b, so convert: -A @ z <= -b

            from scipy.optimize import milp, LinearConstraint, Bounds

            # All variables are binary
            integrality = np.ones(n)
            bounds = Bounds(lb=0, ub=1)

            # Convert >= to <=
            constraints = LinearConstraint(-A, -np.inf, -b)

            result = milp(
                c=prices,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds
            )

            if result.success:
                optimal_value = result.fun
                z_star = np.round(result.x).astype(int)

                if optimal_value < 1.0 - 1e-6:
                    return ArbitrageResult(
                        arbitrage_found=True,
                        profit=1.0 - optimal_value,
                        optimal_outcome=z_star,
                        total_cost=optimal_value,
                        positions=self._build_positions(z_star, prices),
                        solver_status="optimal"
                    )

            return ArbitrageResult(
                arbitrage_found=False,
                profit=0.0,
                optimal_outcome=None,
                total_cost=float(np.sum(prices)),
                positions=[],
                solver_status=result.message if hasattr(result, 'message') else "unknown"
            )

        except Exception as e:
            return self._solve_fallback(prices, A, b)

    def _solve_fallback(
        self,
        prices: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> ArbitrageResult:
        """Fallback solver using simple enumeration for small problems."""
        n = len(prices)

        # Only feasible for small n
        if n > 20:
            return ArbitrageResult(
                arbitrage_found=False,
                profit=0.0,
                optimal_outcome=None,
                total_cost=float(np.sum(prices)),
                positions=[],
                solver_status="fallback: problem too large"
            )

        best_cost = float('inf')
        best_z = None

        # Enumerate all 2^n combinations
        for i in range(2**n):
            z = np.array([(i >> j) & 1 for j in range(n)])

            # Check constraints
            if np.all(A @ z >= b):
                cost = prices @ z
                if cost < best_cost:
                    best_cost = cost
                    best_z = z

        if best_z is not None and best_cost < 1.0 - 1e-6:
            return ArbitrageResult(
                arbitrage_found=True,
                profit=1.0 - best_cost,
                optimal_outcome=best_z,
                total_cost=best_cost,
                positions=self._build_positions(best_z, prices),
                solver_status="fallback: enumeration"
            )

        return ArbitrageResult(
            arbitrage_found=False,
            profit=0.0,
            optimal_outcome=None,
            total_cost=float(np.sum(prices)),
            positions=[],
            solver_status="fallback: no arbitrage found"
        )

    def _build_positions(self, z: np.ndarray, prices: np.ndarray) -> list[dict]:
        """Build trading positions from optimal solution."""
        positions = []
        for i in range(len(z)):
            if z[i] == 1:
                positions.append({
                    "index": i,
                    "action": "BUY",
                    "price": float(prices[i])
                })
        return positions

    def build_constraints_from_dependencies(
        self,
        n_outcomes_a: int,
        n_outcomes_b: int,
        dependencies: List[Dependency]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build constraint matrix from logical dependencies.

        Args:
            n_outcomes_a: Number of outcomes in market A
            n_outcomes_b: Number of outcomes in market B
            dependencies: List of dependencies between outcomes

        Returns:
            (constraint_matrix, bounds, is_equality)

        Example constraints:
        - "Trump wins PA" implies "Republican wins PA":
          z_trump <= z_republican, or: -z_trump + z_republican >= 0

        - "Duke wins 6" excludes "Cornell wins 6":
          z_duke6 + z_cornell6 <= 1, or: -z_duke6 - z_cornell6 >= -1

        - Exactly one outcome per market:
          sum(z_market) = 1
        """
        n_total = n_outcomes_a + n_outcomes_b
        constraints = []
        bounds = []
        is_equality = []

        # Constraint: exactly one outcome per market A
        row_a = np.zeros(n_total)
        row_a[:n_outcomes_a] = 1
        constraints.append(row_a)
        bounds.append(1)
        is_equality.append(True)

        # Constraint: exactly one outcome per market B
        row_b = np.zeros(n_total)
        row_b[n_outcomes_a:] = 1
        constraints.append(row_b)
        bounds.append(1)
        is_equality.append(True)

        # Add dependency constraints
        for dep in dependencies:
            row = np.zeros(n_total)
            idx_a = dep.outcome_a_idx
            idx_b = n_outcomes_a + dep.outcome_b_idx

            if dep.dep_type == DependencyType.IMPLIES:
                # z_a <= z_b => -z_a + z_b >= 0
                row[idx_a] = -1
                row[idx_b] = 1
                bounds.append(0)
                is_equality.append(False)

            elif dep.dep_type == DependencyType.EXCLUDES:
                # z_a + z_b <= 1 => -z_a - z_b >= -1
                row[idx_a] = -1
                row[idx_b] = -1
                bounds.append(-1)
                is_equality.append(False)

            elif dep.dep_type == DependencyType.CUMULATIVE:
                # If z_a = 1, all later outcomes must also be 1
                # This requires more complex handling
                row[idx_a] = -1
                row[idx_b] = 1
                bounds.append(0)
                is_equality.append(False)

            constraints.append(row)

        return (
            np.array(constraints),
            np.array(bounds),
            np.array(is_equality)
        )

    def detect_cross_market_arbitrage(
        self,
        prices_a: list[float],
        prices_b: list[float],
        dependencies: List[Dependency]
    ) -> ArbitrageResult:
        """
        Detect arbitrage across two dependent markets.

        This is the main entry point for combinatorial arbitrage detection.

        Args:
            prices_a: Prices for market A outcomes
            prices_b: Prices for market B outcomes
            dependencies: Logical dependencies between markets

        Returns:
            ArbitrageResult with cross-market arbitrage details
        """
        n_a = len(prices_a)
        n_b = len(prices_b)

        A, b, is_eq = self.build_constraints_from_dependencies(
            n_a, n_b, dependencies
        )

        prices = np.concatenate([prices_a, prices_b])

        return self.detect_arbitrage(prices, A, b, is_eq)


# Singleton instance
constraint_solver = ConstraintSolver()
