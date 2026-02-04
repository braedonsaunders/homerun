# Quantitative Arbitrage Analysis: Lessons from $40M Polymarket Extraction

This document analyzes the homerun codebase against the sophisticated quantitative strategies described in the research paper "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets" (arXiv:2508.03474v1). The paper documents how sophisticated traders extracted $40 million in guaranteed arbitrage profits from Polymarket in one year.

## Executive Summary

The current homerun implementation provides a solid foundation for arbitrage detection but operates at a fundamentally different level than the quantitative systems that extracted $40M. The key gaps are:

| Area | Current Implementation | Quantitative Approach | Gap Severity |
|------|----------------------|----------------------|--------------|
| **Dependency Detection** | Keyword heuristics | Integer Programming + LLM | **Critical** |
| **Optimal Trade Calculation** | `profit = payout - cost` | Bregman Projection | **High** |
| **Computational Tractability** | Brute force (small scale) | Frank-Wolfe Algorithm | **High** |
| **Order Execution** | Sequential `await` | Parallel non-atomic | **High** |
| **Price Analysis** | Mid-point prices | VWAP with order book depth | **Medium** |
| **Position Sizing** | Fixed / Basic Kelly | Modified Kelly with execution risk | **Medium** |
| **Latency** | ~1000ms+ | <30ms decision-to-mempool | **Medium** |

---

## Part 1: The Marginal Polytope Problem

### What the Article Describes

For markets with logical dependencies, simple "YES + NO = $1" checks fail. The research found:
- **17,218 conditions** examined
- **7,051 conditions** (41%) showed single-market arbitrage
- **1,576 dependent market pairs** in 2024 US election alone
- **13 confirmed exploitable** cross-market arbitrage opportunities

The mathematical framework uses the **marginal polytope**:
```
M = conv(Z) where Z = {φ(ω) : ω ∈ Ω}
```

Arbitrage-free prices must lie within M. Anything outside is exploitable.

### Current homerun Implementation

**File:** `backend/services/strategies/contradiction.py`

```python
# Current: Keyword-based heuristics
CONTRADICTION_PAIRS = [
    ("before", "after"),
    ("win", "lose"),
    ("pass", "fail"),
    ("approve", "reject"),
]

def _are_contradictory(self, market_a: Market, market_b: Market) -> bool:
    q_a = market_a.question.lower()
    q_b = market_b.question.lower()

    for word_a, word_b in self.CONTRADICTION_PAIRS:
        if word_a in q_a and word_b in q_b:
            if self._share_topic(q_a, q_b):
                return True
    return False
```

**Problems:**
1. Can't detect complex logical dependencies ("Trump wins PA" implies "Republican wins PA")
2. Misses non-obvious correlations (Duke vs Cornell semi-final constraint)
3. False positives from keyword matching without semantic understanding
4. Scales O(n²) naively vs O(1) per constraint in IP formulation

### Recommended Improvements

#### 1.1 Add Integer Programming for Constraint Detection

**New File:** `backend/services/optimization/constraint_solver.py`

```python
"""
Integer Programming based arbitrage detection.
Uses Gurobi/CVXPY to solve constraint satisfaction over outcome spaces.
"""

import numpy as np
from typing import Optional
try:
    import cvxpy as cp
except ImportError:
    cp = None

class ConstraintSolver:
    """
    Solves arbitrage detection as integer programming problem.

    For n conditions with dependency constraints, instead of checking
    2^n combinations, we describe valid outcomes with linear constraints:

    Z = {z ∈ {0,1}^I : A^T × z ≥ b}

    Then check if current prices lie within the marginal polytope M = conv(Z).
    """

    def __init__(self):
        self.solver = "GLPK_MI" if cp else None

    def detect_arbitrage(
        self,
        prices: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_bounds: np.ndarray
    ) -> Optional[dict]:
        """
        Check if prices admit arbitrage given constraints.

        Args:
            prices: Current market prices [n_conditions]
            constraint_matrix: A in A^T z >= b [n_constraints, n_conditions]
            constraint_bounds: b in A^T z >= b [n_constraints]

        Returns:
            dict with arbitrage details if found, None otherwise
        """
        if cp is None:
            return self._fallback_detect(prices, constraint_matrix, constraint_bounds)

        n = len(prices)

        # Binary variables for each outcome
        z = cp.Variable(n, boolean=True)

        # Valid outcome constraints
        constraints = [
            constraint_matrix @ z >= constraint_bounds,
            cp.sum(z) == 1  # Exactly one outcome occurs
        ]

        # Objective: find if prices are outside polytope
        # If we can find z where price·z < 0 for buying all, arbitrage exists
        objective = cp.Minimize(prices @ z)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver, verbose=False)

            if problem.status == cp.OPTIMAL:
                optimal_value = problem.value
                if optimal_value < 1.0:  # Can buy complete coverage for < $1
                    return {
                        "arbitrage_found": True,
                        "total_cost": optimal_value,
                        "profit": 1.0 - optimal_value,
                        "optimal_outcome": z.value.tolist()
                    }
        except Exception as e:
            pass  # Fall back to heuristics

        return None

    def build_dependency_constraints(
        self,
        market_a_conditions: list[str],
        market_b_conditions: list[str],
        dependencies: list[tuple[int, int, str]]  # (idx_a, idx_b, "implies"|"excludes")
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build constraint matrix from logical dependencies.

        Example: "Trump wins PA" (a[0]) implies "Republican wins PA" (b[0])
        Constraint: z_a0 <= z_b0, or: -z_a0 + z_b0 >= 0
        """
        n_a = len(market_a_conditions)
        n_b = len(market_b_conditions)
        n_total = n_a + n_b

        constraints = []
        bounds = []

        for idx_a, idx_b, dep_type in dependencies:
            row = np.zeros(n_total)
            if dep_type == "implies":
                # z_a <= z_b => -z_a + z_b >= 0
                row[idx_a] = -1
                row[n_a + idx_b] = 1
                bounds.append(0)
            elif dep_type == "excludes":
                # z_a + z_b <= 1 => -z_a - z_b >= -1
                row[idx_a] = -1
                row[n_a + idx_b] = -1
                bounds.append(-1)
            constraints.append(row)

        # Exactly one outcome per market
        # sum(z_a) = 1
        row_a = np.zeros(n_total)
        row_a[:n_a] = 1
        constraints.append(row_a)
        constraints.append(-row_a)
        bounds.extend([1, -1])

        # sum(z_b) = 1
        row_b = np.zeros(n_total)
        row_b[n_a:] = 1
        constraints.append(row_b)
        constraints.append(-row_b)
        bounds.extend([1, -1])

        return np.array(constraints), np.array(bounds)
```

#### 1.2 Add LLM-based Dependency Detection

**New File:** `backend/services/optimization/dependency_detector.py`

```python
"""
LLM-based market dependency detection.

Research paper used DeepSeek-R1-Distill-Qwen-32B with 81.45% accuracy
on complex multi-condition markets.
"""

import json
from typing import Optional
import httpx

class DependencyDetector:
    """
    Use LLM to detect logical dependencies between markets.

    Example dependencies detected:
    - "Trump wins PA" implies "Republican wins PA"
    - "BTC > $100K by March" excludes "BTC < $50K in March"
    - "Duke wins 6 games" excludes "Cornell wins 6 games" (semi-final constraint)
    """

    DEPENDENCY_PROMPT = '''Analyze these two prediction markets for logical dependencies.

Market A: {market_a_question}
Conditions: {market_a_conditions}

Market B: {market_b_question}
Conditions: {market_b_conditions}

Identify ALL logical dependencies between outcomes. A dependency exists when:
1. One outcome IMPLIES another (if A happens, B must happen)
2. Two outcomes are MUTUALLY EXCLUSIVE (both cannot happen)
3. Outcomes have CUMULATIVE relationships (A happening means all "by date X" versions also happen)

Return JSON:
{{
  "dependencies": [
    {{"a_condition": <index>, "b_condition": <index>, "type": "implies|excludes|cumulative", "reason": "..."}}
  ],
  "valid_combinations": <number of valid outcome combinations>,
  "independent": <true if no dependencies found>
}}

Be precise. False negatives (missing dependencies) are costly.'''

    def __init__(self, api_url: str = "http://localhost:11434/api/generate"):
        self.api_url = api_url
        self.model = "deepseek-r1:32b"  # Or any capable local model

    async def detect_dependencies(
        self,
        market_a: dict,
        market_b: dict
    ) -> Optional[dict]:
        """
        Detect dependencies between two markets using LLM.
        """
        prompt = self.DEPENDENCY_PROMPT.format(
            market_a_question=market_a.get("question", ""),
            market_a_conditions=market_a.get("conditions", []),
            market_b_question=market_b.get("question", ""),
            market_b_conditions=market_b.get("conditions", [])
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return json.loads(result.get("response", "{}"))
        except Exception as e:
            pass

        return None

    async def batch_detect(
        self,
        market_pairs: list[tuple[dict, dict]],
        concurrency: int = 10
    ) -> list[dict]:
        """
        Batch process market pairs for dependency detection.

        Research found 1,576 dependent pairs in 46,360 possible pairs (3.4%).
        Pre-filtering with keyword matching reduces API calls significantly.
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def detect_with_limit(pair):
            async with semaphore:
                return await self.detect_dependencies(pair[0], pair[1])

        tasks = [detect_with_limit(pair) for pair in market_pairs]
        return await asyncio.gather(*tasks)
```

---

## Part 2: Bregman Projection

### What the Article Describes

Finding arbitrage is one problem. Calculating the **optimal trade** is another.

The maximum guaranteed profit equals:
```
max_δ [min_ω (δ·φ(ω) - C(θ+δ) + C(θ))] = D(μ*||θ)
```

Where μ* is the **Bregman projection** of current prices θ onto the arbitrage-free manifold M.

For LMSR (Logarithmic Market Scoring Rule), this is the **KL divergence**:
```
D(μ||θ) = Σ μ_i × ln(μ_i / θ_i)
```

### Current homerun Implementation

**File:** `backend/services/strategies/base.py`

```python
def create_opportunity(self, title, description, total_cost, markets, positions, event=None):
    expected_payout = 1.0
    gross_profit = expected_payout - total_cost
    fee = expected_payout * self.fee
    net_profit = gross_profit - fee
    roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
```

**Problems:**
1. Doesn't account for market maker cost function structure
2. Doesn't optimize trade direction (which positions to buy/sell)
3. Doesn't calculate information-theoretic optimal trade size
4. Linear profit calculation vs convex optimization

### Recommended Improvements

#### 2.1 Bregman Projection Implementation

**New File:** `backend/services/optimization/bregman.py`

```python
"""
Bregman Projection for optimal arbitrage trade calculation.

The Bregman divergence respects the market maker's cost function structure,
providing the information-theoretically optimal way to remove arbitrage.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional

class BregmanProjector:
    """
    Compute Bregman projections for LMSR-based prediction markets.

    For LMSR, the convex function R(μ) is negative entropy:
    R(μ) = Σ μ_i × ln(μ_i)

    The Bregman divergence becomes KL divergence:
    D(μ||θ) = Σ μ_i × ln(μ_i / θ_i)
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Args:
            epsilon: Small constant to prevent log(0)
        """
        self.epsilon = epsilon

    def kl_divergence(self, mu: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute KL divergence D(μ||θ).

        This represents the maximum extractable arbitrage profit.
        """
        mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
        theta = np.clip(theta, self.epsilon, 1 - self.epsilon)
        return np.sum(mu * np.log(mu / theta))

    def project_simplex(self, prices: np.ndarray) -> np.ndarray:
        """
        Project prices onto probability simplex.

        Solves: argmin_μ D(μ||θ) s.t. Σμ_i = 1, μ_i ≥ 0

        For KL divergence, the solution is simply normalization.
        """
        prices = np.clip(prices, self.epsilon, None)
        return prices / np.sum(prices)

    def project_polytope(
        self,
        prices: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_bounds: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Project prices onto marginal polytope defined by linear constraints.

        Solves: argmin_μ D(μ||θ) s.t. Aμ ≥ b, Σμ_i = 1

        Returns:
            (projected_prices, arbitrage_profit)
        """
        n = len(prices)
        theta = np.clip(prices, self.epsilon, 1 - self.epsilon)

        # Initial guess: normalized prices
        mu0 = theta / np.sum(theta)

        # Objective: KL divergence
        def objective(mu):
            mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
            return np.sum(mu * np.log(mu / theta))

        # Gradient of KL divergence
        def gradient(mu):
            mu = np.clip(mu, self.epsilon, 1 - self.epsilon)
            return np.log(mu / theta) + 1

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda mu: np.sum(mu) - 1},  # Sum to 1
        ]

        # Add inequality constraints from polytope
        for i in range(len(constraint_bounds)):
            constraints.append({
                "type": "ineq",
                "fun": lambda mu, i=i: constraint_matrix[i] @ mu - constraint_bounds[i]
            })

        # Bounds: probabilities in [0, 1]
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

        if result.success:
            mu_star = result.x
            profit = self.kl_divergence(mu_star, theta)
            return mu_star, profit

        return mu0, 0.0

    def compute_optimal_trade(
        self,
        current_prices: np.ndarray,
        projected_prices: np.ndarray,
        liquidity_parameter: float = 100.0
    ) -> dict:
        """
        Compute optimal trade direction and size from projection.

        Args:
            current_prices: Current market prices θ
            projected_prices: Bregman projection μ*
            liquidity_parameter: LMSR liquidity parameter b

        Returns:
            dict with optimal positions and expected profit
        """
        # Trade direction: move from θ to μ*
        direction = projected_prices - current_prices

        # For LMSR, optimal trade size relates to liquidity parameter
        # shares = b × ln(μ* / θ) for each outcome
        shares = liquidity_parameter * np.log(
            np.clip(projected_prices, self.epsilon, None) /
            np.clip(current_prices, self.epsilon, None)
        )

        # Cost of trade
        cost = liquidity_parameter * (
            np.log(np.sum(np.exp(shares / liquidity_parameter))) -
            np.log(len(current_prices))
        )

        return {
            "direction": direction.tolist(),
            "shares": shares.tolist(),
            "estimated_cost": float(cost),
            "expected_profit": float(self.kl_divergence(projected_prices, current_prices)),
            "positions": [
                {
                    "index": i,
                    "action": "BUY" if shares[i] > 0 else "SELL",
                    "shares": abs(shares[i]),
                    "price_change": direction[i]
                }
                for i in range(len(shares))
                if abs(shares[i]) > 0.01  # Filter negligible positions
            ]
        }
```

---

## Part 3: Frank-Wolfe Algorithm

### What the Article Describes

Computing Bregman projection directly is intractable when the marginal polytope has exponentially many vertices. The **Frank-Wolfe algorithm** reduces this to a sequence of linear programs:

```
1. Start with small set of vertices Z₀
2. For iteration t:
   a. Solve convex optimization over conv(Z_{t-1})
   b. Find new descent vertex by solving IP: z_t = argmin_{z∈Z} ∇F(μ_t)·z
   c. Add to active set: Z_t = Z_{t-1} ∪ {z_t}
   d. Stop if gap g(μ_t) ≤ ε
```

The research showed 50-150 iterations sufficient for markets with thousands of conditions.

### Current homerun Implementation

No iterative optimization algorithms are implemented. The system uses direct heuristic checks.

### Recommended Improvements

#### 3.1 Frank-Wolfe Implementation

**New File:** `backend/services/optimization/frank_wolfe.py`

```python
"""
Frank-Wolfe algorithm for Bregman projection onto marginal polytopes.

Makes optimization tractable even for outcome spaces with 2^63 possibilities
by building the polytope iteratively through linear programming.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

@dataclass
class FrankWolfeResult:
    """Result of Frank-Wolfe optimization."""
    optimal_prices: np.ndarray
    arbitrage_profit: float
    iterations: int
    converged: bool
    active_vertices: int
    gap_history: list[float]

class FrankWolfeSolver:
    """
    Frank-Wolfe algorithm with integer programming oracle.

    Solves: min_μ F(μ) s.t. μ ∈ M = conv(Z)

    Where Z is defined by integer constraints (valid market outcomes)
    and F is the Bregman divergence.
    """

    def __init__(
        self,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-6,
        initial_contraction: float = 0.1,
        ip_timeout: float = 30.0
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.epsilon = initial_contraction  # Barrier parameter
        self.ip_timeout = ip_timeout

    def solve(
        self,
        prices: np.ndarray,
        ip_oracle: Callable[[np.ndarray], np.ndarray],
        bregman_objective: Callable[[np.ndarray], float],
        bregman_gradient: Callable[[np.ndarray], np.ndarray]
    ) -> FrankWolfeResult:
        """
        Run Frank-Wolfe optimization.

        Args:
            prices: Current market prices θ
            ip_oracle: Function that solves min_{z∈Z} c·z
            bregman_objective: F(μ) - the divergence from prices
            bregman_gradient: ∇F(μ)

        Returns:
            FrankWolfeResult with optimal projection
        """
        n = len(prices)

        # Initialize with a feasible point (uniform distribution on active set)
        # First vertex from IP oracle with zero gradient
        z0 = ip_oracle(np.zeros(n))
        active_set = [z0]

        # Current iterate: start at first vertex
        mu = z0.copy().astype(float)

        # Barrier Frank-Wolfe: contract polytope to bound gradients
        interior_point = np.ones(n) / n

        gap_history = []

        for t in range(self.max_iterations):
            # Compute contracted iterate for gradient evaluation
            mu_contracted = (1 - self.epsilon) * mu + self.epsilon * interior_point

            # Gradient at contracted point
            grad = bregman_gradient(mu_contracted)

            # Oracle call: find vertex minimizing gradient
            z_new = ip_oracle(grad)

            # Compute Frank-Wolfe gap
            gap = np.dot(grad, mu - z_new)
            gap_history.append(gap)

            # Check convergence
            if gap < self.convergence_threshold:
                return FrankWolfeResult(
                    optimal_prices=mu,
                    arbitrage_profit=bregman_objective(mu),
                    iterations=t + 1,
                    converged=True,
                    active_vertices=len(active_set),
                    gap_history=gap_history
                )

            # Add new vertex to active set if novel
            is_novel = True
            for z in active_set:
                if np.allclose(z, z_new):
                    is_novel = False
                    break
            if is_novel:
                active_set.append(z_new)

            # Solve subproblem over convex hull of active set
            mu = self._solve_subproblem(
                active_set, prices, bregman_objective
            )

            # Adaptive epsilon reduction (Barrier Frank-Wolfe)
            g_u = np.dot(bregman_gradient(interior_point), interior_point - z_new)
            if g_u < 0 and gap / (-4 * g_u) < self.epsilon:
                self.epsilon = min(gap / (-4 * g_u), self.epsilon / 2)

        return FrankWolfeResult(
            optimal_prices=mu,
            arbitrage_profit=bregman_objective(mu),
            iterations=self.max_iterations,
            converged=False,
            active_vertices=len(active_set),
            gap_history=gap_history
        )

    def _solve_subproblem(
        self,
        active_set: list[np.ndarray],
        prices: np.ndarray,
        objective: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """
        Solve convex optimization over convex hull of active vertices.

        min_λ F(Σ λ_i z_i) s.t. Σλ_i = 1, λ_i ≥ 0
        """
        from scipy.optimize import minimize

        k = len(active_set)
        Z = np.array(active_set)  # k x n matrix

        def objective_lambda(lam):
            mu = Z.T @ lam  # Convex combination
            return objective(mu)

        # Initial: uniform weights
        lam0 = np.ones(k) / k

        # Constraints: sum to 1, non-negative
        constraints = [{"type": "eq", "fun": lambda lam: np.sum(lam) - 1}]
        bounds = [(0, 1)] * k

        result = minimize(
            objective_lambda,
            lam0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds
        )

        if result.success:
            return Z.T @ result.x
        return Z.T @ lam0

class IPOracle:
    """
    Integer Programming oracle for Frank-Wolfe.

    Solves: min_{z∈Z} c·z where Z = {z ∈ {0,1}^n : A^T z ≥ b}
    """

    def __init__(self, constraint_matrix: np.ndarray, constraint_bounds: np.ndarray):
        self.A = constraint_matrix
        self.b = constraint_bounds

        try:
            import cvxpy as cp
            self.cp = cp
            self.solver = "GLPK_MI"
        except ImportError:
            self.cp = None

    def __call__(self, c: np.ndarray) -> np.ndarray:
        """Find vertex minimizing c·z subject to constraints."""
        if self.cp is None:
            return self._fallback_oracle(c)

        n = len(c)
        z = self.cp.Variable(n, boolean=True)

        constraints = [self.A @ z >= self.b]
        objective = self.cp.Minimize(c @ z)

        problem = self.cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver, verbose=False)
            if problem.status == self.cp.OPTIMAL:
                return np.round(z.value).astype(float)
        except Exception:
            pass

        return self._fallback_oracle(c)

    def _fallback_oracle(self, c: np.ndarray) -> np.ndarray:
        """Simple fallback: return unit vector for minimum c_i."""
        z = np.zeros(len(c))
        z[np.argmin(c)] = 1
        return z
```

---

## Part 4: Execution Under Non-Atomic Constraints

### What the Article Describes

CLOB execution is **sequential, not atomic**. Your arbitrage plan:
- Buy YES at $0.30, Buy NO at $0.30 → Cost $0.60, Payout $1.00

Reality:
- Submit YES → Fills at $0.30 ✓
- Price updates due to your order
- Submit NO → Fills at $0.78 ✗
- Actual cost: $1.08, Loss: -$0.08

The research only counted opportunities with **≥$0.05 profit margin** to account for execution risk.

**Fast wallet execution:**
- WebSocket price feed: <5ms
- Decision computation: <10ms
- Direct RPC submission: ~15ms
- **All legs submitted within 30ms** → Same block inclusion

### Current homerun Implementation

**File:** `backend/services/trading.py`

```python
async def execute_opportunity(self, opportunity_id, positions, size_usd):
    orders = []
    for position in positions:  # SEQUENTIAL!
        order = await self.place_order(...)  # Waits for each
        orders.append(order)
    return orders
```

**Problems:**
1. Sequential execution allows price movement between legs
2. No VWAP calculation across order book
3. ~1 second execution delay
4. No parallel order submission

### Recommended Improvements

#### 4.1 Parallel Order Execution

**Modify:** `backend/services/trading.py`

```python
import asyncio
from typing import List

async def execute_opportunity_parallel(
    self,
    opportunity_id: str,
    positions: list[dict],
    size_usd: float
) -> list[Order]:
    """
    Execute all legs of an arbitrage opportunity in PARALLEL.

    This is critical for non-atomic CLOB execution. All orders
    should be submitted within the same block window (~2 seconds)
    to minimize execution risk.
    """

    # Pre-validate all positions before any execution
    validation_errors = []
    for position in positions:
        token_id = position.get("token_id")
        if not token_id:
            validation_errors.append(f"Missing token_id in position")
            continue
        price = position.get("price", 0)
        if price <= 0:
            validation_errors.append(f"Invalid price {price} for {token_id}")

    if validation_errors:
        raise ValueError(f"Position validation failed: {validation_errors}")

    # Build all orders simultaneously
    async def place_single_order(position: dict) -> Order:
        token_id = position.get("token_id")
        price = position.get("price")
        position_usd = size_usd / len(positions)
        shares = position_usd / price

        return await self.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=price,
            size=shares,
            market_question=position.get("market"),
            opportunity_id=opportunity_id
        )

    # Execute ALL orders in parallel
    # asyncio.gather submits all coroutines before any await
    tasks = [place_single_order(pos) for pos in positions]
    orders = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful_orders = []
    failed_orders = []

    for i, result in enumerate(orders):
        if isinstance(result, Exception):
            failed_orders.append((positions[i], str(result)))
        elif result.status == OrderStatus.FAILED:
            failed_orders.append((positions[i], result.error_message))
        else:
            successful_orders.append(result)

    # If partial execution, we have exposure risk
    if failed_orders and successful_orders:
        logger.warning(
            f"PARTIAL EXECUTION: {len(successful_orders)} succeeded, "
            f"{len(failed_orders)} failed. Exposure risk!"
        )
        # Could implement automatic hedging here

    return successful_orders
```

#### 4.2 VWAP Price Analysis

**New File:** `backend/services/execution/vwap.py`

```python
"""
Volume-Weighted Average Price (VWAP) analysis for execution planning.

Instead of assuming fills at quoted prices, calculate expected
execution price across order book depth.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: float  # In shares

@dataclass
class OrderBook:
    """Complete order book for a token."""
    bids: list[OrderBookLevel]  # Sorted descending by price
    asks: list[OrderBookLevel]  # Sorted ascending by price

@dataclass
class VWAPResult:
    """Result of VWAP calculation."""
    vwap: float
    total_available: float
    slippage_from_mid: float
    levels_consumed: int
    fill_probability: float

class VWAPCalculator:
    """
    Calculate Volume-Weighted Average Price for order execution.

    Research methodology:
    - Calculate VWAP from all trades in each Polygon block (~2s)
    - Only count arbitrage if |VWAP_yes + VWAP_no - 1.0| > 0.02
    """

    def calculate_buy_vwap(
        self,
        order_book: OrderBook,
        size_usd: float
    ) -> VWAPResult:
        """
        Calculate VWAP for buying a given USD amount.

        Walks through ask side of order book.
        """
        if not order_book.asks:
            return VWAPResult(
                vwap=0,
                total_available=0,
                slippage_from_mid=0,
                levels_consumed=0,
                fill_probability=0
            )

        remaining_usd = size_usd
        total_shares = 0
        total_cost = 0
        levels_consumed = 0

        for level in order_book.asks:
            level_value = level.price * level.size

            if remaining_usd <= 0:
                break

            if level_value <= remaining_usd:
                # Consume entire level
                total_shares += level.size
                total_cost += level_value
                remaining_usd -= level_value
            else:
                # Partial fill on this level
                shares_to_buy = remaining_usd / level.price
                total_shares += shares_to_buy
                total_cost += remaining_usd
                remaining_usd = 0

            levels_consumed += 1

        if total_shares == 0:
            return VWAPResult(
                vwap=order_book.asks[0].price if order_book.asks else 0,
                total_available=0,
                slippage_from_mid=0,
                levels_consumed=0,
                fill_probability=0
            )

        vwap = total_cost / total_shares

        # Calculate slippage from mid price
        best_ask = order_book.asks[0].price if order_book.asks else vwap
        best_bid = order_book.bids[0].price if order_book.bids else vwap
        mid = (best_ask + best_bid) / 2
        slippage = (vwap - mid) / mid if mid > 0 else 0

        # Fill probability based on how much was available
        fill_probability = min(1.0, (total_cost) / size_usd) if size_usd > 0 else 1.0

        return VWAPResult(
            vwap=vwap,
            total_available=total_cost,
            slippage_from_mid=slippage,
            levels_consumed=levels_consumed,
            fill_probability=fill_probability
        )

    def estimate_arbitrage_profit(
        self,
        yes_book: OrderBook,
        no_book: OrderBook,
        size_usd: float
    ) -> dict:
        """
        Estimate actual arbitrage profit accounting for order book depth.

        Returns realistic profit after VWAP analysis.
        """
        # Half the capital to each side
        half_size = size_usd / 2

        yes_vwap = self.calculate_buy_vwap(yes_book, half_size)
        no_vwap = self.calculate_buy_vwap(no_book, half_size)

        # Total cost using VWAP
        total_vwap_cost = yes_vwap.vwap + no_vwap.vwap

        # Minimum achievable position (limited by liquidity)
        achievable_size = min(
            yes_vwap.total_available,
            no_vwap.total_available
        )

        # Joint fill probability
        fill_probability = yes_vwap.fill_probability * no_vwap.fill_probability

        # Realistic profit
        if total_vwap_cost < 1.0 and achievable_size > 0:
            gross_profit = 1.0 - total_vwap_cost
            # Risk-adjusted profit
            expected_profit = gross_profit * fill_probability
        else:
            gross_profit = 0
            expected_profit = 0

        return {
            "yes_vwap": yes_vwap.vwap,
            "no_vwap": no_vwap.vwap,
            "total_vwap_cost": total_vwap_cost,
            "gross_profit": gross_profit,
            "fill_probability": fill_probability,
            "expected_profit": expected_profit,
            "max_executable_usd": achievable_size,
            "yes_slippage": yes_vwap.slippage_from_mid,
            "no_slippage": no_vwap.slippage_from_mid,
            "profitable": total_vwap_cost < 0.95  # 5% margin for execution risk
        }
```

#### 4.3 Latency-Optimized Execution

**New File:** `backend/services/execution/fast_executor.py`

```python
"""
Latency-optimized execution for arbitrage trades.

Target: <30ms from detection to mempool submission.
Eliminates the 1-second delay in current implementation.
"""

import asyncio
import time
from typing import Optional
from dataclasses import dataclass
import aiohttp

@dataclass
class ExecutionMetrics:
    """Metrics for execution latency tracking."""
    detection_time: float
    validation_time: float
    signing_time: float
    submission_time: float
    total_latency_ms: float

class FastExecutor:
    """
    Minimizes latency from opportunity detection to order submission.

    Research timeline for sophisticated systems:
    - WebSocket price feed: <5ms
    - Decision computation: <10ms (pre-calculated)
    - Direct RPC submission: ~15ms
    - Total: ~30ms vs retail's ~2,650ms
    """

    def __init__(
        self,
        rpc_url: str = "https://polygon-rpc.com",
        max_latency_ms: float = 100.0
    ):
        self.rpc_url = rpc_url
        self.max_latency_ms = max_latency_ms
        self._session: Optional[aiohttp.ClientSession] = None
        self._precomputed_orders: dict = {}  # Cache for common trades

    async def ensure_session(self):
        """Ensure persistent HTTP session for minimal connection overhead."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=5)
            )

    def precompute_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str
    ):
        """
        Pre-compute and cache order signatures for expected trades.

        When an opportunity is detected, order is already signed
        and ready to submit, eliminating signing latency.
        """
        # In production, would pre-sign orders for common price levels
        key = f"{token_id}:{price}:{size}:{side}"
        self._precomputed_orders[key] = {
            "token_id": token_id,
            "price": price,
            "size": size,
            "side": side,
            "precomputed_at": time.time()
        }

    async def execute_fast(
        self,
        positions: list[dict],
        size_usd: float
    ) -> tuple[list[dict], ExecutionMetrics]:
        """
        Execute with minimal latency.

        Key optimizations:
        1. Parallel order building
        2. Single batched RPC call
        3. Pre-computed signatures where possible
        4. Persistent HTTP connection
        """
        start = time.perf_counter()

        await self.ensure_session()

        validation_start = time.perf_counter()
        # Validation should be <1ms
        for pos in positions:
            if pos.get("price", 0) <= 0:
                raise ValueError("Invalid price")
        validation_time = (time.perf_counter() - validation_start) * 1000

        signing_start = time.perf_counter()
        # Build all orders in parallel
        orders = await asyncio.gather(*[
            self._build_order_fast(pos, size_usd / len(positions))
            for pos in positions
        ])
        signing_time = (time.perf_counter() - signing_start) * 1000

        submission_start = time.perf_counter()
        # Submit all in single batch RPC call
        results = await self._batch_submit(orders)
        submission_time = (time.perf_counter() - submission_start) * 1000

        total_latency = (time.perf_counter() - start) * 1000

        metrics = ExecutionMetrics(
            detection_time=0,  # Filled by caller
            validation_time=validation_time,
            signing_time=signing_time,
            submission_time=submission_time,
            total_latency_ms=total_latency
        )

        return results, metrics

    async def _build_order_fast(self, position: dict, usd_amount: float) -> dict:
        """Build order with minimal overhead."""
        return {
            "token_id": position["token_id"],
            "price": position["price"],
            "size": usd_amount / position["price"],
            "side": "BUY"
        }

    async def _batch_submit(self, orders: list[dict]) -> list[dict]:
        """
        Submit multiple orders in single RPC batch.

        Reduces network round-trips from N to 1.
        """
        # In production, would use actual CLOB batch submission
        results = []
        for order in orders:
            results.append({
                "order_id": f"order_{time.time_ns()}",
                "status": "submitted",
                "token_id": order["token_id"]
            })
        return results
```

---

## Part 5: Enhanced Position Sizing

### What the Article Describes

Modified Kelly criterion accounting for execution risk:
```
f* = (b×p - q) / b × √p
```

Where:
- b = arbitrage profit percentage
- p = probability of full execution
- q = 1 - p

### Current homerun Implementation

**File:** `backend/services/auto_trader.py`

```python
def _calculate_position_size(self, opp: ArbitrageOpportunity) -> float:
    if self.config.position_size_method == "kelly":
        win_prob = 1 - opp.risk_score  # Approximate
        expected_return = opp.roi_percent / 100
        kelly_fraction = (expected_return * win_prob - (1 - win_prob)) / expected_return
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        size = self.config.max_position_size_usd * kelly_fraction
```

**Problems:**
1. Uses risk_score as win probability (not execution probability)
2. Doesn't account for order book depth
3. No execution risk adjustment (√p factor)

### Recommended Improvements

**Modify:** `backend/services/auto_trader.py`

```python
def _calculate_position_size_advanced(
    self,
    opp: ArbitrageOpportunity,
    vwap_result: Optional[dict] = None
) -> float:
    """
    Calculate position size with execution risk adjustment.

    Uses modified Kelly: f* = (b×p - q) / b × √p

    Where p is execution probability, not win probability
    (arbitrage has ~100% win rate if executed correctly).
    """
    # Base expected return
    expected_return = opp.roi_percent / 100

    # Execution probability from VWAP analysis
    if vwap_result:
        exec_prob = vwap_result.get("fill_probability", 0.9)
    else:
        # Estimate from liquidity
        if opp.min_liquidity < 1000:
            exec_prob = 0.5
        elif opp.min_liquidity < 5000:
            exec_prob = 0.75
        elif opp.min_liquidity < 20000:
            exec_prob = 0.9
        else:
            exec_prob = 0.95

    # Modified Kelly with execution risk
    # Standard Kelly: f = (bp - q) / b
    # Modified: f = (bp - q) / b × √p (conservative for execution uncertainty)
    if expected_return > 0:
        q = 1 - exec_prob
        standard_kelly = (expected_return * exec_prob - q) / expected_return
        # Apply √p adjustment for execution risk
        adjusted_kelly = standard_kelly * (exec_prob ** 0.5)
        kelly_fraction = max(0, min(adjusted_kelly, 0.25))
    else:
        kelly_fraction = 0

    # Base size from Kelly
    size = self.config.max_position_size_usd * kelly_fraction

    # Cap at order book depth
    if vwap_result:
        max_from_liquidity = vwap_result.get("max_executable_usd", float("inf"))
        size = min(size, max_from_liquidity * 0.5)  # Don't take >50% of book

    # Minimum viable size (must exceed fees)
    min_viable = opp.fee * 10  # At least 10x the fee to be worth it

    if size < min_viable:
        return 0  # Not worth executing

    return max(size, settings.MIN_ORDER_SIZE_USD)
```

---

## Part 6: Implementation Priority Matrix

| Priority | Component | Effort | Impact | Dependencies |
|----------|-----------|--------|--------|--------------|
| **P0** | Parallel Order Execution | Low | High | None |
| **P0** | VWAP Price Analysis | Medium | High | Order book data |
| **P1** | Bregman Projection | Medium | High | NumPy/SciPy |
| **P1** | Modified Kelly Sizing | Low | Medium | VWAP |
| **P2** | Frank-Wolfe Solver | High | High | CVXPY/Gurobi |
| **P2** | LLM Dependency Detection | Medium | High | LLM API |
| **P2** | Integer Programming | High | High | CVXPY/Gurobi |
| **P3** | Fast Executor | Medium | Medium | Infrastructure |
| **P3** | Pre-computed Orders | Medium | Medium | Trading service |

### Recommended Implementation Order

1. **Week 1-2: Execution Layer**
   - Parallel order execution (modify `trading.py`)
   - Basic VWAP calculator
   - Remove 1-second execution delay

2. **Week 3-4: Optimization Foundation**
   - Bregman projection implementation
   - Modified Kelly position sizing
   - Integration with existing strategies

3. **Week 5-8: Advanced Detection**
   - Integer programming setup (CVXPY + optional Gurobi)
   - Frank-Wolfe solver
   - LLM dependency detection

4. **Week 9-12: Infrastructure**
   - WebSocket price feeds
   - Latency monitoring
   - Production hardening

---

## Part 7: New Dependencies Required

Add to `requirements.txt`:

```
# Optimization
cvxpy>=1.4.0
numpy>=1.24.0
scipy>=1.11.0

# Optional: Gurobi (requires license)
# gurobipy>=10.0.0

# LLM Integration (optional)
httpx>=0.25.0

# WebSocket for real-time feeds
websockets>=12.0
```

---

## Conclusion

The gap between homerun's current implementation and the quantitative systems that extracted $40M is substantial but addressable. The key insight from the research is:

> "The difference isn't just speed. It's mathematical infrastructure."

Current homerun has solid foundations:
- ✓ Multiple arbitrage strategies
- ✓ Risk scoring system
- ✓ Position sizing (basic)
- ✓ Circuit breakers
- ✓ Paper trading simulation

What's missing to compete at the $40M level:
- ✗ Integer programming for constraint satisfaction
- ✗ Bregman projection for optimal trades
- ✗ Frank-Wolfe for computational tractability
- ✗ Parallel non-atomic execution
- ✗ VWAP-based pricing
- ✗ Sub-100ms latency

The research paper's concluding insight applies directly:

> "The math works. The infrastructure exists. The only question is execution."

With the improvements outlined above, homerun can evolve from a heuristic arbitrage scanner to a mathematically rigorous quantitative system.
