from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class DecisionCheck:
    key: str
    label: str
    passed: bool
    score: float | None = None
    detail: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyDecision:
    decision: str  # selected | skipped | blocked | failed
    reason: str
    score: float | None = None
    size_usd: float | None = None
    checks: list[DecisionCheck] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


class TraderStrategy(Protocol):
    key: str

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        ...


class BaseTraderStrategy:
    key = "base"

    def evaluate(self, signal: Any, context: dict[str, Any]) -> StrategyDecision:
        return StrategyDecision(
            decision="skipped",
            reason="Base strategy does not implement evaluate()",
            score=0.0,
            checks=[
                DecisionCheck(
                    key="implemented",
                    label="Strategy implemented",
                    passed=False,
                    detail="No strategy-specific evaluator.",
                )
            ],
        )
