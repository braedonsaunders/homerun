"""
LLM-based market dependency detection.

Research paper used DeepSeek-R1-Distill-Qwen-32B with 81.45% accuracy
on complex multi-condition markets.

This module detects logical dependencies between markets that enable
combinatorial arbitrage:
- "Trump wins PA" implies "Republican wins PA"
- "BTC > $100K by March" excludes "BTC < $50K in March"
- "Duke wins 6 games" excludes "Cornell wins 6 games" (tournament constraint)

Research findings:
- 40,057 independent pairs (no arbitrage possible)
- 1,576 dependent pairs (potential arbitrage)
- 374 satisfied strict combinatorial conditions
- 13 manually verified as exploitable
"""

import json
import asyncio
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .constraint_solver import Dependency, DependencyType


@dataclass
class MarketInfo:
    """Information about a market for dependency analysis."""
    id: str
    question: str
    outcomes: list[str]  # List of outcome descriptions
    prices: list[float]  # Corresponding prices


@dataclass
class DependencyAnalysis:
    """Result of LLM dependency analysis."""
    dependencies: List[Dependency]
    valid_combinations: int
    total_combinations: int
    is_independent: bool
    confidence: float
    raw_response: Optional[str] = None


class DependencyDetector:
    """
    Use LLM to detect logical dependencies between markets.

    Supports multiple backends:
    - Local Ollama (default, no API key needed)
    - OpenAI API
    - Anthropic API
    """

    DEPENDENCY_PROMPT = '''Analyze these two prediction markets for logical dependencies.

Market A: {market_a_question}
Outcomes: {market_a_outcomes}

Market B: {market_b_question}
Outcomes: {market_b_outcomes}

A logical dependency exists when:
1. IMPLIES: If outcome A_i is true, outcome B_j MUST be true
   Example: "Trump wins Pennsylvania" IMPLIES "A Republican wins Pennsylvania"

2. EXCLUDES: Outcomes A_i and B_j CANNOT both be true
   Example: "Democrats win by 10+ points" EXCLUDES "Republicans win by 5+ points"

3. CUMULATIVE: If A_i is true, all "by later date" versions are also true
   Example: "Event by March" being true means "Event by June" is also true

IMPORTANT: Only report dependencies that are LOGICALLY CERTAIN, not just probable.

Return valid JSON only:
{{
  "dependencies": [
    {{"a_outcome": <index 0-based>, "b_outcome": <index 0-based>, "type": "implies|excludes|cumulative", "reason": "brief explanation"}}
  ],
  "valid_combinations": <number>,
  "is_independent": <true if no dependencies>,
  "confidence": <0.0 to 1.0>
}}'''

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.1:8b",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        """
        Args:
            backend: "ollama", "openai", or "anthropic"
            model: Model to use
            api_url: API endpoint (default based on backend)
            api_key: API key (required for openai/anthropic)
            timeout: Request timeout in seconds
        """
        self.backend = backend
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

        if api_url:
            self.api_url = api_url
        elif backend == "ollama":
            self.api_url = "http://localhost:11434/api/generate"
        elif backend == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
        elif backend == "anthropic":
            self.api_url = "https://api.anthropic.com/v1/messages"
        else:
            self.api_url = "http://localhost:11434/api/generate"

    async def detect_dependencies(
        self,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> DependencyAnalysis:
        """
        Detect dependencies between two markets using LLM.

        Args:
            market_a: First market info
            market_b: Second market info

        Returns:
            DependencyAnalysis with detected dependencies
        """
        # First, try heuristic detection (fast, no API call)
        heuristic_result = self._heuristic_detect(market_a, market_b)
        if heuristic_result.is_independent and heuristic_result.confidence > 0.9:
            return heuristic_result

        # Fall back to LLM if heuristics uncertain
        if not HTTPX_AVAILABLE:
            return heuristic_result

        prompt = self.DEPENDENCY_PROMPT.format(
            market_a_question=market_a.question,
            market_a_outcomes=json.dumps(market_a.outcomes),
            market_b_question=market_b.question,
            market_b_outcomes=json.dumps(market_b.outcomes)
        )

        try:
            response = await self._call_llm(prompt)
            return self._parse_response(response, market_a, market_b)
        except Exception as e:
            # Return heuristic result on failure
            return heuristic_result

    def _heuristic_detect(
        self,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> DependencyAnalysis:
        """
        Fast heuristic dependency detection without LLM.

        Catches common patterns:
        - Same entity with implies relationship (Trump wins X implies Republican wins X)
        - Date-based cumulative relationships
        - Mutually exclusive outcomes
        """
        dependencies = []
        q_a = market_a.question.lower()
        q_b = market_b.question.lower()

        # Extract key entities
        entities_a = self._extract_entities(q_a)
        entities_b = self._extract_entities(q_b)
        common_entities = entities_a & entities_b

        if not common_entities:
            # No common entities, likely independent
            return DependencyAnalysis(
                dependencies=[],
                valid_combinations=len(market_a.outcomes) * len(market_b.outcomes),
                total_combinations=len(market_a.outcomes) * len(market_b.outcomes),
                is_independent=True,
                confidence=0.8
            )

        # Check for implies relationships
        implies_patterns = [
            (r"trump", r"republican"),
            (r"biden|harris|democrat", r"democrat"),
            (r"wins?\s+by\s+(\d+)\+?\s+points?", r"wins?"),
        ]

        for pattern_a, pattern_b in implies_patterns:
            if re.search(pattern_a, q_a) and re.search(pattern_b, q_b):
                # Potential implies relationship
                for i, out_a in enumerate(market_a.outcomes):
                    for j, out_b in enumerate(market_b.outcomes):
                        if self._outcomes_related(out_a, out_b, "implies"):
                            dependencies.append(Dependency(
                                market_a_idx=0,
                                outcome_a_idx=i,
                                market_b_idx=1,
                                outcome_b_idx=j,
                                dep_type=DependencyType.IMPLIES,
                                reason=f"Heuristic: {pattern_a} implies {pattern_b}"
                            ))

        # Check for date-based cumulative relationships
        date_pattern = r"by\s+(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}/\d{1,2})"
        dates_a = re.findall(date_pattern, q_a)
        dates_b = re.findall(date_pattern, q_b)

        if dates_a and dates_b and common_entities:
            # Same event with different dates - cumulative relationship
            for i, out_a in enumerate(market_a.outcomes):
                if "yes" in out_a.lower():
                    for j, out_b in enumerate(market_b.outcomes):
                        if "yes" in out_b.lower():
                            dependencies.append(Dependency(
                                market_a_idx=0,
                                outcome_a_idx=i,
                                market_b_idx=1,
                                outcome_b_idx=j,
                                dep_type=DependencyType.CUMULATIVE,
                                reason="Same event with different dates"
                            ))

        # Check for exclusion relationships
        exclusion_patterns = [
            (r"above|over|more than|greater", r"below|under|less than|fewer"),
            (r"before", r"after"),
            (r"win", r"lose"),
        ]

        for pattern_a, pattern_b in exclusion_patterns:
            if re.search(pattern_a, q_a) and re.search(pattern_b, q_b):
                for i, out_a in enumerate(market_a.outcomes):
                    if "yes" in out_a.lower():
                        for j, out_b in enumerate(market_b.outcomes):
                            if "yes" in out_b.lower():
                                dependencies.append(Dependency(
                                    market_a_idx=0,
                                    outcome_a_idx=i,
                                    market_b_idx=1,
                                    outcome_b_idx=j,
                                    dep_type=DependencyType.EXCLUDES,
                                    reason=f"Heuristic: {pattern_a} excludes {pattern_b}"
                                ))

        n_a = len(market_a.outcomes)
        n_b = len(market_b.outcomes)
        total = n_a * n_b
        # Rough estimate of valid combinations
        valid = max(1, total - len(dependencies))

        return DependencyAnalysis(
            dependencies=dependencies,
            valid_combinations=valid,
            total_combinations=total,
            is_independent=len(dependencies) == 0,
            confidence=0.6 if dependencies else 0.7
        )

    def _extract_entities(self, text: str) -> set:
        """Extract key entities from text."""
        # Common entities in prediction markets
        entities = set()

        # Politicians
        politicians = ["trump", "biden", "harris", "desantis", "obama"]
        for p in politicians:
            if p in text:
                entities.add(p)

        # Parties
        if "republican" in text or "gop" in text:
            entities.add("republican")
        if "democrat" in text:
            entities.add("democrat")

        # States
        states = ["pennsylvania", "georgia", "michigan", "wisconsin", "arizona", "nevada"]
        for s in states:
            if s in text:
                entities.add(s)

        # Crypto
        cryptos = ["bitcoin", "btc", "ethereum", "eth"]
        for c in cryptos:
            if c in text:
                entities.add(c)

        return entities

    def _outcomes_related(self, out_a: str, out_b: str, rel_type: str) -> bool:
        """Check if two outcomes have a specific relationship."""
        out_a = out_a.lower()
        out_b = out_b.lower()

        if rel_type == "implies":
            # YES in A might imply YES in B
            return "yes" in out_a and "yes" in out_b

        return False

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if self.backend == "ollama":
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
                    return response.json().get("response", "{}")

            elif self.backend == "openai":
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"}
                    }
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]

            elif self.backend == "anthropic":
                response = await client.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                if response.status_code == 200:
                    return response.json()["content"][0]["text"]

        return "{}"

    def _parse_response(
        self,
        response: str,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> DependencyAnalysis:
        """Parse LLM response into DependencyAnalysis."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            dependencies = []
            for dep in data.get("dependencies", []):
                dep_type = dep.get("type", "excludes")
                if dep_type == "implies":
                    dtype = DependencyType.IMPLIES
                elif dep_type == "cumulative":
                    dtype = DependencyType.CUMULATIVE
                else:
                    dtype = DependencyType.EXCLUDES

                dependencies.append(Dependency(
                    market_a_idx=0,
                    outcome_a_idx=int(dep.get("a_outcome", 0)),
                    market_b_idx=1,
                    outcome_b_idx=int(dep.get("b_outcome", 0)),
                    dep_type=dtype,
                    reason=dep.get("reason", "")
                ))

            n_a = len(market_a.outcomes)
            n_b = len(market_b.outcomes)

            return DependencyAnalysis(
                dependencies=dependencies,
                valid_combinations=data.get("valid_combinations", n_a * n_b),
                total_combinations=n_a * n_b,
                is_independent=data.get("is_independent", len(dependencies) == 0),
                confidence=data.get("confidence", 0.8),
                raw_response=response
            )

        except Exception as e:
            return DependencyAnalysis(
                dependencies=[],
                valid_combinations=len(market_a.outcomes) * len(market_b.outcomes),
                total_combinations=len(market_a.outcomes) * len(market_b.outcomes),
                is_independent=True,
                confidence=0.5,
                raw_response=response
            )

    async def batch_detect(
        self,
        market_pairs: List[Tuple[MarketInfo, MarketInfo]],
        concurrency: int = 5
    ) -> List[DependencyAnalysis]:
        """
        Batch process market pairs for dependency detection.

        Research found 1,576 dependent pairs in 46,360 possible pairs (3.4%).
        Pre-filtering with heuristics reduces API calls significantly.

        Args:
            market_pairs: List of (market_a, market_b) tuples
            concurrency: Maximum concurrent LLM calls

        Returns:
            List of DependencyAnalysis results
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def detect_with_limit(pair: Tuple[MarketInfo, MarketInfo]):
            async with semaphore:
                return await self.detect_dependencies(pair[0], pair[1])

        tasks = [detect_with_limit(pair) for pair in market_pairs]
        return await asyncio.gather(*tasks)


# Singleton instance
dependency_detector = DependencyDetector()
