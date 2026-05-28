"""CI lint: flag raw wall-clock reads in strategy code.

Strategies must read time via ``self.now()`` / ``self.now_ms()`` /
``self.now_us()`` (or the ``utcnow()`` helper, which honours the backtest
replay clock) so decisions are a pure function of recorded inputs and
replay is deterministic.  Raw ``datetime.now()`` / ``datetime.utcnow()``
/ ``time.time()`` bypass the replay clock and silently leak wall-clock
time into backtests.

Exit code 1 if any un-annotated raw clock read is found.  A genuine
non-decision use (e.g. measuring elapsed wall time for a perf log) may be
suppressed with a trailing ``# clock-ok: <reason>`` comment.

Usage:  python scripts/lint_strategy_clock.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "services" / "strategies"

# base.py defines now()/now_ms() and is the sanctioned wall-clock source.
_EXEMPT_FILES = {"base.py"}

_PATTERNS = (
    re.compile(r"\bdatetime\.now\("),
    re.compile(r"\bdatetime\.utcnow\("),
    re.compile(r"\btime\.time\("),
)
_SUPPRESS = re.compile(r"#\s*clock-ok\b")


def scan() -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for path in sorted(_STRATEGIES_DIR.rglob("*.py")):
        if path.name in _EXEMPT_FILES:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _SUPPRESS.search(line):
                continue
            if any(p.search(line) for p in _PATTERNS):
                rel = path.relative_to(_STRATEGIES_DIR.parent.parent)
                findings.append((str(rel), i, line.strip()))
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        print("strategy-clock lint: clean — no raw wall-clock reads in strategies.")
        return 0
    print(f"strategy-clock lint: {len(findings)} raw wall-clock read(s) — "
          "route through self.now()/self.now_ms() or add '# clock-ok: <reason>':")
    for rel, ln, code in findings:
        print(f"  {rel}:{ln}: {code}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
