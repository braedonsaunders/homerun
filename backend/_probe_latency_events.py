import asyncio
from sqlalchemy import text
from models.database import AsyncSessionLocal

STAGES = [
    "emit_to_queue_wake_ms",
    "wake_to_context_ready_ms",
    "context_ready_to_decision_ms",
    "ws_release_to_decision_ms",
    "decision_to_submit_start_ms",
    "submit_round_trip_ms",
    "emit_to_submit_start_ms",
]


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    i = max(0, min(len(s) - 1, int(round((len(s) - 1) * p))))
    return s[i]


async def main():
    async with AsyncSessionLocal() as s:
        for win in (30, 120):
            rows = (await s.execute(text(f"""
                SELECT source, payload_json, created_at
                FROM trader_events
                WHERE event_type = 'execution_latency'
                  AND created_at > now() - interval '{win} min'
                ORDER BY created_at DESC
                LIMIT 5000
            """))).all()
            print(f"\n========= execution_latency events, last {win}min: {len(rows)} =========")
            by_source = {}
            for r in rows:
                pj = r.payload_json if isinstance(r.payload_json, dict) else {}
                lat = pj.get("latency") if isinstance(pj.get("latency"), dict) else pj
                src = (r.source or "?").strip().lower()
                by_source.setdefault(src, []).append(lat)
            for src, samples in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
                print(f"\n  source={src!r}  n={len(samples)}")
                for stage in STAGES:
                    vals = []
                    for lat in samples:
                        v = lat.get(stage)
                        if v is None:
                            continue
                        try:
                            vals.append(int(float(v)))
                        except Exception:
                            pass
                    if not vals:
                        continue
                    print(f"    {stage:32} p50={pct(vals,0.5):6} p95={pct(vals,0.95):7} "
                          f"max={max(vals):7}  (n={len(vals)})")
            if rows:
                break  # got data in the 30min window; skip the wider one


asyncio.run(main())
