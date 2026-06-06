"""Quick structured-log query helper for tools/.harness/*.log (JSONL).

Usage:
  python tools/logq.py <plane> span
  python tools/logq.py <plane> grep <substr> [n]
  python tools/logq.py <plane> errors [n]
  python tools/logq.py <plane> fn <function_substr> [n]
"""
import sys, json, os

def load(plane):
    path = os.path.join(os.path.dirname(__file__), ".harness", f"{plane}.log")
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out

def fmt(o):
    ts = str(o.get("timestamp", ""))[11:19]
    lvl = str(o.get("level", ""))[:4]
    fn = str(o.get("function", ""))
    msg = str(o.get("message", ""))
    data = o.get("data")
    extra = ""
    if isinstance(data, dict) and data:
        items = list(data.items())[:6]
        extra = " {" + ", ".join(f"{k}={v}" for k, v in items) + "}"
    return f"{ts} {lvl} {fn} | {msg[:140]}{extra[:160]}"

def main():
    plane = sys.argv[1]
    cmd = sys.argv[2]
    rows = load(plane)
    if cmd == "span":
        if rows:
            print("SPAN:", str(rows[0].get("timestamp"))[:19], "->",
                  str(rows[-1].get("timestamp"))[:19], f"({len(rows)} lines)")
        return
    if cmd == "errors":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        hits = [o for o in rows if o.get("level") in ("ERROR", "WARNING")]
        for o in hits[-n:]:
            print(fmt(o))
        return
    if cmd == "grep":
        sub = sys.argv[3].lower()
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
        hits = [o for o in rows if sub in str(o.get("message", "")).lower()
                or sub in str(o.get("function", "")).lower()]
        for o in hits[-n:]:
            print(fmt(o))
        print(f"[{len(hits)} total matches]")
        return
    if cmd == "fn":
        sub = sys.argv[3].lower()
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
        hits = [o for o in rows if sub in str(o.get("function", "")).lower()]
        for o in hits[-n:]:
            print(fmt(o))
        print(f"[{len(hits)} total matches]")
        return

if __name__ == "__main__":
    main()
