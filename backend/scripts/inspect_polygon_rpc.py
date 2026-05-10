"""Inspect stored Polygon RPC settings — tells you exactly why
``_load_endpoints_from_db`` did or did not pick up the URL you typed
into the SettingsPanel.

Run from the backend directory with the same env this project's
backend uses (so DATABASE_URL and APP_SECRETS_KEY resolve identically):

    python -m scripts.inspect_polygon_rpc

Reports:
  * whether AppSettings row 'default' exists
  * whether polygon_rpc_url / polygon_ws_url columns are NULL
  * whether the stored value is encrypted (``enc::v1::`` prefix)
  * whether ``decrypt_secret`` returns content
  * whether ``_normalize_rpc_http_url`` accepts the decoded value
  * a masked preview of the resulting URL

Never prints the API key — path components after ``/polygon/`` are
masked ``****`` so transcripts and screenshots stay safe.
"""

from __future__ import annotations

import asyncio
import sys


def _mask(url: str) -> str:
    if not url:
        return ""
    head, sep, tail = url.partition("://")
    if not sep:
        return url[:8] + "..."
    if "/" not in tail:
        return url
    host, _, path = tail.partition("/")
    if not path:
        return url
    # Mask everything after the host's first path segment.
    parts = path.split("/")
    if len(parts) >= 2:
        parts[-1] = "****" if parts[-1] else parts[-1]
    masked_path = "/".join(parts)
    return f"{head}://{host}/{masked_path}"


async def main() -> int:
    from sqlalchemy import select
    from models.database import AsyncSessionLocal, AppSettings
    from utils.secrets import decrypt_secret, is_encrypted
    from services.wallet_ws_monitor import _normalize_rpc_http_url

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AppSettings).where(AppSettings.id == "default")
        )
        row = result.scalar_one_or_none()
        if row is None:
            print("AppSettings row 'default' is MISSING. Save anything in the "
                  "SettingsPanel UI to create the row, then re-run.")
            return 1

        rpc_raw = getattr(row, "polygon_rpc_url", None)
        ws_raw = getattr(row, "polygon_ws_url", None)

        print("=== AppSettings 'default' ===")
        print(f"  polygon_rpc_url present:    {bool(rpc_raw)}")
        if rpc_raw:
            print(f"    stored length:            {len(rpc_raw)}")
            print(f"    is_encrypted prefix:      {is_encrypted(rpc_raw)}")
        print(f"  polygon_ws_url present:     {bool(ws_raw)}")
        if ws_raw:
            print(f"    stored length:            {len(ws_raw)}")
            print(f"    is_encrypted prefix:      {is_encrypted(ws_raw)}")

        print()
        print("=== Decryption ===")
        if rpc_raw:
            try:
                rpc_decoded = decrypt_secret(rpc_raw) or ""
            except Exception as exc:
                print(f"  RPC decrypt RAISED: {type(exc).__name__}: {exc}")
                rpc_decoded = ""
            if rpc_decoded:
                print(f"  RPC decrypt OK (len {len(rpc_decoded)})")
                rpc_norm = _normalize_rpc_http_url(rpc_decoded)
                if rpc_norm:
                    print(f"  RPC normalized: {_mask(rpc_norm)}")
                else:
                    print(f"  RPC decoded but normalize REJECTED. "
                          f"Decoded prefix: {rpc_decoded[:16]!r}")
            else:
                print("  RPC decrypt returned empty.")
                if is_encrypted(rpc_raw):
                    print("  Stored value is encrypted but cannot be decrypted —")
                    print("  APP_SECRETS_KEY changed since you saved, or the env")
                    print("  var isn't loaded on this process. Re-enter the URL")
                    print("  in the SettingsPanel UI to re-encrypt with the")
                    print("  currently-loaded APP_SECRETS_KEY.")
                else:
                    print("  Stored value is plaintext but decrypt_secret returned "
                          "empty — likely an empty/whitespace string.")
        else:
            print("  RPC: no value stored.")

        if ws_raw:
            try:
                ws_decoded = decrypt_secret(ws_raw) or ""
            except Exception as exc:
                print(f"  WS decrypt RAISED: {type(exc).__name__}: {exc}")
                ws_decoded = ""
            if ws_decoded:
                print(f"  WS decrypt OK (len {len(ws_decoded)}) -> {_mask(ws_decoded)}")
            else:
                print("  WS decrypt returned empty.")
        else:
            print("  WS: no value stored.")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
