from services.snapshot_broadcaster import SnapshotBroadcaster


def test_crypto_markets_dedup_sig_changes_when_later_market_price_changes():
    broadcaster = SnapshotBroadcaster()
    base_markets = [
        {
            "id": "btc-5m-1",
            "up_price": 0.51,
            "down_price": 0.49,
            "combined": 1.0,
            "oracle_updated_at_ms": 1_700_000_000_000,
            "updated_at_ms": 1_700_000_000_000,
        },
        {
            "id": "eth-5m-1",
            "up_price": 0.44,
            "down_price": 0.56,
            "combined": 1.0,
            "oracle_updated_at_ms": 1_700_000_000_000,
            "updated_at_ms": 1_700_000_000_000,
        },
    ]
    updated_markets = [dict(row) for row in base_markets]
    updated_markets[1]["up_price"] = 0.46
    updated_markets[1]["down_price"] = 0.54

    base_sig = broadcaster._compute_dedup_sig("crypto_markets_update", {"markets": base_markets})
    updated_sig = broadcaster._compute_dedup_sig("crypto_markets_update", {"markets": updated_markets})

    assert base_sig != updated_sig


def test_crypto_markets_dedup_sig_is_order_invariant():
    broadcaster = SnapshotBroadcaster()
    markets = [
        {
            "id": "eth-5m-1",
            "up_price": 0.44,
            "down_price": 0.56,
            "combined": 1.0,
            "oracle_updated_at_ms": 1_700_000_000_000,
            "updated_at_ms": 1_700_000_000_000,
        },
        {
            "id": "btc-5m-1",
            "up_price": 0.51,
            "down_price": 0.49,
            "combined": 1.0,
            "oracle_updated_at_ms": 1_700_000_000_000,
            "updated_at_ms": 1_700_000_000_000,
        },
    ]
    reordered = list(reversed(markets))

    sig_a = broadcaster._compute_dedup_sig("crypto_markets_update", {"markets": markets})
    sig_b = broadcaster._compute_dedup_sig("crypto_markets_update", {"markets": reordered})

    assert sig_a == sig_b
