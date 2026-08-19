"""The on-chain taker fee off ``OrderFilled`` must survive onto the event.

The decoder always read the ``fee`` word out of the CLOB V2 log, but the
value used to be dropped before persistence.  It is the only ground-truth
fee in the system — Polymarket's REST ``/trades`` payload has no fee field
and no maker/taker flag — so these tests pin both the 6-decimal scaling
and the role attribution that makes the number interpretable.
"""

from __future__ import annotations

import pytest

from services.wallet_ws_monitor import (
    ORDER_FILLED_TOPIC,
    _parse_order_filled_log,
)


MAKER = "0x1111111111111111111111111111111111111111"
TAKER = "0x2222222222222222222222222222222222222222"


def _topic_addr(addr: str) -> str:
    return "0x" + "0" * 24 + addr[2:].lower()


def _word(value: int) -> str:
    return f"{value:064x}"


def _order_filled_log(
    *,
    side: int = 0,
    token_id: int = 12345,
    maker_amount: int = 50_000_000,   # 50 USDC  (6dp)
    taker_amount: int = 100_000_000,  # 100 tokens (6dp)
    fee: int = 1_750_000,             # $1.75    (6dp)
) -> dict:
    data = "0x" + "".join(
        _word(v)
        for v in (side, token_id, maker_amount, taker_amount, fee, 0, 0)
    )
    return {
        "address": "0xE111180000d2663C0091e4f400237545B87B996B",
        "topics": [
            ORDER_FILLED_TOPIC,
            "0x" + _word(999),
            _topic_addr(MAKER),
            _topic_addr(TAKER),
        ],
        "data": data,
        "transactionHash": "0xdeadbeef",
        "logIndex": "0x3",
    }


def test_fee_word_is_decoded_from_the_log():
    parsed = _parse_order_filled_log(_order_filled_log(fee=1_750_000))
    assert parsed is not None
    assert parsed["fee"] == 1_750_000


def test_fee_scales_from_usdc_6dp_to_usd():
    """$1.75 is the documented crypto max at p=0.50 on 100 shares."""
    parsed = _parse_order_filled_log(_order_filled_log(fee=1_750_000))
    assert parsed["fee"] / 1e6 == pytest.approx(1.75)


def test_zero_fee_is_preserved_not_dropped():
    """Geopolitics markets are genuinely fee-free — 0 is a real value."""
    parsed = _parse_order_filled_log(_order_filled_log(fee=0))
    assert parsed is not None
    assert parsed["fee"] == 0


def test_maker_and_taker_are_distinguishable():
    """``fee_usd`` is meaningless without knowing the side of the book:
    only takers are charged, only makers accrue rebates."""
    parsed = _parse_order_filled_log(_order_filled_log())
    assert parsed["maker"].lower() == MAKER.lower()
    assert parsed["taker"].lower() == TAKER.lower()
    assert parsed["maker"].lower() != parsed["taker"].lower()


def test_event_carries_fee_and_role_fields():
    from services.wallet_ws_monitor import WalletTradeEvent

    fields = WalletTradeEvent.__dataclass_fields__
    assert "fee_usd" in fields
    assert "role" in fields
    # Defaults must be safe for RTDS-sourced events, which have neither.
    assert fields["fee_usd"].default == 0.0
    assert fields["role"].default == ""


def test_orm_model_has_fee_and_role_columns():
    from services.wallet_ws_monitor import WalletMonitorEvent

    cols = WalletMonitorEvent.__table__.columns
    assert "fee_usd" in cols
    assert "role" in cols
    # Nullable: pre-capture rows must stay NULL rather than be back-filled
    # with 0.0, which is indistinguishable from a real zero-fee fill.
    assert cols["fee_usd"].nullable
    assert cols["role"].nullable
