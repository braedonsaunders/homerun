from services.market_cache import CachedMarket, CachedUsername, _utcnow_naive


def _default_value(column):
    default_callable = column.default.arg
    try:
        return default_callable()
    except TypeError:
        return default_callable(None)


def test_utcnow_naive_returns_tz_naive_datetime():
    value = _utcnow_naive()
    assert value.tzinfo is None


def test_cached_market_and_username_defaults_are_tz_naive():
    market_cached_at = _default_value(CachedMarket.__table__.c.cached_at)
    market_updated_at = _default_value(CachedMarket.__table__.c.updated_at)
    username_cached_at = _default_value(CachedUsername.__table__.c.cached_at)
    username_updated_at = _default_value(CachedUsername.__table__.c.updated_at)

    assert market_cached_at.tzinfo is None
    assert market_updated_at.tzinfo is None
    assert username_cached_at.tzinfo is None
    assert username_updated_at.tzinfo is None
