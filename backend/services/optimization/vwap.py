"""
Volume-Weighted Average Price (VWAP) analysis for execution planning.

Instead of assuming fills at quoted mid prices, this module calculates
expected execution price across order book depth, providing realistic
profit estimates.

Based on methodology from "Unravelling the Probabilistic Forest" paper:
- Calculate VWAP from trades within each Polygon block (~2 seconds)
- Only count arbitrage if |VWAP_yes + VWAP_no - 1.0| > 0.02
- Account for slippage and fill probability
"""

from dataclasses import dataclass


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

    @classmethod
    def from_clob_response(cls, data: dict) -> "OrderBook":
        """Create OrderBook from Polymarket CLOB API response."""
        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        return cls(bids=bids, asks=asks)


@dataclass
class VWAPResult:
    """Result of VWAP calculation."""
    vwap: float
    total_available: float  # USD value available at this VWAP
    slippage_bps: float  # Slippage from mid in basis points
    levels_consumed: int
    fill_probability: float


class VWAPCalculator:
    """
    Calculate Volume-Weighted Average Price for order execution.

    Research methodology:
    - Calculate VWAP from all trades in each Polygon block (~2s)
    - Only count arbitrage if |VWAP_yes + VWAP_no - 1.0| > 0.02
    - Use VWAP to estimate realistic execution prices
    """

    def __init__(self, min_profit_threshold: float = 0.05):
        """
        Args:
            min_profit_threshold: Minimum profit margin to consider valid.
                                  Research paper used $0.05 to account for
                                  execution risk.
        """
        self.min_profit_threshold = min_profit_threshold

    def calculate_buy_vwap(
        self,
        order_book: OrderBook,
        size_usd: float
    ) -> VWAPResult:
        """
        Calculate VWAP for buying a given USD amount.

        Walks through ask side of order book to determine actual
        execution price accounting for depth.

        Args:
            order_book: Order book with bids and asks
            size_usd: Total USD to spend

        Returns:
            VWAPResult with weighted average price and metrics
        """
        if not order_book.asks:
            return VWAPResult(
                vwap=0,
                total_available=0,
                slippage_bps=0,
                levels_consumed=0,
                fill_probability=0
            )

        remaining_usd = size_usd
        total_shares = 0
        total_cost = 0
        levels_consumed = 0

        for level in order_book.asks:
            if remaining_usd <= 0:
                break

            level_value = level.price * level.size

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
                slippage_bps=0,
                levels_consumed=0,
                fill_probability=0
            )

        vwap = total_cost / total_shares

        # Calculate slippage from mid price
        best_ask = order_book.asks[0].price if order_book.asks else vwap
        best_bid = order_book.bids[0].price if order_book.bids else vwap
        mid = (best_ask + best_bid) / 2 if (best_ask + best_bid) > 0 else vwap
        slippage_bps = ((vwap - mid) / mid * 10000) if mid > 0 else 0

        # Fill probability based on liquidity
        fill_probability = min(1.0, total_cost / size_usd) if size_usd > 0 else 1.0

        return VWAPResult(
            vwap=vwap,
            total_available=total_cost,
            slippage_bps=slippage_bps,
            levels_consumed=levels_consumed,
            fill_probability=fill_probability
        )

    def calculate_sell_vwap(
        self,
        order_book: OrderBook,
        size_shares: float
    ) -> VWAPResult:
        """
        Calculate VWAP for selling a given number of shares.

        Walks through bid side of order book.
        """
        if not order_book.bids:
            return VWAPResult(
                vwap=0,
                total_available=0,
                slippage_bps=0,
                levels_consumed=0,
                fill_probability=0
            )

        remaining_shares = size_shares
        total_shares_sold = 0
        total_proceeds = 0
        levels_consumed = 0

        for level in order_book.bids:
            if remaining_shares <= 0:
                break

            if level.size <= remaining_shares:
                total_shares_sold += level.size
                total_proceeds += level.size * level.price
                remaining_shares -= level.size
            else:
                total_shares_sold += remaining_shares
                total_proceeds += remaining_shares * level.price
                remaining_shares = 0

            levels_consumed += 1

        if total_shares_sold == 0:
            return VWAPResult(
                vwap=order_book.bids[0].price if order_book.bids else 0,
                total_available=0,
                slippage_bps=0,
                levels_consumed=0,
                fill_probability=0
            )

        vwap = total_proceeds / total_shares_sold

        best_ask = order_book.asks[0].price if order_book.asks else vwap
        best_bid = order_book.bids[0].price if order_book.bids else vwap
        mid = (best_ask + best_bid) / 2 if (best_ask + best_bid) > 0 else vwap
        slippage_bps = ((mid - vwap) / mid * 10000) if mid > 0 else 0

        fill_probability = min(1.0, total_shares_sold / size_shares) if size_shares > 0 else 1.0

        return VWAPResult(
            vwap=vwap,
            total_available=total_proceeds,
            slippage_bps=slippage_bps,
            levels_consumed=levels_consumed,
            fill_probability=fill_probability
        )

    def estimate_arbitrage_profit(
        self,
        yes_book: OrderBook,
        no_book: OrderBook,
        size_usd: float,
        fee_rate: float = 0.02
    ) -> dict:
        """
        Estimate realistic arbitrage profit accounting for order book depth.

        This is the key insight from the research: naive mid-price calculations
        overestimate profits. VWAP gives realistic execution estimates.

        Args:
            yes_book: Order book for YES token
            no_book: Order book for NO token
            size_usd: Total USD to invest
            fee_rate: Polymarket fee rate (default 2%)

        Returns:
            dict with realistic profit estimates and execution metrics
        """
        # Split capital between YES and NO
        half_size = size_usd / 2

        yes_vwap = self.calculate_buy_vwap(yes_book, half_size)
        no_vwap = self.calculate_buy_vwap(no_book, half_size)

        # Total cost using VWAP (per dollar of exposure)
        total_vwap_cost = yes_vwap.vwap + no_vwap.vwap

        # Minimum achievable position (limited by liquidity)
        achievable_usd = min(
            yes_vwap.total_available,
            no_vwap.total_available
        )

        # Joint fill probability
        fill_probability = yes_vwap.fill_probability * no_vwap.fill_probability

        # Calculate profits
        if total_vwap_cost < 1.0 and achievable_usd > 0:
            # Guaranteed payout is $1 per share
            gross_profit_per_dollar = 1.0 - total_vwap_cost
            fee_per_dollar = 1.0 * fee_rate  # Fee on payout
            net_profit_per_dollar = gross_profit_per_dollar - fee_per_dollar

            # Scale to actual position size
            position_size = min(achievable_usd, size_usd)
            gross_profit = gross_profit_per_dollar * position_size
            net_profit = net_profit_per_dollar * position_size

            # Risk-adjusted expected profit
            expected_profit = net_profit * fill_probability

            roi_percent = (net_profit_per_dollar / total_vwap_cost) * 100
        else:
            gross_profit = 0
            net_profit = 0
            expected_profit = 0
            roi_percent = 0
            position_size = 0

        # Research paper threshold: only count if profit > $0.05 margin
        is_profitable = (
            net_profit > self.min_profit_threshold and
            fill_probability > 0.5 and
            total_vwap_cost < 0.95  # 5% safety margin
        )

        return {
            "yes_vwap": yes_vwap.vwap,
            "no_vwap": no_vwap.vwap,
            "total_vwap_cost": total_vwap_cost,
            "yes_slippage_bps": yes_vwap.slippage_bps,
            "no_slippage_bps": no_vwap.slippage_bps,
            "fill_probability": fill_probability,
            "max_executable_usd": achievable_usd,
            "recommended_size_usd": position_size,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "expected_profit": expected_profit,
            "roi_percent": roi_percent,
            "is_profitable": is_profitable,
            "yes_levels_consumed": yes_vwap.levels_consumed,
            "no_levels_consumed": no_vwap.levels_consumed,
        }

    def get_mid_price(self, order_book: OrderBook) -> float:
        """Get mid price from order book."""
        if not order_book.bids or not order_book.asks:
            if order_book.asks:
                return order_book.asks[0].price
            if order_book.bids:
                return order_book.bids[0].price
            return 0

        return (order_book.bids[0].price + order_book.asks[0].price) / 2

    def get_spread_bps(self, order_book: OrderBook) -> float:
        """Get bid-ask spread in basis points."""
        if not order_book.bids or not order_book.asks:
            return float('inf')

        mid = self.get_mid_price(order_book)
        if mid <= 0:
            return float('inf')

        spread = order_book.asks[0].price - order_book.bids[0].price
        return (spread / mid) * 10000
