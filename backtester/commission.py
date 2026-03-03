"""
backtester/commission.py
-------------------------
Exact Upstox brokerage and regulatory charge model.

THEORY — WHY CHARGES MATTER IN BACKTESTING:
Every trade in India has multiple layers of cost beyond just the broker's
commission. Missing even one layer can make a losing strategy look profitable.
A ₹20 brokerage on both legs of an intraday trade already costs ₹40.
Add STT, GST, transaction charges, stamp duty, SEBI fees — the real cost
of a round-trip can be ₹80–₹150+ for small trades.

COMPLETE UPSTOX CHARGE STRUCTURE (as of 2025, sourced from upstox.com):
─────────────────────────────────────────────────────────────────────────

A) BROKERAGE (Upstox):
   ┌─────────────────────────┬─────────────────────────────────────────┐
   │ Segment                 │ Brokerage per executed order             │
   ├─────────────────────────┼─────────────────────────────────────────┤
   │ Equity Delivery         │ min(₹20, 0.1% of trade value)           │
   │ Equity Intraday (MIS)   │ min(₹20, 0.05% of trade value)          │
   │ Equity Futures (NRML)   │ min(₹20, 0.05% of trade value)          │
   │ Equity Options          │ ₹20 flat (regardless of premium/value)  │
   │ Currency Futures        │ min(₹20, 0.05% of trade value)          │
   │ Currency Options        │ ₹20 flat                                 │
   │ Commodity Futures       │ min(₹20, 0.05% of trade value)          │
   │ Commodity Options       │ ₹20 flat                                 │
   └─────────────────────────┴─────────────────────────────────────────┘

B) STT / CTT (Securities / Commodity Transaction Tax — Government):
   ┌─────────────────────────┬──────────────┬──────────┬───────────────┐
   │ Segment                 │ Buy Side      │ Sell Side│ Basis          │
   ├─────────────────────────┼──────────────┼──────────┼───────────────┤
   │ Equity Delivery         │ 0.1%         │ 0.1%    │ Trade value    │
   │ Equity Intraday         │ 0.025%       │ 0.025%  │ Trade value    │
   │ Equity Futures          │ 0%           │ 0.0125% │ Trade value    │
   │ Equity Options (buy)    │ 0.0625%      │ 0%      │ Premium value  │
   │ Equity Options (sell)   │ 0%           │ 0.125%  │ Trade value    │
   │ Currency F&O            │ 0%           │ 0%      │ (No STT)       │
   │ Commodity (CTT)         │ 0%           │ 0.01%   │ Trade value    │
   └─────────────────────────┴──────────────┴──────────┴───────────────┘

C) EXCHANGE TRANSACTION CHARGES (NSE, charged on both sides):
   ┌─────────────────────────┬─────────────────────────────────────────┐
   │ Segment                 │ Rate                                     │
   ├─────────────────────────┼─────────────────────────────────────────┤
   │ Equity (EQ)             │ 0.00297% of trade value                 │
   │ Equity Futures          │ 0.00173% of trade value                 │
   │ Equity Options          │ 0.03503% of premium value               │
   │ Currency Futures        │ 0.00035% of trade value                 │
   │ Currency Options        │ 0.03503% of premium value               │
   │ Commodity               │ 0.0026% of trade value                  │
   └─────────────────────────┴─────────────────────────────────────────┘

D) SEBI TURNOVER FEE: ₹10 per crore = 0.000010% of trade value (both sides)

E) GST: 18% of (brokerage + transaction charges)
   Applied per leg (buy and sell separately).

F) STAMP DUTY (Buy side only — uniform across all states since 2020):
   ┌─────────────────────────┬─────────────────────────────────────────┐
   │ Segment                 │ Rate                                     │
   ├─────────────────────────┼─────────────────────────────────────────┤
   │ Equity Delivery         │ 0.015% of trade value                   │
   │ Equity Intraday         │ 0.003% of trade value                   │
   │ Equity Futures          │ 0.002% of trade value                   │
   │ Equity Options          │ 0.003% of premium value                 │
   │ Currency / Commodity    │ 0.0001% of trade value                  │
   └─────────────────────────┴─────────────────────────────────────────┘

G) DP CHARGES (Delivery sell side only): ₹18.5 per scrip (not per share)
   Applied only when SELLING shares from your Demat account (delivery trades).

USAGE:
    from backtester.commission import CommissionModel, Segment

    model = CommissionModel()

    # For a buy trade: 100 shares of INFY at ₹1500 each (delivery)
    charges = model.calculate(
        segment=Segment.EQUITY_DELIVERY,
        side="BUY",
        quantity=100,
        price=1500.0,
        lot_size=1
    )
    print(charges)
    # {'brokerage': 20.0, 'stt': 150.0, 'transaction': 4.46, ...}

    # For options: 1 lot of NIFTY 22500 CE at ₹200 premium, lot_size=50
    charges = model.calculate(
        segment=Segment.EQUITY_OPTIONS,
        side="BUY",
        quantity=50,   # 1 lot
        price=200.0,   # premium price per unit
        lot_size=50
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Segment enum — maps instrument types to charge categories
# ---------------------------------------------------------------------------

class Segment(Enum):
    """
    Charge segment for a trade. Use this to correctly route brokerage/STT.

    Mapping from strategy instrument types:
        EQUITY (CNC/delivery)        → EQUITY_DELIVERY
        EQUITY (MIS/intraday)        → EQUITY_INTRADAY
        FUTSTK / FUTIDX              → EQUITY_FUTURES
        OPTSTK / OPTIDX              → EQUITY_OPTIONS
        FUTCUR / OPTCUR              → CURRENCY_FUTURES / CURRENCY_OPTIONS
        FUTCOM / OPTCOM              → COMMODITY_FUTURES / COMMODITY_OPTIONS
    """
    EQUITY_DELIVERY   = "equity_delivery"
    EQUITY_INTRADAY   = "equity_intraday"
    EQUITY_FUTURES    = "equity_futures"
    EQUITY_OPTIONS    = "equity_options"
    CURRENCY_FUTURES  = "currency_futures"
    CURRENCY_OPTIONS  = "currency_options"
    COMMODITY_FUTURES = "commodity_futures"
    COMMODITY_OPTIONS = "commodity_options"


# ---------------------------------------------------------------------------
# Result dataclass — one instance per trade leg (buy or sell)
# ---------------------------------------------------------------------------

@dataclass
class ChargeBreakdown:
    """
    Complete charge breakdown for a single trade leg (buy or sell).

    All values are in Indian Rupees (₹).
    total = brokerage + stt + transaction_charge + sebi_fee + gst + stamp_duty
            + dp_charge (if applicable)
    """
    segment:            str     = ""
    side:               str     = ""        # "BUY" or "SELL"
    quantity:           int     = 0
    price:              float   = 0.0
    trade_value:        float   = 0.0       # quantity × price

    # Individual charges
    brokerage:          float   = 0.0
    stt:                float   = 0.0       # Securities Transaction Tax
    transaction_charge: float   = 0.0       # NSE/BSE exchange fee
    sebi_fee:           float   = 0.0       # ₹10/crore SEBI turnover fee
    gst:                float   = 0.0       # 18% on (brokerage + tx charge)
    stamp_duty:         float   = 0.0       # Buy side only, segment-specific
    dp_charge:          float   = 0.0       # ₹18.5 on delivery sell only

    total:              float   = 0.0       # Sum of all charges above

    def __str__(self) -> str:
        lines = [
            f"Charges for {self.side} {self.quantity} units @ ₹{self.price:.2f} "
            f"[{self.segment}]",
            f"  Trade Value      : ₹{self.trade_value:>10.2f}",
            f"  Brokerage        : ₹{self.brokerage:>10.2f}",
            f"  STT/CTT          : ₹{self.stt:>10.2f}",
            f"  Transaction Chg  : ₹{self.transaction_charge:>10.2f}",
            f"  SEBI Fee         : ₹{self.sebi_fee:>10.2f}",
            f"  GST (18%)        : ₹{self.gst:>10.2f}",
            f"  Stamp Duty       : ₹{self.stamp_duty:>10.2f}",
            f"  DP Charges       : ₹{self.dp_charge:>10.2f}",
            f"  ─────────────────────────────",
            f"  TOTAL CHARGES    : ₹{self.total:>10.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commission model
# ---------------------------------------------------------------------------

class CommissionModel:
    """
    Calculates exact Upstox brokerage + all regulatory charges for any trade.

    Design:
    - Stateless: each calculate() call is independent.
    - One method per charge type: easy to update when rates change.
    - Returns ChargeBreakdown so the backtester can log each component.

    All rates are stored as class constants so they can be updated in one place
    when SEBI/NSE revises charges (which happens periodically).
    """

    # ── Brokerage caps ─────────────────────────────────────────────────────
    BROKERAGE_CAP           = 20.0          # ₹20 maximum per executed order

    # Brokerage percentage rates (used when trade value is very small)
    BROKERAGE_PCT = {
        Segment.EQUITY_DELIVERY:   0.001,    # 0.1%
        Segment.EQUITY_INTRADAY:   0.0005,   # 0.05%
        Segment.EQUITY_FUTURES:    0.0005,   # 0.05%
        Segment.EQUITY_OPTIONS:    None,      # Flat ₹20 always
        Segment.CURRENCY_FUTURES:  0.0005,   # 0.05%
        Segment.CURRENCY_OPTIONS:  None,      # Flat ₹20 always
        Segment.COMMODITY_FUTURES: 0.0005,   # 0.05%
        Segment.COMMODITY_OPTIONS: None,      # Flat ₹20 always
    }

    # ── STT / CTT rates ────────────────────────────────────────────────────
    # (buy_rate, sell_rate) — 0.0 means not charged on that side
    STT_RATES = {
        Segment.EQUITY_DELIVERY:   (0.001,   0.001),    # 0.1% both sides
        Segment.EQUITY_INTRADAY:   (0.00025, 0.00025),  # 0.025% both sides
        Segment.EQUITY_FUTURES:    (0.0,     0.0001250),# 0.0125% sell only
        # Options: buy=0.0625% on premium, sell=0.125% on intrinsic value
        # We approximate sell side as 0.125% on trade_value for simplicity
        Segment.EQUITY_OPTIONS:    (0.000625, 0.00125), # see note above
        Segment.CURRENCY_FUTURES:  (0.0,     0.0),      # No STT on currency
        Segment.CURRENCY_OPTIONS:  (0.0,     0.0),
        Segment.COMMODITY_FUTURES: (0.0,     0.0001),   # CTT 0.01% sell
        Segment.COMMODITY_OPTIONS: (0.0,     0.0001),
    }

    # ── NSE Exchange Transaction Charge rates ──────────────────────────────
    # Applied on BOTH buy and sell sides
    TRANSACTION_RATES = {
        Segment.EQUITY_DELIVERY:   0.0000297,  # 0.00297%
        Segment.EQUITY_INTRADAY:   0.0000297,  # same as equity
        Segment.EQUITY_FUTURES:    0.0000173,  # 0.00173%
        Segment.EQUITY_OPTIONS:    0.0003503,  # 0.03503% on premium
        Segment.CURRENCY_FUTURES:  0.0000035,  # 0.00035%
        Segment.CURRENCY_OPTIONS:  0.0003503,
        Segment.COMMODITY_FUTURES: 0.000026,   # 0.0026%
        Segment.COMMODITY_OPTIONS: 0.000026,
    }

    # ── SEBI Turnover Fee ──────────────────────────────────────────────────
    SEBI_RATE = 0.0000001   # ₹10 per crore = 0.000001% = 0.0000001 (both sides)

    # ── GST ────────────────────────────────────────────────────────────────
    GST_RATE = 0.18         # 18% on (brokerage + transaction charges)

    # ── Stamp Duty rates (buy side only) ──────────────────────────────────
    STAMP_DUTY_RATES = {
        Segment.EQUITY_DELIVERY:   0.00015,    # 0.015%
        Segment.EQUITY_INTRADAY:   0.000003,   # 0.0003% (₹300/crore)
        Segment.EQUITY_FUTURES:    0.000002,   # 0.0002% (₹200/crore)
        Segment.EQUITY_OPTIONS:    0.000003,   # 0.0003% on premium
        Segment.CURRENCY_FUTURES:  0.000001,   # 0.0001%
        Segment.CURRENCY_OPTIONS:  0.000001,
        Segment.COMMODITY_FUTURES: 0.000001,
        Segment.COMMODITY_OPTIONS: 0.000001,
    }

    # ── DP Charges (delivery sell only) ───────────────────────────────────
    DP_CHARGE_PER_SCRIP = 18.5   # ₹18.5 per scrip (not per share)

    # ---------------------------------------------------------------------------

    def calculate(
        self,
        segment:   Segment,
        side:      str,       # "BUY" or "SELL"
        quantity:  int,
        price:     float,
        lot_size:  int = 1,   # for F&O: number of units per lot (used for labels only)
    ) -> ChargeBreakdown:
        """
        Calculate all charges for one executed order leg.

        Args:
            segment:  Charge segment (Segment enum).
            side:     "BUY" or "SELL".
            quantity: Number of shares / units in the executed order.
            price:    Execution price per share/unit.
                      For options: this is the PREMIUM per unit.
            lot_size: Lot size (informational, affects nothing in the math).

        Returns:
            ChargeBreakdown with all charge components and their total.

        Note:
            Call this function TWICE per round trip:
            once for the entry (BUY) and once for the exit (SELL).
            Sum both ChargeBreakdown.total values for the full round-trip cost.
        """
        # -- Validate inputs
        side = side.upper().strip()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got: '{side}'")
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got: {quantity}")
        if price <= 0.0:
            raise ValueError(f"price must be > 0.0, got: {price}")

        trade_value = quantity * price   # Total transaction value (₹)

        result = ChargeBreakdown(
            segment    = segment.value,
            side       = side,
            quantity   = quantity,
            price      = price,
            trade_value= trade_value,
        )

        # 1. Brokerage
        result.brokerage = self._brokerage(segment, trade_value)

        # 2. STT / CTT
        result.stt = self._stt(segment, side, trade_value)

        # 3. Exchange Transaction Charge
        result.transaction_charge = self._transaction(segment, trade_value)

        # 4. SEBI Turnover Fee
        result.sebi_fee = self._sebi(trade_value)

        # 5. GST (on brokerage + transaction charge)
        result.gst = (result.brokerage + result.transaction_charge) * self.GST_RATE

        # 6. Stamp Duty (buy side only)
        result.stamp_duty = self._stamp_duty(segment, side, trade_value)

        # 7. DP Charges (delivery sell only — per scrip, not per quantity)
        result.dp_charge = self._dp_charge(segment, side)

        # 8. Total
        result.total = (
            result.brokerage
            + result.stt
            + result.transaction_charge
            + result.sebi_fee
            + result.gst
            + result.stamp_duty
            + result.dp_charge
        )

        # Round everything to 2 decimal places (paise precision)
        for attr in ("brokerage", "stt", "transaction_charge", "sebi_fee",
                     "gst", "stamp_duty", "dp_charge", "total"):
            setattr(result, attr, round(getattr(result, attr), 2))

        return result

    def round_trip_cost(
        self,
        segment:  Segment,
        quantity: int,
        entry_price: float,
        exit_price:  float,
        lot_size: int = 1,
    ) -> dict:
        """
        Convenience method: calculate charges for a complete round trip
        (entry BUY + exit SELL).

        Returns a dict with buy_charges, sell_charges, and total_charges.
        """
        buy  = self.calculate(segment, "BUY",  quantity, entry_price, lot_size)
        sell = self.calculate(segment, "SELL", quantity, exit_price,  lot_size)
        return {
            "buy_charges":   buy,
            "sell_charges":  sell,
            "total_charges": round(buy.total + sell.total, 2),
        }

    # ---------------------------------------------------------------------------
    # Private charge calculation methods
    # ---------------------------------------------------------------------------

    def _brokerage(self, segment: Segment, trade_value: float) -> float:
        """
        Brokerage: min(₹20, pct × trade_value) or flat ₹20 for options.
        """
        pct = self.BROKERAGE_PCT[segment]
        if pct is None:
            return self.BROKERAGE_CAP    # Options: always flat ₹20
        return min(self.BROKERAGE_CAP, pct * trade_value)

    def _stt(self, segment: Segment, side: str, trade_value: float) -> float:
        """
        STT/CTT: rate varies by segment and side (buy vs sell).
        """
        buy_rate, sell_rate = self.STT_RATES[segment]
        rate = buy_rate if side == "BUY" else sell_rate
        return trade_value * rate

    def _transaction(self, segment: Segment, trade_value: float) -> float:
        """
        Exchange transaction charge: applied on both buy and sell.
        """
        return trade_value * self.TRANSACTION_RATES[segment]

    def _sebi(self, trade_value: float) -> float:
        """
        SEBI Turnover Fee: ₹10 per crore = 0.0000001 × trade_value.
        Applied on both sides.
        """
        return trade_value * self.SEBI_RATE

    def _stamp_duty(self, segment: Segment, side: str, trade_value: float) -> float:
        """
        Stamp Duty: buy side only. Rate varies by segment.
        Since 2020, uniform stamp duty applies across all Indian states.
        """
        if side == "SELL":
            return 0.0
        return trade_value * self.STAMP_DUTY_RATES[segment]

    def _dp_charge(self, segment: Segment, side: str) -> float:
        """
        DP (Depository Participant) charge: applies only when SELLING
        equity delivery shares from the Demat account. ₹18.5 per scrip.
        """
        if segment == Segment.EQUITY_DELIVERY and side == "SELL":
            return self.DP_CHARGE_PER_SCRIP
        return 0.0


# ---------------------------------------------------------------------------
# Module-level default instance (import and use directly)
# ---------------------------------------------------------------------------
commission_model = CommissionModel()


# ---------------------------------------------------------------------------
# Segment inference helper — map instrument_type + holding_type → Segment
# ---------------------------------------------------------------------------

def infer_segment(instrument_type: str, holding_type: str = "CNC") -> Segment:
    """
    Automatically determine the correct Segment enum value.

    Args:
        instrument_type (str): From instrument_manager — "EQUITY", "FUTSTK",
                               "FUTIDX", "OPTSTK", "OPTIDX", "FUTCOM",
                               "OPTCOM", "FUTCUR", "OPTCUR".
        holding_type (str):    "MIS" for intraday, "CNC" for delivery,
                               "NRML" for F&O overnight.

    Returns:
        Segment enum value.

    Raises:
        ValueError: If the combination is unrecognised.
    """
    instrument_type = instrument_type.upper().strip()
    holding_type    = holding_type.upper().strip()

    mapping = {
        ("EQUITY", "CNC"):  Segment.EQUITY_DELIVERY,
        ("EQUITY", "NRML"): Segment.EQUITY_DELIVERY,
        ("EQUITY", "MIS"):  Segment.EQUITY_INTRADAY,
        ("INDEX",  "MIS"):  Segment.EQUITY_INTRADAY,

        ("FUTSTK", "NRML"): Segment.EQUITY_FUTURES,
        ("FUTSTK", "MIS"):  Segment.EQUITY_FUTURES,
        ("FUTIDX", "NRML"): Segment.EQUITY_FUTURES,
        ("FUTIDX", "MIS"):  Segment.EQUITY_FUTURES,

        ("OPTSTK", "NRML"): Segment.EQUITY_OPTIONS,
        ("OPTSTK", "MIS"):  Segment.EQUITY_OPTIONS,
        ("OPTIDX", "NRML"): Segment.EQUITY_OPTIONS,
        ("OPTIDX", "MIS"):  Segment.EQUITY_OPTIONS,

        ("FUTCUR", "NRML"): Segment.CURRENCY_FUTURES,
        ("FUTCUR", "MIS"):  Segment.CURRENCY_FUTURES,
        ("OPTCUR", "NRML"): Segment.CURRENCY_OPTIONS,
        ("OPTCUR", "MIS"):  Segment.CURRENCY_OPTIONS,

        ("FUTCOM", "NRML"): Segment.COMMODITY_FUTURES,
        ("FUTCOM", "MIS"):  Segment.COMMODITY_FUTURES,
        ("OPTCOM", "NRML"): Segment.COMMODITY_OPTIONS,
        ("OPTCOM", "MIS"):  Segment.COMMODITY_OPTIONS,
    }

    key = (instrument_type, holding_type)
    if key not in mapping:
        raise ValueError(
            f"Unknown instrument_type + holding_type combination: "
            f"'{instrument_type}' + '{holding_type}'. "
            f"Valid instrument types: EQUITY, INDEX, FUTSTK, FUTIDX, "
            f"OPTSTK, OPTIDX, FUTCUR, OPTCUR, FUTCOM, OPTCOM. "
            f"Valid holding types: MIS, CNC, NRML."
        )
    return mapping[key]
