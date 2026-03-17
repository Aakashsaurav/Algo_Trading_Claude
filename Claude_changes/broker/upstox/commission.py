"""
broker/upstox/commission.py
----------------------------
Exact Upstox charge model — all 7 layers of Indian market costs.
THIS FILE IS LOCKED. Do not modify without explicit approval.
"""
from dataclasses import dataclass
from enum import Enum


class Segment(Enum):
    EQUITY_DELIVERY   = "equity_delivery"
    EQUITY_INTRADAY   = "equity_intraday"
    EQUITY_FUTURES    = "equity_futures"
    EQUITY_OPTIONS    = "equity_options"
    CURRENCY_FUTURES  = "currency_futures"
    CURRENCY_OPTIONS  = "currency_options"
    COMMODITY_FUTURES = "commodity_futures"
    COMMODITY_OPTIONS = "commodity_options"


@dataclass
class ChargeBreakdown:
    segment: str = ""; side: str = ""; quantity: int = 0
    price: float = 0.0; trade_value: float = 0.0
    brokerage: float = 0.0; stt: float = 0.0
    transaction_charge: float = 0.0; sebi_fee: float = 0.0
    gst: float = 0.0; stamp_duty: float = 0.0
    dp_charge: float = 0.0; total: float = 0.0


class CommissionModel:
    BROKERAGE_CAP = 20.0
    BROKERAGE_PCT = {
        Segment.EQUITY_DELIVERY: 0.001, Segment.EQUITY_INTRADAY: 0.0005,
        Segment.EQUITY_FUTURES: 0.0005, Segment.EQUITY_OPTIONS: None,
        Segment.CURRENCY_FUTURES: 0.0005, Segment.CURRENCY_OPTIONS: None,
        Segment.COMMODITY_FUTURES: 0.0005, Segment.COMMODITY_OPTIONS: None,
    }
    STT_RATES = {
        Segment.EQUITY_DELIVERY: (0.001, 0.001),
        Segment.EQUITY_INTRADAY: (0.00025, 0.00025),
        Segment.EQUITY_FUTURES: (0.0, 0.0001250),
        Segment.EQUITY_OPTIONS: (0.000625, 0.00125),
        Segment.CURRENCY_FUTURES: (0.0, 0.0),
        Segment.CURRENCY_OPTIONS: (0.0, 0.0),
        Segment.COMMODITY_FUTURES: (0.0, 0.0001),
        Segment.COMMODITY_OPTIONS: (0.0, 0.0001),
    }
    TRANSACTION_RATES = {
        Segment.EQUITY_DELIVERY: 0.0000297, Segment.EQUITY_INTRADAY: 0.0000297,
        Segment.EQUITY_FUTURES: 0.0000173, Segment.EQUITY_OPTIONS: 0.0003503,
        Segment.CURRENCY_FUTURES: 0.0000035, Segment.CURRENCY_OPTIONS: 0.0003503,
        Segment.COMMODITY_FUTURES: 0.000026, Segment.COMMODITY_OPTIONS: 0.000026,
    }
    SEBI_RATE = 0.0000001
    GST_RATE = 0.18
    STAMP_DUTY_RATES = {
        Segment.EQUITY_DELIVERY: 0.00015, Segment.EQUITY_INTRADAY: 0.000003,
        Segment.EQUITY_FUTURES: 0.000002, Segment.EQUITY_OPTIONS: 0.000003,
        Segment.CURRENCY_FUTURES: 0.000001, Segment.CURRENCY_OPTIONS: 0.000001,
        Segment.COMMODITY_FUTURES: 0.000001, Segment.COMMODITY_OPTIONS: 0.000001,
    }
    DP_CHARGE_PER_SCRIP = 18.5

    def calculate(self, segment: Segment, side: str, quantity: int,
                  price: float, lot_size: int = 1) -> ChargeBreakdown:
        side = side.upper().strip()
        if side not in ("BUY", "SELL"): raise ValueError(f"side must be BUY/SELL, got: {side}")
        if quantity <= 0: raise ValueError(f"quantity must be > 0, got: {quantity}")
        if price <= 0.0: raise ValueError(f"price must be > 0.0, got: {price}")
        tv = quantity * price
        r = ChargeBreakdown(segment=segment.value, side=side, quantity=quantity,
                            price=price, trade_value=tv)
        r.brokerage = min(self.BROKERAGE_CAP, (self.BROKERAGE_PCT[segment] or 0) * tv) \
                      if self.BROKERAGE_PCT[segment] else self.BROKERAGE_CAP
        buy_rate, sell_rate = self.STT_RATES[segment]
        r.stt = tv * (buy_rate if side == "BUY" else sell_rate)
        r.transaction_charge = tv * self.TRANSACTION_RATES[segment]
        r.sebi_fee = tv * self.SEBI_RATE
        r.gst = (r.brokerage + r.transaction_charge) * self.GST_RATE
        r.stamp_duty = tv * self.STAMP_DUTY_RATES[segment] if side == "BUY" else 0.0
        r.dp_charge = self.DP_CHARGE_PER_SCRIP if (segment == Segment.EQUITY_DELIVERY and side == "SELL") else 0.0
        r.total = round(r.brokerage + r.stt + r.transaction_charge +
                        r.sebi_fee + r.gst + r.stamp_duty + r.dp_charge, 2)
        for attr in ("brokerage","stt","transaction_charge","sebi_fee","gst","stamp_duty","dp_charge"):
            setattr(r, attr, round(getattr(r, attr), 2))
        return r


commission_model = CommissionModel()


def infer_segment(instrument_type: str, holding_type: str = "CNC") -> Segment:
    it = instrument_type.upper(); ht = holding_type.upper()
    if it == "EQUITY":
        return Segment.EQUITY_INTRADAY if ht == "MIS" else Segment.EQUITY_DELIVERY
    if it in ("FUTSTK", "FUTIDX"): return Segment.EQUITY_FUTURES
    if it in ("OPTSTK", "OPTIDX"): return Segment.EQUITY_OPTIONS
    if it in ("FUTCUR",): return Segment.CURRENCY_FUTURES
    if it in ("OPTCUR",): return Segment.CURRENCY_OPTIONS
    if it in ("FUTCOM",): return Segment.COMMODITY_FUTURES
    if it in ("OPTCOM",): return Segment.COMMODITY_OPTIONS
    raise ValueError(f"Unknown instrument_type={instrument_type!r}, holding_type={holding_type!r}")