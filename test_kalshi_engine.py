#!/usr/bin/env python3
"""
test_kalshi_engine.py  --  offline checks of kalshi_engine's decision rules (no network).

Run: python test_kalshi_engine.py      (exits non-zero on the first failure)
"""

import argparse
import base64
import math
import os
import tempfile
import time

import kalshi_engine as K


class FakeSurface:
    """Stands in for the Deribit surface: P(YES) chosen per strike, scaled by the vol multiplier."""
    def __init__(self, spot, probs):
        self.spot, self.ok, self.probs = spot, True, probs


def fake_model_prob(surf, p, t, vol_mult):
    base = surf.probs[(p["kind"], p["k1"], p["k2"])]
    # tails get likelier with more vol, middles less: enough to exercise the cautious band
    return min(0.999, base * vol_mult) if base < 0.3 else max(0.001, base / vol_mult)


def args(**kw):
    d = dict(series=["KXBTCD", "KXBTC"], allow_maker_fees=False, balance_includes_resting=False, edge=0.02,
             min_hours=1.0, max_hours=72, min_yes=0.03, max_yes=0.25, yes_min=0.10, yes_max=0.70,
             yes_min_hours=12, vol_haircut=0.15, vol_bump=0.10, kelly=0.25, cap_market=0.05, cap_event=0.15,
             cap_cluster=0.30, cap_gross=0.90, max_orders=40, daily_stop=0.10, max_dd=0.25)
    d.update(kw)
    return argparse.Namespace(**d)


class MemStore(K.Store):
    def __init__(self):
        super().__init__(":memory:")


def market(ticker, kind, k1, k2, hours, bid, ask, event="E1"):
    st = {"above": "greater", "below": "less", "between": "between"}[kind]
    return {"ticker": ticker, "event_ticker": event, "strike_type": st, "floor_strike": k1 if kind != "below" else None,
            "cap_strike": k2 if kind == "between" else (k1 if kind == "below" else None),
            "close_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + hours * 3600)),
            "yes_bid_dollars": f"{bid:.2f}" if bid is not None else None,
            "yes_ask_dollars": f"{ask:.2f}" if ask is not None else None, "price_level_structure": "linear_cent"}


def engine(markets, probs, a=None, spot=85000):
    a = a or args()
    e = K.Engine(a, None, None, MemStore())
    e.surfaces = {"BTC": FakeSurface(spot, probs)}
    e.fee_ok = {s: True for s in a.series}
    u = {}
    for m in markets:
        p = e.parsed(m, "BTC")
        hours = (K.ts(m["close_time"]) - time.time()) / 3600
        u[m["ticker"]] = {"m": m, "p": p, "hours": hours, "event": m["event_ticker"],
                          "bid": K.f(m["yes_bid_dollars"], None), "ask": K.f(m["yes_ask_dollars"], None)}
    return e, u


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def test_signing():
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    key = ed25519.Ed25519PrivateKey.generate()
    path = tempfile.mktemp()
    open(path, "wb").write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                             serialization.NoEncryption()))
    api = K.Kalshi(K.BASES["demo"], "kid", path)
    os.remove(path)
    h = api.sign("delete", "/portfolio/events/orders/abc")
    key.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
                            (h["KALSHI-ACCESS-TIMESTAMP"] + "DELETE/trade-api/v2/portfolio/events/orders/abc").encode())
    check(True, "signature covers timestamp + METHOD + full path without query")


def test_quotes():
    K.CL.model_prob = fake_model_prob
    probs = {("above", 90000.0, None): 0.04,      # far tail: market pays 12c for a ~4% event
             ("above", 86000.0, None): 0.45,      # near the money: no NO quote (YES 45c > max_yes)
             ("between", 84000.0, 86000.0): 0.50, # middle range, underbid
             ("between", 88000.0, 90000.0): 0.06}
    ms = [market("T90", "above", 90000.0, None, 20, 0.10, 0.12),
          market("T86", "above", 86000.0, None, 20, 0.44, 0.46),
          market("R84", "between", 84000.0, 86000.0, 20, 0.30, 0.40, "E2"),
          market("R84-late", "between", 84000.0, 86000.0, 6, 0.30, 0.40, "E3"),
          market("R88", "between", 88000.0, 90000.0, 20, 0.10, 0.13, "E4")]
    e, u = engine(ms, probs)
    e._avail = [1000.0]
    want = e.desired(u, {}, 1000.0, e.exposure(u, {}))
    band_t90 = e.fair_band(u["T90"]["p"], u["T90"]["hours"])
    px, n = want[("T90", "ask", False)]
    check(px >= math.ceil((band_t90[2] + 0.02) * 100) / 100 - 1e-9, f"NO quote on T90: YES ask {px} >= cautious fair {band_t90[2]:.3f} + 2c")
    check(px > 0.10, "NO quote does not cross the best YES bid (post_only safe)")
    check(px >= 0.11, "NO quote joins/improves the ask by at most a cent")
    check(("T86", "ask", False) not in want, "no NO quote on a 45c near-the-money market (outside 3-25c)")
    pxb, nb = want[("R84", "bid", False)]
    band_r = e.fair_band(u["R84"]["p"], u["R84"]["hours"])
    check(band_r[0] - pxb >= 0.02 - 1e-9 and pxb < 0.40, f"YES bid on 20h range at {pxb} <= cautious fair {band_r[0]:.3f} - 2c, below ask")
    check(("R84-late", "bid", False) not in want, "no YES bid on a range with < 12h left")
    check(("R88", "bid", False) not in want, "no penny YES bid on a tail range (our price must be 10-70c)")
    total = sum(p * c if s == "bid" else (1 - p) * c for (t, s, ro), (p, c) in want.items())
    check(total <= 0.90 * 1000 + 1e-6, f"total entry cost {total:.2f} within gross cap 900")
    for (t, s, ro), (p, c) in want.items():
        cost = p * c if s == "bid" else (1 - p) * c
        check(cost <= 0.05 * 1000 + 1e-6, f"{t} {s}: cost {cost:.2f} within per-market cap 50")
    return e, u


def test_exit_and_halt():
    e, u = test_quotes()
    positions = {"T90": ("NO", 40, 35.2)}
    e._avail = [1000.0]
    want = e.desired(u, positions, 1000.0, e.exposure(u, positions))
    mid = e.fair_band(u["T90"]["p"], u["T90"]["hours"])[1]
    px, n = want[("T90", "bid", True)]
    check(n == 40 and abs(px - math.floor(mid * 100) / 100) < 1e-9 and px < 0.12,
          f"held NO gets a reduce-only YES bid at fair ({px}) for all 40 contracts, below the ask")
    e.s.put("halted", "test")
    e._avail = [1000.0]
    want = e.desired(u, positions, 1000.0, e.exposure(u, positions))
    check(set(k for k in want if not k[2]) == set(), "halted: no entry orders")
    check(("T90", "bid", True) in want, "halted: exits still rest")


def test_cluster_cap():
    K.CL.model_prob = fake_model_prob
    probs = {("above", 90000.0 + 500 * i, None): 0.03 for i in range(12)}
    ms = [market(f"U{i}", "above", 90000.0 + 500 * i, None, 20, 0.08, 0.10, f"E{i}") for i in range(12)]
    e, u = engine(ms, probs, args(cap_cluster=0.10, cap_event=1, cap_market=1, kelly=10))
    e._avail = [1000.0]
    want = e.desired(u, {}, 1000.0, e.exposure(u, {}))
    total = sum((1 - p) * c for (t, s, ro), (p, c) in want.items())
    check(total <= 100 + 1e-6, f"12 'BTC up' tails capped together at 10% of $1000: {total:.2f}")


def test_cash_cap():
    K.CL.model_prob = fake_model_prob
    probs = {("above", 90000.0 + 500 * i, None): 0.03 for i in range(12)}
    ms = [market(f"U{i}", "above", 90000.0 + 500 * i, None, 20, 0.08, 0.10, f"E{i}") for i in range(12)]
    e, u = engine(ms, probs, args(cap_cluster=1, cap_event=1, cap_market=1, kelly=10))
    e._avail = [37.0]
    want = e.desired(u, {}, 1000.0, e.exposure(u, {}))
    total = sum((1 - p) * c for (t, s, ro), (p, c) in want.items())
    check(total <= 37 + 1e-6, f"orders never exceed spendable cash ($37): {total:.2f}")


def test_paper_fills():
    class FakeAPI:
        def __init__(self):
            self.tr = {}
        def trades(self, ticker, min_ts):
            return self.tr.get(ticker, [])
        def market(self, ticker):
            return {"result": "no"}
    api, st = FakeAPI(), MemStore()
    b = K.PaperBroker(api, st, 100.0, 0.5)
    oid = b.place("T90", "ask", 0.12, 20)              # NO at 88c
    now = time.time() + 5
    stamp = lambda dt: time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now + dt)) + ".000000Z"
    api.tr["T90"] = [{"created_time": stamp(1), "yes_price_dollars": "0.1100", "count_fp": "50", "taker_outcome_side": "yes"},
                     {"created_time": stamp(2), "yes_price_dollars": "0.1300", "count_fp": "50", "taker_outcome_side": "no"},
                     {"created_time": stamp(3), "yes_price_dollars": "0.1200", "count_fp": "10", "taker_outcome_side": "yes"}]
    b.s.put("paper_orders", {**b.s.get("paper_orders")})
    o = b.s.get("paper_orders")
    o[oid]["placed"] = now
    b.s.put("paper_orders", o)
    api_market_before = b.s.get("paper_cash")
    b.sync(st)
    check(b.s.get("paper_pos") == {} , "position settled the same cycle (fake market resolved NO)")
    rows = st.db.execute("select kind, n, price, pnl from events order by rowid").fetchall()
    fills = [r for r in rows if r["kind"] == "PAPER-FILL"]
    check(len(fills) == 1 and fills[0]["n"] == 5 and abs(fills[0]["price"] - 0.88) < 1e-9,
          "only the taker-YES trade at >= our ask filled us: 5 = 50% of 10 contracts, at our 88c NO price")
    settle = [r for r in rows if r["kind"] == "PAPER-SETTLE"][0]
    check(abs(settle["pnl"] - 5 * 0.12) < 1e-9, "settled NO pays $1 each: +0.60 on 5 contracts bought at 88c")
    check(abs(b.s.get("paper_cash") - (api_market_before - 5 * 0.88 + 5)) < 1e-9, "paper cash = start - cost + payout")


if __name__ == "__main__":
    for t in (test_signing, test_quotes, test_exit_and_halt, test_cluster_cap, test_cash_cap, test_paper_fills):
        print(t.__name__)
        t()
    print("\nall checks passed")
