#!/usr/bin/env python3
"""
kalshi_engine.py  --  maker-only engine on Kalshi crypto markets, priced off Deribit options

Strategy (validated by kalshi_capacity.py / kalshi_backtest.py on real Kalshi fills and
settlements): crypto takers on Kalshi overpay for tails. Rest an order on the other side, at a
price the options market says beats fair value, and let them come to you.

Every --poll seconds:
  1. FAIR      Deribit options surface for BTC/ETH; each market's YES probability at IV x
               {0.85, 1, 1.1}; each side uses the extreme that hurts it (cautious)
  2. QUOTE     NO  (= resting YES ask) on markets whose YES trades at 3-25c, ask >= cautious
                   fair + --edge, joining/improving the best ask without crossing
               YES (resting bid) on price RANGES with >= --yes-min-hours left, bid <= cautious
                   fair - --edge
               post_only: never takes, never pays the taker fee (crypto series here are
               fee_type 'quadratic' = no maker fee; the engine reads each series' fee type and
               skips any series with maker fees unless --allow-maker-fees)
  3. SIZE      quarter-Kelly, capped per market / event / cluster (asset + the move that hurts)
               / gross, from the account's real balance
  4. EXIT      every position gets a reduce-only resting order at FAIR value: the edge is
               the gap to fair, so when the market pays fair we're out -- no taker fee
  5. MANAGE    requote when the desired price changes, cancel what no longer has edge
  6. GUARD     daily loss / drawdown kill switch (two consecutive cycles), cancel-all on
               halt and on shutdown (SIGTERM)

Kalshi's API is the source of truth for cash, positions and orders (live); Kalshi settles
positions itself. Paper mode fills resting orders only when a real public trade crossed them.

Setup:  pip install curl_cffi cryptography
        Kalshi API key: KALSHI_KEY_ID and KALSHI_KEY_FILE (path to the private key PEM, RSA or
        Ed25519) in .env. Demo keys only work with --env demo, production keys with --env prod.
Run:    python kalshi_engine.py                           # paper, production market data
        python kalshi_engine.py --live --env demo         # real orders on Kalshi's demo exchange
        python kalshi_engine.py --live --env prod         # real money
        python kalshi_engine.py --live --env prod --report
        python kalshi_engine.py --live --env prod --flatten   # cancel all, exit positions at the bid
"""

import argparse
import base64
import json
import math
import os
import signal
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

from curl_cffi import requests

import polymarket_crypto_longshots as CL

BASES = {"prod": "https://external-api.kalshi.com/trade-api/v2",
         "demo": "https://external-api.demo.kalshi.co/trade-api/v2"}
DATA_BASE = BASES["prod"]          # public market data always from production
SERIES = {"KXBTCD": "BTC", "KXETHD": "ETH", "KXBTC": "BTC", "KXETH": "ETH"}


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def cent_up(x):
    return math.ceil(round(x * 100, 6)) / 100


def cent_down(x):
    return math.floor(round(x * 100, 6)) / 100


# ----------------------------------------------------------------- API client
class Kalshi:
    """REST client. Auth per Kalshi docs: headers KALSHI-ACCESS-KEY / -TIMESTAMP (ms) /
    -SIGNATURE = base64(sign(timestamp + METHOD + path-without-query)), RSA-PSS SHA-256
    (salt = digest length) or Ed25519."""

    def __init__(self, base, key_id=None, key_file=None):
        self.base, self.key_id, self.key = base, key_id, None
        self.s = requests.Session(impersonate="chrome")
        if key_file:
            from cryptography.hazmat.primitives import serialization
            if os.name == "posix" and os.stat(key_file).st_mode & 0o077:
                sys.exit(f"{key_file} is readable by other users; run: chmod 600 {key_file}")
            self.key = serialization.load_pem_private_key(open(key_file, "rb").read(), password=None)

    def sign(self, method, path):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding, ed25519
        stamp = str(int(time.time() * 1000))
        msg = (stamp + method.upper() + urlparse(self.base + path).path).encode()
        if isinstance(self.key, ed25519.Ed25519PrivateKey):
            sig = self.key.sign(msg)
        else:
            sig = self.key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                 salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": stamp,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    def req(self, method, path, params=None, body=None, auth=False, base=None, retries=4):
        url = (base or self.base) + path
        for attempt in range(retries):
            headers = {"Content-Type": "application/json"}
            if auth:
                headers.update(self.sign(method, path))
            try:
                r = self.s.request(method, url, params=params, json=body, headers=headers, timeout=20)
            except Exception as e:
                log(f"  {method} {path}: {type(e).__name__}")
                time.sleep(1 + attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
            return r.json() if r.text else {}
        raise RuntimeError(f"{method} {path}: no response after {retries} tries")

    # -- public market data (production)
    def markets(self, series):
        out, cursor = [], None
        for _ in range(20):
            p = {"series_ticker": series, "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = self.req("GET", "/markets", p, base=DATA_BASE)
            out += d.get("markets") or []
            cursor = d.get("cursor")
            if not cursor:
                return out
        return out

    def market(self, ticker):
        return self.req("GET", f"/markets/{ticker}", base=DATA_BASE).get("market") or {}

    def series(self, ticker):
        return self.req("GET", f"/series/{ticker}", base=DATA_BASE).get("series") or {}

    def trades(self, ticker, min_ts):
        return self.req("GET", "/markets/trades", {"ticker": ticker, "min_ts": int(min_ts), "limit": 1000},
                        base=DATA_BASE).get("trades") or []

    # -- account (authenticated, on --env)
    def balance(self):
        return f(self.req("GET", "/portfolio/balance", auth=True).get("balance_dollars"))

    def positions(self):
        out, cursor = [], None
        for _ in range(20):
            p = {"limit": 1000, "count_filter": "position"}
            if cursor:
                p["cursor"] = cursor
            d = self.req("GET", "/portfolio/positions", p, auth=True)
            out += d.get("market_positions") or []
            cursor = d.get("cursor")
            if not cursor:
                return out
        return out

    def resting(self):
        out, cursor = [], None
        for _ in range(20):
            p = {"status": "resting", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = self.req("GET", "/portfolio/orders", p, auth=True)
            out += d.get("orders") or []
            cursor = d.get("cursor")
            if not cursor:
                return out
        return out

    def create(self, ticker, book_side, price, count, reduce_only=False, tif="good_till_canceled"):
        body = {"ticker": ticker, "side": book_side, "price": f"{price:.2f}", "count": str(int(count)),
                "time_in_force": tif, "post_only": tif == "good_till_canceled",
                "self_trade_prevention_type": "taker_at_cross", "reduce_only": reduce_only,
                "client_order_id": str(uuid.uuid4()), "cancel_order_on_pause": True}
        return self.req("POST", "/portfolio/events/orders", body=body, auth=True)

    def cancel(self, order_id, ticker):
        return self.req("DELETE", f"/portfolio/events/orders/{order_id}", {"market_ticker": ticker}, auth=True)


# ----------------------------------------------------------------- brokers
class LiveBroker:
    """Kalshi is the source of truth: no local position bookkeeping to drift."""

    def __init__(self, api):
        self.api = api

    def cash(self):
        return self.api.balance()

    def positions(self):
        """{ticker: (side, contracts, cost)}; Kalshi reports NO as negative position."""
        out = {}
        for p in self.api.positions():
            n = f(p.get("position_fp"))
            if abs(n) >= 1:
                out[p["ticker"]] = ("YES" if n > 0 else "NO", abs(n), f(p.get("market_exposure_dollars")))
        return out

    def orders(self):
        """{order_id: {ticker, book_side, price (YES), remaining, reduce_only}}"""
        return {o["order_id"]: {"ticker": o["ticker"], "book_side": o.get("book_side"),
                                "price": f(o.get("yes_price_dollars")), "remaining": f(o.get("remaining_count_fp")),
                                "reduce_only": bool(o.get("reduce_only"))} for o in self.api.resting()}

    def place(self, ticker, book_side, price, count, reduce_only=False):
        r = self.api.create(ticker, book_side, price, count, reduce_only)
        return r.get("order_id")

    def cancel(self, oid, ticker):
        self.api.cancel(oid, ticker)

    def sync(self, store):
        pass


class PaperBroker:
    """Simulated account on real data: resting orders fill only when a real public trade crossed
    them (taker bought YES at >= our ask / sold YES at <= our bid), --paper-share of its size."""

    def __init__(self, api, store, bankroll, share):
        self.api, self.s, self.share = api, store, share
        if self.s.get("paper_cash") is None:
            self.s.put("paper_cash", bankroll)
            self.s.put("paper_pos", {})
            self.s.put("paper_orders", {})

    def cash(self):
        reserved = sum(o["remaining"] * (o["price"] if o["book_side"] == "bid" else 1 - o["price"])
                       for o in self.s.get("paper_orders").values() if not o["reduce_only"])
        return self.s.get("paper_cash") - reserved

    def positions(self):
        return {t: tuple(v) for t, v in self.s.get("paper_pos").items()}

    def orders(self):
        return dict(self.s.get("paper_orders"))

    def place(self, ticker, book_side, price, count, reduce_only=False):
        oid = "paper-" + uuid.uuid4().hex[:10]
        orders = self.s.get("paper_orders")
        orders[oid] = {"ticker": ticker, "book_side": book_side, "price": price, "remaining": count,
                       "reduce_only": reduce_only, "placed": time.time()}
        self.s.put("paper_orders", orders)
        return oid

    def cancel(self, oid, ticker):
        orders = self.s.get("paper_orders")
        orders.pop(oid, None)
        self.s.put("paper_orders", orders)

    def sync(self, store):
        orders, pos, cash = self.s.get("paper_orders"), self.s.get("paper_pos"), self.s.get("paper_cash")
        for oid, o in list(orders.items()):
            try:
                trades = self.api.trades(o["ticker"], o["placed"])
            except RuntimeError as e:   # one market's feed failing shouldn't stop the others
                log(f"paper: trades for {o['ticker']} unavailable ({e}); retry next cycle")
                continue
            for t in trades:
                if ts(t["created_time"]) < o["placed"] or o["remaining"] < 1:
                    continue
                y, taker = f(t["yes_price_dollars"]), t.get("taker_outcome_side") or t.get("taker_side")
                crossed = (o["book_side"] == "ask" and taker == "yes" and y >= o["price"]) or \
                          (o["book_side"] == "bid" and taker == "no" and y <= o["price"])
                if not crossed:
                    continue
                n = math.floor(min(o["remaining"], f(t["count_fp"]) * self.share))
                if n < 1:
                    continue
                o["remaining"] -= n
                side, unit = ("NO", 1 - o["price"]) if o["book_side"] == "ask" else ("YES", o["price"])
                cur = pos.get(o["ticker"])
                if cur and cur[0] != side:            # reduce-only exit against an open position
                    held, cost = cur[1], cur[2]
                    n = min(n, held)
                    cash += n * (1 - unit)            # selling the held side at the complementary price
                    left = held - n
                    pos[o["ticker"]] = [cur[0], left, cost * left / held] if left >= 1 else None
                    if left < 1:
                        pos.pop(o["ticker"], None)
                    store.event("PAPER-EXIT", o["ticker"], cur[0], n, 1 - unit)
                else:
                    cash -= n * unit
                    pos[o["ticker"]] = [side, (cur[1] if cur else 0) + n, (cur[2] if cur else 0) + n * unit]
                    store.event("PAPER-FILL", o["ticker"], side, n, unit)
                o["placed"] = ts(t["created_time"]) + 1e-3
            if o["remaining"] < 1:
                orders.pop(oid, None)
        for tk in list(pos):                           # settlement
            m = self.api.market(tk)
            if m.get("result") in ("yes", "no"):
                side, n, cost = pos.pop(tk)
                payout = n * (1.0 if (m["result"] == "yes") == (side == "YES") else 0.0)
                cash += payout
                store.event("PAPER-SETTLE", tk, side, n, payout / n if n else 0, pnl=payout - cost)
        self.s.put("paper_orders", orders)
        self.s.put("paper_pos", pos)
        self.s.put("paper_cash", cash)


# ----------------------------------------------------------------- state
class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            create table if not exists kv(k text primary key, v text);
            create table if not exists equity(ts text, cash real, positions real, total real);
            create table if not exists events(ts text, kind text, ticker text, side text, n real, price real, pnl real);
        """)

    def get(self, k, default=None):
        r = self.db.execute("select v from kv where k=?", (k,)).fetchall()
        return json.loads(r[0]["v"]) if r else default

    def put(self, k, v):
        self.db.execute("insert or replace into kv values(?,?)", (k, json.dumps(v)))
        self.db.commit()

    def event(self, kind, ticker, side, n, price, pnl=None):
        self.db.execute("insert into events values(?,?,?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), kind, ticker, side, n, price, pnl))
        self.db.commit()
        log(f"{kind:<12} {side:<3} {n:>7.0f} @ {price:.2f}{'' if pnl is None else f'  pnl {pnl:+.2f}'}  {ticker}")


# ----------------------------------------------------------------- engine
class Engine:
    def __init__(self, a, api, broker, store):
        self.a, self.api, self.b, self.s = a, api, broker, store
        self.surfaces, self.fee_ok = {}, {}

    def check_fees(self):
        for ser in self.a.series:
            s = self.api.series(ser)
            ft = s.get("fee_type")
            self.fee_ok[ser] = ft == "quadratic" or self.a.allow_maker_fees
            log(f"series {ser}: fee_type {ft}{'' if self.fee_ok[ser] else ' -> SKIPPED (maker fees)'}")

    def fair_band(self, p, hours):
        surf = self.surfaces.get(p["asset"])
        if not surf or hours <= 0:
            return None
        t = hours / 24 / 365
        ps = [CL.model_prob(surf, p, t, m) for m in (1 - self.a.vol_haircut, 1.0, 1 + self.a.vol_bump)]
        return min(ps), ps[1], max(ps)

    @staticmethod
    def parsed(m, asset):
        st, lo, hi = m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")
        if st == "greater" and lo is not None:
            return {"asset": asset, "kind": "above", "k1": float(lo), "k2": None}
        if st == "less" and hi is not None:
            return {"asset": asset, "kind": "below", "k1": float(hi), "k2": None}
        if st == "between" and lo is not None and hi is not None:
            return {"asset": asset, "kind": "between", "k1": float(lo), "k2": float(hi)}
        return None

    def cluster(self, p, side):
    # BTC and ETH move together (2026-09-21: +7% / +6% in a day), so up/down is one cluster across assets
        a, spot = p["asset"], self.surfaces[p["asset"]].spot
        if p["kind"] == "above":
            return "crypto:down" if side == "YES" else "crypto:up"
        if p["kind"] == "below":
            return "crypto:up" if side == "YES" else "crypto:down"
        if side == "YES":
            return f"{a}:range"
        return "crypto:up" if p["k1"] > spot else "crypto:down" if p["k2"] < spot else f"{a}:pin"

    def universe(self):
        out = {}
        now = time.time()
        for ser in self.a.series:
            if not self.fee_ok.get(ser):
                continue
            asset = SERIES[ser]
            for m in self.api.markets(ser):
                p = self.parsed(m, asset)
                if not p or m.get("price_level_structure", "linear_cent") != "linear_cent":
                    continue
                hours = (ts(m["close_time"]) - now) / 3600
                if self.a.min_hours <= hours <= self.a.max_hours:
                    out[m["ticker"]] = {"m": m, "p": p, "hours": hours, "event": m["event_ticker"],
                                        "bid": f(m.get("yes_bid_dollars"), None), "ask": f(m.get("yes_ask_dollars"), None)}
        return out

    def desired(self, u, positions, equity, exposure):
        """{(ticker, book_side, reduce_only): (price, count)}"""
        want = {}
        by_mkt, by_evt, by_clu, gross = exposure
        for tk, x in u.items():
            band = self.fair_band(x["p"], x["hours"])
            if not band:
                continue
            lo, mid, hi = band
            bid, ask = x["bid"], x["ask"]
            held = positions.get(tk)
            # exit: rest at fair value for everything we hold (reduce-only, maker)
            if held:
                side, n, _ = held
                if side == "NO":   # sell NO = buy YES at the YES price that values our NO at fair
                    px = cent_down(mid)
                    if 0.01 <= px <= 0.99 and (ask is None or px < ask):
                        want[(tk, "bid", True)] = (px, n)
                else:
                    px = cent_up(mid)
                    if 0.01 <= px <= 0.99 and (bid is None or px > bid):
                        want[(tk, "ask", True)] = (px, n)
            if self.s.get("halted"):
                continue
            # entries
            ref = ask if ask is not None else bid
            if ref is not None and self.a.min_yes <= ref < self.a.max_yes and (not held or held[0] == "NO"):
                px = cent_up(hi + self.a.edge)                       # our YES ask (= NO at 1 - px)
                if ask is not None:
                    px = max(px, round(ask - 0.01, 2))
                if bid is not None and px <= bid:
                    px = round(bid + 0.01, 2)
                if px < 0.99:
                    self._size(want, tk, x, "ask", px, 1 - px, 1 - hi, equity, by_mkt, by_evt, by_clu, gross)
            if x["p"]["kind"] == "between" and x["hours"] >= self.a.yes_min_hours and bid is not None \
                    and self.a.yes_min <= bid < self.a.yes_max and (not held or held[0] == "YES"):
                px = min(cent_down(lo - self.a.edge), round(bid + 0.01, 2))
                if ask is not None and px >= ask:
                    px = round(ask - 0.01, 2)
                # the 10-70c band applies to OUR price (as in kalshi_backtest), not the market's bid:
                # no penny YES bids on tail ranges, where the model is least reliable
                if self.a.yes_min <= px < self.a.yes_max and lo - px >= self.a.edge:
                    self._size(want, tk, x, "bid", px, px, lo, equity, by_mkt, by_evt, by_clu, gross)
        return want

    def _size(self, want, tk, x, book_side, px, unit, q, equity, by_mkt, by_evt, by_clu, gross):
        side = "NO" if book_side == "ask" else "YES"
        clu = self.cluster(x["p"], side)
        kelly = self.a.kelly * max(q - unit, 0) / max(1 - unit, 1e-9)
        budget = min(kelly * equity,
                     self.a.cap_market * equity - by_mkt[tk],
                     self.a.cap_event * equity - by_evt[x["event"]],
                     self.a.cap_cluster * equity - by_clu[clu],
                     self.a.cap_gross * equity - gross[0],
                     self._avail[0])
        n = math.floor(budget / unit) if unit > 0 else 0
        if n >= 1:
            want[(tk, book_side, False)] = (px, n)
            cost = n * unit
            by_mkt[tk] += cost
            by_evt[x["event"]] += cost
            by_clu[clu] += cost
            gross[0] += cost
            self._avail[0] -= cost

    def exposure(self, u, positions):
        by_mkt, by_evt, by_clu, gross = defaultdict(float), defaultdict(float), defaultdict(float), [0.0]
        for tk, (side, n, cost) in positions.items():
            by_mkt[tk] += cost
            if tk in u:
                by_evt[u[tk]["event"]] += cost
                by_clu[self.cluster(u[tk]["p"], side)] += cost
            gross[0] += cost
        return by_mkt, by_evt, by_clu, gross

    def mark(self, u, positions):
        total = 0.0
        for tk, (side, n, cost) in positions.items():
            band = self.fair_band(u[tk]["p"], u[tk]["hours"]) if tk in u else None
            total += n * ((band[1] if side == "YES" else 1 - band[1]) if band else cost / max(n, 1))
        return total

    def guard(self, cash, pos_value):
        total = cash + pos_value
        self.s.db.execute("insert into equity values(?,?,?,?)",
                          (datetime.now(timezone.utc).isoformat(), cash, pos_value, total))
        self.s.db.commit()
        peak = max(self.s.get("peak") or total, total)
        self.s.put("peak", peak)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.s.get("day") != day:
            self.s.put("day", day)
            self.s.put("day_start", total)
        start = self.s.get("day_start") or total
        breach = (f"daily loss {total / start - 1:+.1%}" if total < start * (1 - self.a.daily_stop) else
                  f"drawdown {total / peak - 1:+.1%}" if total < peak * (1 - self.a.max_dd) else None)
        strikes = self.s.get("strikes", 0) + 1 if breach else 0
        self.s.put("strikes", strikes)
        if breach and strikes >= 2 and not self.s.get("halted"):
            self.s.put("halted", breach)
            log(f"KILL SWITCH: {breach} -- entries stopped, exits keep resting")
        return total

    def cycle(self):
        self.surfaces = {a: CL.Surface(a) for a in {SERIES[s] for s in self.a.series}}
        self.surfaces = {a: s for a, s in self.surfaces.items() if s.ok}
        self.b.sync(self.s)
        u = self.universe()
        positions, orders = self.b.positions(), self.b.orders()
        # Kalshi's order listing has no reduce_only flag: an order against the side we hold is
        # our exit (entries never oppose a held position), so classify by that.
        for o in orders.values():
            held = positions.get(o["ticker"])
            o["reduce_only"] = bool(held) and (held[0] == "NO") == (o["book_side"] == "bid")
        cash = self.b.cash()   # AVAILABLE balance: excludes funds held by resting buy orders
        reserved = sum(o["remaining"] * (1 - o["price"] if o["book_side"] == "ask" else o["price"])
                       for o in orders.values() if not o["reduce_only"])
        held = 0.0 if self.a.balance_includes_resting else reserved
        total = self.guard(cash + held, self.mark(u, positions))
        self._avail = [cash + held]  # the order set is rebuilt from scratch, so current holds are spendable
        want = self.desired(u, positions, total, self.exposure(u, positions))
        have = {}
        for oid, o in orders.items():
            have.setdefault((o["ticker"], o["book_side"], o["reduce_only"]), []).append((oid, o))
        placed = cancelled = 0
        for key, lst in have.items():
            target = want.get(key)
            for oid, o in lst:
                if target and abs(o["price"] - target[0]) < 1e-9 and o["remaining"] <= target[1] + 1e-9 and \
                        o is lst[0][1]:
                    want.pop(key)                       # already resting at the right price
                    continue
                self.b.cancel(oid, o["ticker"])
                cancelled += 1
        for (tk, book_side, ro), (px, n) in sorted(want.items(), key=lambda kv: kv[0][2], reverse=True):
            if placed >= self.a.max_orders:
                break
            try:
                self.b.place(tk, book_side, px, n, ro)
                placed += 1
                self.s.event("EXIT-ORDER" if ro else "ORDER", tk, "NO" if book_side == "ask" else "YES",
                             n, 1 - px if book_side == "ask" else px)
            except RuntimeError as e:
                log(f"order rejected {tk}: {e}")
        log(f"equity {total:.2f} (cash {cash:.2f}), {len(positions)} positions, {len(orders) - cancelled + placed} "
            f"resting, +{placed} placed / -{cancelled} cancelled, {len(u)} markets"
            + (f"  HALTED: {self.s.get('halted')}" if self.s.get("halted") else ""))

    def cancel_all(self):
        for oid, o in self.b.orders().items():
            try:
                self.b.cancel(oid, o["ticker"])
            except RuntimeError as e:
                log(f"cancel failed {oid}: {e}")

    def flatten(self):
        """Cancel everything, then exit each position immediately at the best bid (taker, pays fee)."""
        self.cancel_all()
        for tk, (side, n, cost) in self.b.positions().items():
            m = self.api.market(tk)
            if side == "NO":   # buy YES at the ask to close NO
                px = f(m.get("yes_ask_dollars"), None)
                book = "bid"
            else:
                px = f(m.get("yes_bid_dollars"), None)
                book = "ask"
            if px is None:
                log(f"no {('asks' if side == 'NO' else 'bids')} on {tk}; left to settle")
                continue
            self.api.create(tk, book, px, n, reduce_only=True, tif="immediate_or_cancel")
            log(f"FLATTEN {side} {n:.0f} on {tk} at YES {px:.2f}")


def report(store, broker):
    log(f"cash {broker.cash():.2f}")
    for tk, (side, n, cost) in broker.positions().items():
        print(f"  {side:<3} {n:>7.0f}  cost {cost:>8.2f}  {tk}")
    rows = store.db.execute("select * from events where kind like '%SETTLE%' or kind like '%EXIT%' order by ts").fetchall()
    pnl = sum(r["pnl"] or 0 for r in rows)
    print(f"  logged settlements/exits: {len(rows)}, settled P&L (paper) {pnl:+.2f}")
    eq = store.db.execute("select * from equity order by ts").fetchall()
    if eq:
        print(f"  equity {eq[0]['total']:.2f} at {eq[0]['ts'][:16]} -> {eq[-1]['total']:.2f} at {eq[-1]['ts'][:16]}"
              + (f"  HALTED: {store.get('halted')}" if store.get("halted") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="real orders (default: paper on real market data)")
    ap.add_argument("--env", choices=list(BASES), default="demo", help="exchange for orders/account (live)")
    ap.add_argument("--db", help="state file (default kalshi_<paper|demo|prod>.db)")
    ap.add_argument("--bankroll", type=float, default=1000, help="paper starting cash")
    ap.add_argument("--paper-share", type=float, default=0.25, help="paper: share of a crossing trade we get")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--flatten", action="store_true", help="cancel all orders and exit all positions now")
    ap.add_argument("--resume", action="store_true", help="clear a kill-switch halt")
    ap.add_argument("--series", nargs="+", default=["KXBTCD", "KXETHD"],
                    help="range series (KXBTC/KXETH) lost money in the backtest; above/below only by default")
    ap.add_argument("--allow-maker-fees", action="store_true")
    ap.add_argument("--balance-includes-resting", action="store_true",
                    help="set if a demo test shows Kalshi's balance does NOT drop when a buy order rests")
    ap.add_argument("--edge", type=float, default=0.02)
    ap.add_argument("--min-hours", type=float, default=1.0, help="no quotes in a market's last hour by default")
    ap.add_argument("--max-hours", type=float, default=72)
    ap.add_argument("--min-yes", type=float, default=0.03)
    ap.add_argument("--max-yes", type=float, default=0.25)
    ap.add_argument("--yes-min", type=float, default=0.10)
    ap.add_argument("--yes-max", type=float, default=0.70)
    ap.add_argument("--yes-min-hours", type=float, default=12)
    ap.add_argument("--vol-haircut", type=float, default=0.15)
    ap.add_argument("--vol-bump", type=float, default=0.10)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--cap-market", type=float, default=0.05)
    ap.add_argument("--cap-event", type=float, default=0.15)
    ap.add_argument("--cap-cluster", type=float, default=0.15,
                    help="cap per direction across BTC+ETH; bounds a 2026-09-21-style rally to about this loss")
    ap.add_argument("--cap-gross", type=float, default=0.90)
    ap.add_argument("--max-orders", type=int, default=40, help="new orders per cycle")
    ap.add_argument("--daily-stop", type=float, default=0.10)
    ap.add_argument("--max-dd", type=float, default=0.25)
    a = ap.parse_args()

    load_env()
    mode = a.env if a.live else "paper"
    store = Store(a.db or f"kalshi_{mode}.db")
    if a.resume:
        store.put("halted", None)
        store.put("strikes", 0)
        store.put("day", None)
        log("kill switch cleared")
    if a.live:
        kid, kf = os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE")
        if not (kid and kf):
            sys.exit("live mode needs KALSHI_KEY_ID and KALSHI_KEY_FILE in .env")
        api = Kalshi(BASES[a.env], kid, kf)
        broker = LiveBroker(api)
    else:
        api = Kalshi(BASES["prod"])
        broker = PaperBroker(api, store, a.bankroll, a.paper_share)
    eng = Engine(a, api, broker, store)
    if a.report:
        report(store, broker)
        return
    if a.flatten:
        if not a.live:
            sys.exit("--flatten is for live accounts")
        eng.flatten()
        return
    log(f"kalshi engine: {mode.upper()}  cash {broker.cash():.2f}")
    eng.check_fees()
    if not a.once:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            try:
                eng.cycle()
            except Exception as e:
                log(f"cycle error: {type(e).__name__}: {e}")
            if a.once:
                break
            time.sleep(a.poll)
    finally:
        if not a.once:
            log("shutdown: cancelling resting orders")
            eng.cancel_all()


if __name__ == "__main__":
    main()
