#!/usr/bin/env python3
"""
polymarket_engine.py  --  autonomous execution engine: "be the house for longshot buyers"

Edge being harvested (documented, not assumed):
  - arXiv 2609.12878 (588M Polymarket trades, 2022-2026): buyers of <10c
    contracts who ACCEPT posted offers lose ~31c per $1 (equal-weighted);
    longshots lose -16% in politics, -15% crypto, -25% weather, but are ~fair
    in sports. Losses are largest around 20-25c. Parent events with MANY child
    markets have worse longshot returns.
  - Our own longshot_hv_samples.csv: 2-10c at 7-21d realized below price;
    buying NO there returned +1.6% to +5.3% per holding period (small n).
  => The person on the other side of a lottery ticket -- the resting SELLER of
     cheap YES, i.e. the resting BUYER of expensive NO -- is paid by the bias.

What the engine does, every --poll seconds:
  1. SCAN     open non-sports markets in bias-heavy categories, YES priced
              [--min-yes, --max-yes), ending in [--min-days, --max-days],
              volume >= --min-volume
  2. VALUE    crypto price markets: fair YES from the Deribit options surface
              (polymarket_crypto_longshots), at IV and IV x 1.1, whichever is
              higher; other crypto markets skipped. Everything else:
              fair YES = k_category * YES mid (k from the paper's category
              returns, haircut by --k-haircut); fair NO = 1 - fair YES
  3. QUOTE    post a MAKER bid on the NO token: join/improve the best NO bid,
              never above fair NO - --min-edge, never crossing the ask
              (never pays the spread; lottery buyers come to us)
     ranked by edge per year of capital locked (short-dated first)
  4. SIZE     quarter-Kelly on the NO bet, capped per market / per event /
              per category / gross deployed, from LIVE wallet cash
  5. MANAGE   cancel+requote stale orders; crypto: sell once the bid reaches
              Deribit fair value (edge captured, free the capital); exit if the thesis breaks
              (YES mid >= max(--stop-mult x entry, --stop-yes)); optionally
              sell NO >= --recycle-at early to recycle capital
  6. SETTLE   resolved markets -> realize PnL (paper) / flag for redeem (live)
  7. GUARD    daily loss > --daily-stop or drawdown > --max-dd: cancel all,
              stop opening; state persists in engine.db across restarts

Paper mode is the default and simulates fills conservatively: a resting bid
fills only when the NO ask later trades through it. Live mode needs --live
plus POLY_FUNDER and POLY_PRIVATE_KEY_FILE (path to a chmod-600 file
outside the repo holding the key; POLY_PRIVATE_KEY also works) and `pip install py-clob-client-v2`.

Setup:  pip install curl_cffi            (live: also py-clob-client-v2)
Run:    python polymarket_engine.py                    # paper, loops forever
        python polymarket_engine.py --once             # one cycle, print book
        python polymarket_engine.py --report           # PnL / positions from engine.db
        python polymarket_engine.py --live --close <market_id|all>   # manual exit at the bid
        python polymarket_engine.py --live             # real orders (read RISK below)

RISK, stated once: this sells lottery tickets. It wins small and often and
loses ~1/price times the stake when a longshot hits -- and longshots in the
same event/region hit TOGETHER, which is why caps are per event and per
category. The edge is ~1-3c per contract before adverse selection; it is a
grind, not a jackpot. Run paper for 2+ weeks and compare engine.db against
these numbers before --live.
"""

import argparse
import json
import math
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta

import polymarket_crypto_longshots as CL

try:
    from curl_cffi import requests
except ImportError:
    sys.exit("curl_cffi not installed. Run:  pip install curl_cffi")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
SESSION = requests.Session(impersonate="chrome")

# tag slug -> (category, k = realized/price for longshots). k from arXiv
# 2609.12878 Table 8 (equal child-market weighting, purchases < 10c):
# politics -16.3%, crypto -14.8%, weather -25.2%, finance -2.6%. Sports is
# excluded (no bias). Unlisted tags fall back to the pooled -6.3%.
CATEGORIES = {
    "politics": ("politics", 0.84), "elections": ("politics", 0.84),
    "geopolitics": ("politics", 0.84), "world": ("politics", 0.84),
    "crypto": ("crypto", 0.85), "weather": ("weather", 0.75),
    "economy": ("finance", 0.97), "finance": ("finance", 0.97),
    "tech": ("other", 0.94), "pop-culture": ("other", 0.94), "business": ("other", 0.94),
}


# ----------------------------------------------------------------- helpers
def now_utc():
    return datetime.now(timezone.utc)


def load_env(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_json(url, params=None, retries=3, backoff=1.0):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=25)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(backoff * (attempt + 1) * 2)
                continue
            if r.status_code >= 400:
                return None
            return r.json()
        except Exception:
            time.sleep(backoff * (attempt + 1))
    return None


def parse_iso(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00").replace(" ", "T"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_list(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v) if isinstance(v, str) else None
    except json.JSONDecodeError:
        return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def round_down(x, tick):
    return math.floor(x / tick + 1e-9) * tick


def log(msg):
    print(f"[{now_utc():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------- market data
def resolve_tags(slugs):
    out = {}
    for s in slugs:
        t = get_json(f"{GAMMA}/tags/slug/{s}")
        if isinstance(t, dict) and "id" in t:
            out[s] = t["id"]
        time.sleep(0.1)
    return out


def scan_open(tag_id, max_days, page_size=100, offset_cap=1000, max_total=30000):
    """All open markets under a tag ending within max_days. Gamma stops paging at an
    offset cap, so walk forward by restarting from the latest endDate seen."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    end_min = now_utc().strftime(fmt)
    end_max = (now_utc() + timedelta(days=max_days)).strftime(fmt)
    out, seen = [], set()
    while len(out) < max_total:
        page, offset = [], 0
        while offset < offset_cap:
            page = get_json(f"{GAMMA}/markets", params={
                "closed": "false", "active": "true", "limit": page_size, "offset": offset,
                "tag_id": tag_id, "end_date_min": end_min, "end_date_max": end_max,
                "order": "endDate", "ascending": "true"})
            if not isinstance(page, list) or not page:
                return out
            for m in page:
                if m.get("id") not in seen:
                    seen.add(m.get("id"))
                    out.append(m)
            if len(page) < page_size:
                return out
            offset += page_size
            time.sleep(0.1)
        latest = max((e for e in (parse_iso(m.get("endDate")) for m in page) if e), default=None)
        if latest is None:
            return out
        nxt = latest.strftime(fmt)
        if nxt <= end_min:  # more than offset_cap markets share one endDate: step past it
            nxt = (latest + timedelta(seconds=1)).strftime(fmt)
        end_min = nxt
    return out


def tokens_of(m):
    """(yes_token, no_token, yes_index) for a binary Yes/No market, else None."""
    outs, toks = parse_list(m.get("outcomes")), parse_list(m.get("clobTokenIds"))
    if not (outs and toks and len(outs) == len(toks) == 2):
        return None
    labels = [str(o).strip().lower() for o in outs]
    if labels != ["yes", "no"] and labels != ["no", "yes"]:
        return None
    y = labels.index("yes")
    return toks[y], toks[1 - y], y


def fetch_book(token):
    d = get_json(f"{CLOB}/book", params={"token_id": token})
    if not isinstance(d, dict):
        return None
    bids = sorted([(to_float(b.get("price")), to_float(b.get("size"))) for b in d.get("bids", [])],
                  key=lambda x: -(x[0] or 0))
    asks = sorted([(to_float(a.get("price")), to_float(a.get("size"))) for a in d.get("asks", [])],
                  key=lambda x: x[0] if x[0] is not None else 9)
    bids = [(p, s) for p, s in bids if p and s]
    asks = [(p, s) for p, s in asks if p and s]
    return {"bids": bids, "asks": asks,
            "bid": bids[0][0] if bids else None, "ask": asks[0][0] if asks else None}


def market_status(market_id):
    """None if still open; else final YES payout (1.0 / 0.0 / fractional)."""
    m = get_json(f"{GAMMA}/markets/{market_id}")
    if not isinstance(m, dict) or not m.get("closed"):
        return None
    outs, fin = parse_list(m.get("outcomes")), parse_list(m.get("outcomePrices"))
    if not (outs and fin):
        return None
    labels = [str(o).strip().lower() for o in outs]
    p = to_float(fin[labels.index("yes")] if "yes" in labels else fin[0])
    if p is None or 0.02 < p < 0.98:   # closed but not yet settled by UMA
        return None
    return round(p)


# ----------------------------------------------------------------- state
SCHEMA = """
create table if not exists orders(id text primary key, token text, market_id text, event_id text,
  category text, question text, side text, price real, size real, filled real default 0,
  status text, created text, yes_mid real, end_date text);
create table if not exists positions(token text primary key, market_id text, event_id text,
  category text, question text, shares real, cost real, entry_yes real, end_date text,
  opened text, status text default 'open', exit_value real, closed text);
create table if not exists equity(ts text, cash real, position_value real, total real);
create table if not exists kv(k text primary key, v text);
"""


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def q(self, sql, *a):
        return self.db.execute(sql, a).fetchall()

    def x(self, sql, *a):
        self.db.execute(sql, a)
        self.db.commit()

    def get(self, k, default=None):
        r = self.q("select v from kv where k=?", k)
        return json.loads(r[0]["v"]) if r else default

    def put(self, k, v):
        self.x("insert or replace into kv values(?,?)", k, json.dumps(v))

    def open_orders(self):
        return self.q("select * from orders where status='open'")

    def open_positions(self):
        return self.q("select * from positions where status='open'")

    def add_fill(self, o, shares, price):
        self.x("update orders set filled=filled+?, status=case when filled+?>=size-1e-6 then 'filled' "
               "else status end where id=?", shares, shares, o["id"])
        p = self.q("select * from positions where token=?", o["token"])
        if p and p[0]["status"] == "open":
            self.x("update positions set shares=shares+?, cost=cost+? where token=?",
                   shares, shares * price, o["token"])
        else:
            self.x("insert or replace into positions(token,market_id,event_id,category,question,shares,"
                   "cost,entry_yes,end_date,opened,status) values(?,?,?,?,?,?,?,?,?,?,'open')",
                   o["token"], o["market_id"], o["event_id"], o["category"], o["question"],
                   shares, shares * price, o["yes_mid"], o["end_date"], now_utc().isoformat())


# ----------------------------------------------------------------- brokers
class PaperBroker:
    """Simulated wallet. Maker bids fill only when the NO ask trades through
    them on a later poll -- conservative, so paper fill rates UNDERSTATE live."""

    def __init__(self, store, bankroll):
        self.s = store
        if self.s.get("paper_cash") is None:
            self.s.put("paper_cash", bankroll)

    def balance(self):
        return self.s.get("paper_cash")

    def place_bid(self, token, price, size):
        oid = f"paper-{token[-8:]}-{int(time.time() * 1000)}"
        return oid

    def cancel(self, oid):
        return True

    def sync(self, order, book):
        """Return shares newly filled for this order."""
        if book and book["ask"] is not None and book["ask"] <= order["price"]:
            avail = sum(s for p, s in book["asks"] if p <= order["price"])
            fill = min(order["size"] - order["filled"], avail)
            self.s.put("paper_cash", self.s.get("paper_cash") - fill * order["price"])
            return fill
        return 0.0

    def sell(self, token, shares, price):
        """Returns shares actually sold."""
        self.s.put("paper_cash", self.s.get("paper_cash") + shares * price)
        return shares

    def settle(self, pos, payout_no):
        self.s.put("paper_cash", self.s.get("paper_cash") + pos["shares"] * payout_no)

    def exchange_positions(self):
        return None  # paper state lives only in the db


class LiveBroker:
    """Real orders through Polymarket CLOB V2 (py-clob-client-v2; V1 stopped
    working at the April 2026 cutover). Collateral is pUSD.
    POLY_SIGNATURE_TYPE: 0 EOA, 1 email/Magic proxy, 2 browser-wallet Safe,
    3 deposit wallet (newer accounts). `--diagnose` shows which one holds funds."""

    def __init__(self, store, sig_type=None):
        try:
            from py_clob_client_v2 import (ClobClient, OrderArgs, OrderType, OrderPayload,
                                           BalanceAllowanceParams, AssetType, Side)
        except ImportError:
            sys.exit("Live mode needs:  pip install py-clob-client-v2")
        key, funder = os.environ.get("POLY_PRIVATE_KEY"), os.environ.get("POLY_FUNDER")
        key_file = os.environ.get("POLY_PRIVATE_KEY_FILE")
        if not key and key_file:
            if os.name == "posix" and os.stat(key_file).st_mode & 0o077:
                sys.exit(f"{key_file} is readable by other users; run: chmod 600 {key_file}")
            with open(key_file, encoding="utf-8") as f:
                key = f.read().strip()
        if not key:
            sys.exit("Set POLY_PRIVATE_KEY_FILE (preferred) or POLY_PRIVATE_KEY")
        if sig_type is None:
            sig_type = int(os.environ.get("POLY_SIGNATURE_TYPE", "2"))
        self.OrderArgs, self.OrderType, self.OrderPayload, self.Side = OrderArgs, OrderType, OrderPayload, Side
        self.BalParams, self.AssetType = BalanceAllowanceParams, AssetType
        self.c = ClobClient(CLOB, chain_id=137, key=key, signature_type=sig_type,
                            funder=funder if sig_type else None)
        self.c.set_api_creds(self.c.create_or_derive_api_key())
        self.s = store

    def balance(self):
        params = self.BalParams(asset_type=self.AssetType.COLLATERAL)
        try:  # the server caches balances; without a refresh it can report 0
            self.c.update_balance_allowance(params)
        except Exception as e:
            log(f"balance refresh failed: {e}")
        b = self.c.get_balance_allowance(params)
        return float(b["balance"]) / 1e6

    def place_bid(self, token, price, size):
        # post_only: rest as a maker or be rejected -- never pay the spread
        r = self.c.create_and_post_order(
            order_args=self.OrderArgs(token_id=token, price=price, size=size, side=self.Side.BUY),
            order_type=self.OrderType.GTC, post_only=True)
        if not (r and r.get("success")):
            log(f"order rejected: {r}")
            return None
        return r.get("orderID")

    def cancel(self, oid):
        self.c.cancel_order(self.OrderPayload(orderID=oid))
        return True

    def sync(self, order, book):
        o = self.c.get_order(order["id"]) or {}
        matched = float(o.get("size_matched", 0) or 0)
        return max(0.0, matched - order["filled"])

    def exchange_positions(self):
        """{token: position} as Polymarket sees the funder wallet, or None if unavailable."""
        funder = os.environ.get("POLY_FUNDER")
        rows = get_json(f"{DATA}/positions", params={"user": funder, "limit": 500}) if funder else None
        if not isinstance(rows, list):
            return None
        return {str(r["asset"]): r for r in rows if r.get("asset")}

    def shares(self, token):
        params = self.BalParams(asset_type=self.AssetType.CONDITIONAL, token_id=token)
        try:
            self.c.update_balance_allowance(params)
        except Exception:
            pass
        return float(self.c.get_balance_allowance(params)["balance"]) / 1e6

    def sell(self, token, shares, price):
        """FAK sell at >= price. Returns shares actually sold (partial fills happen)."""
        before = self.shares(token)
        r = self.c.create_and_post_order(
            order_args=self.OrderArgs(token_id=token, price=price, size=shares, side=self.Side.SELL),
            order_type=self.OrderType.FAK)
        if not (r and r.get("success")):
            log(f"sell rejected: {r}")
            return 0.0
        sold = to_float(r.get("makingAmount"))
        if not sold:  # response without fill amounts: measure the position instead
            time.sleep(3)
            sold = before - self.shares(token)
        return max(0.0, min(shares, sold))

    def settle(self, pos, payout_no):
        # Winning shares must be redeemed on-chain (CTF redeemPositions) or via the
        # Polymarket UI; the engine records the value and flags it.
        log(f"REDEEM NEEDED: {pos['question'][:70]} ({pos['shares']:.1f} NO shares)")


# ----------------------------------------------------------------- engine
class Engine:
    def __init__(self, args, store, broker):
        self.a, self.s, self.b = args, store, broker
        self.tags = {}

    # ---- valuation / sizing
    def fair_no(self, yes_mid, k):
        return 1.0 - min(1.0, yes_mid * k * (1 + self.a.k_haircut))

    # ---- crypto fair value from Deribit options
    def refresh_surfaces(self):
        if "crypto" not in self.a.tags:
            return
        self.surfaces = {}
        for asset in CL.ASSETS:
            try:
                s = CL.Surface(asset)
            except Exception as e:
                log(f"deribit {asset}: {type(e).__name__}: {e}")
                continue
            if s.ok:
                self.surfaces[asset] = s
        if not self.surfaces:
            log("deribit: no surfaces -- crypto markets skipped this cycle")

    def crypto_parse(self, question):
        p = CL.parse_question(question)
        return p if p and p["asset"] in getattr(self, "surfaces", {}) else None

    def deribit_mid_no(self, parsed, days):
        """Unbiased P(NO) at market IV: for VALUING positions, not for deciding trades."""
        surf = self.surfaces.get(parsed["asset"])
        if surf is None or days <= 0:
            return None
        return 1.0 - CL.model_prob(surf, parsed, days / 365, 1.0)

    def deribit_fair_no(self, parsed, days):
        """Conservative P(NO) for SELLING the YES: options already overprice tails,
        and we bump vol further, taking whichever makes YES more likely."""
        surf = self.surfaces.get(parsed["asset"])
        if surf is None or days <= 0:
            return None
        t = days / 365
        p_yes = max(CL.model_prob(surf, parsed, t, 1.0),
                    CL.model_prob(surf, parsed, t, 1.0 + self.a.deribit_vol_bump))
        return 1.0 - p_yes

    def kelly_frac(self, q_no, price):
        if price >= 1 or q_no <= price:
            return 0.0
        return self.a.kelly * (q_no - price) / (1 - price)

    def reserved(self):
        return sum(o["price"] * (o["size"] - o["filled"]) for o in self.s.open_orders())

    def available(self):
        # conservative in live mode if the exchange balance already nets out resting orders
        return self.b.balance() - self.reserved()

    def equity(self):
        pv = 0.0
        for p in self.s.open_positions():
            pv += p["shares"] * (self.s.get(f"mark:{p['token']}", p["cost"] / max(p["shares"], 1e-9)))
        return self.b.balance(), pv

    def exposure(self):
        by_mkt, by_evt, by_cat, gross = {}, {}, {}, 0.0
        rows = [(p["market_id"], p["event_id"], p["category"], p["cost"]) for p in self.s.open_positions()]
        rows += [(o["market_id"], o["event_id"], o["category"], o["price"] * (o["size"] - o["filled"]))
                 for o in self.s.open_orders()]
        for m, e, c, v in rows:
            by_mkt[m] = by_mkt.get(m, 0) + v
            by_evt[e] = by_evt.get(e, 0) + v
            by_cat[c] = by_cat.get(c, 0) + v
            gross += v
        return by_mkt, by_evt, by_cat, gross

    # ---- risk guard
    def guard(self):
        cash, pv = self.equity()
        total = cash + pv
        self.s.x("insert into equity values(?,?,?,?)", now_utc().isoformat(), cash, pv, total)
        peak = max(self.s.get("peak") or total, total)
        self.s.put("peak", peak)
        day = now_utc().strftime("%Y-%m-%d")
        if self.s.get("day") != day:
            self.s.put("day", day)
            self.s.put("day_start", total)
        day_start = self.s.get("day_start", total)
        halted = self.s.get("halted")
        breach = None
        if total < day_start * (1 - self.a.daily_stop):
            breach = f"daily loss {total / day_start - 1:+.1%} on {day}"
        elif total < peak * (1 - self.a.max_dd):
            breach = f"drawdown {total / peak - 1:+.1%} from peak {peak:.2f}"
        # A fill debits cash before the positions feed shows the shares, so one bad
        # reading is usually accounting lag. Halt only if it persists a second cycle.
        strikes = self.s.get("breach_strikes", 0) + 1 if breach else 0
        self.s.put("breach_strikes", strikes)
        if breach and not halted:
            if strikes >= 2:
                halted = breach
            else:
                log(f"loss check tripped once ({breach}); rechecking next cycle before halting")
        if halted and not self.s.get("halted"):
            log(f"KILL SWITCH: {halted} -- cancelling all orders, no new entries")
            self.s.put("halted", halted)
            self.cancel_all()
        return total, bool(halted)

    def reconcile(self):
        """Make the db match the exchange: adopt positions filled while the engine was
        stopped (or fills it missed) and correct drifted share counts."""
        held = self.b.exchange_positions()
        if held is None:
            return
        db = {p["token"]: p for p in self.s.open_positions()}
        for tok, x in held.items():
            size = to_float(x.get("size")) or 0.0
            avg = to_float(x.get("avgPrice")) or 0.0
            if size < 0.01 or x.get("redeemable"):
                continue
            p = db.get(tok)
            if p:
                if abs(p["shares"] - size) >= 0.01:
                    log(f"RECONCILE shares {p['shares']:.2f} -> {size:.2f}  {p['question'][:60]}")
                    self.s.x("update positions set shares=?, cost=? where token=?", size, size * avg, tok)
                continue
            o = self.s.q("select * from orders where token=? order by created desc limit 1", tok)
            if o:
                o = o[0]
                meta = (o["market_id"], o["event_id"], o["category"], o["question"], o["yes_mid"], o["end_date"])
                self.s.x("update orders set status='filled', filled=size where id=?", o["id"])
            elif str(x.get("outcome", "")).lower() == "no":
                m = get_json(f"{GAMMA}/markets", params={"clob_token_ids": tok})
                m = m[0] if isinstance(m, list) and m else {}
                meta = (str(m.get("id", x.get("conditionId"))), str(x.get("eventId") or m.get("id")), "manual",
                        x.get("title") or m.get("question") or "", 1 - avg, x.get("endDate") or m.get("endDate") or "")
            else:
                continue  # a YES bought by hand: not the engine's to manage
            self.s.x("insert or replace into positions(token,market_id,event_id,category,question,shares,"
                     "cost,entry_yes,end_date,opened,status) values(?,?,?,?,?,?,?,?,?,?,'open')",
                     tok, *meta[:4], size, size * avg, meta[4], meta[5], now_utc().isoformat())
            log(f"ADOPT {size:.1f} NO @ {avg:.3f} found on exchange  {meta[3][:60]}")

    def cancel_all(self):
        for o in self.s.open_orders():
            self.b.cancel(o["id"])
            self.s.x("update orders set status='cancelled' where id=?", o["id"])

    # ---- cycle steps
    def sync_orders(self):
        for o in self.s.open_orders():
            book = fetch_book(o["token"])
            fill = self.b.sync(o, book)
            if fill > 0:
                self.s.add_fill(o, fill, o["price"])
                log(f"FILL  {fill:.1f} NO @ {o['price']:.3f}  {o['question'][:70]}")
            if self.s.q("select status from orders where id=?", o["id"])[0][0] != "open":
                continue
            age_min = (now_utc() - parse_iso(o["created"])).total_seconds() / 60
            stale = age_min >= self.a.reprice_min
            # crypto fair value moves with spot: pull a resting bid the moment it's no longer cheap
            parsed = self.crypto_parse(o["question"]) if o["category"] == "crypto" else None
            if parsed and not stale:
                days = (parse_iso(o["end_date"]) - now_utc()).total_seconds() / 86400
                fair = self.deribit_fair_no(parsed, days)
                if fair is None or o["price"] > fair - self.a.min_edge:
                    stale = True
                    log(f"PULL  NO @ {o['price']:.3f} now above fair {fair if fair is not None else float('nan'):.3f} - edge"
                        f"  {o['question'][:60]}")
            if stale:
                self.b.cancel(o["id"])
                self.s.x("update orders set status='cancelled' where id=?", o["id"])

    def settle_and_exit(self):
        for p in self.s.open_positions():
            res = market_status(p["market_id"])
            if res is not None:
                payout_no = 1.0 - res
                self.b.settle(p, payout_no)
                self.s.x("update positions set status='closed', exit_value=?, closed=? where token=?",
                         p["shares"] * payout_no, now_utc().isoformat(), p["token"])
                log(f"SETTLED {'WIN ' if payout_no else 'LOSS'} pnl={p['shares'] * payout_no - p['cost']:+.2f}"
                    f"  {p['question'][:70]}")
                continue
            book = fetch_book(p["token"])
            # Value and stop on something a thin book can't fake: quiet hours often leave
            # a one-sided book with a stray 1c bid, which once valued a near-certain
            # winner at ~0 and tripped the kill switch.
            two_sided = bool(book and book["bid"] is not None and book["ask"] is not None
                             and book["ask"] - book["bid"] <= self.a.max_mark_spread)
            parsed = self.crypto_parse(p["question"]) if p["category"] == "crypto" else None
            days = ((parse_iso(p["end_date"]) or now_utc()) - now_utc()).total_seconds() / 86400
            fair = self.deribit_fair_no(parsed, days) if parsed else None
            if fair is not None:
                no_value = fair  # conservative: drives the stop
                self.s.put(f"mark:{p['token']}", self.deribit_mid_no(parsed, days))
            elif two_sided:
                no_value = (book["bid"] + book["ask"]) / 2
                self.s.put(f"mark:{p['token']}", no_value)
            else:
                continue  # no trustworthy price this cycle: keep the last mark, don't trade
            yes_now = 1.0 - no_value
            stop_at = max(self.a.stop_mult * (p["entry_yes"] or 0), self.a.stop_yes)
            reason = None
            if yes_now >= stop_at:
                reason = f"STOP yes {yes_now:.3f} >= {stop_at:.3f}"
            elif self.a.recycle_at and two_sided and book["bid"] >= self.a.recycle_at:
                reason = f"RECYCLE bid {book['bid']:.3f}"
            elif parsed and two_sided and self.a.edge_exit is not None:
                # the edge was the gap to fair value: once the market pays fair, holding to
                # resolution only adds risk and locks capital -- take it and redeploy
                fair_mid = self.deribit_mid_no(parsed, days)
                if fair_mid is not None and book["bid"] >= fair_mid - self.a.edge_exit:
                    reason = f"EDGE-DONE bid {book['bid']:.3f} >= fair {fair_mid:.3f}"
            if reason:
                if not two_sided:
                    log(f"{reason} but the book is too thin to sell into; holding  {p['question'][:60]}")
                    continue
                px = max(round(book["bid"] - 0.02, 3), 0.01) if reason.startswith("STOP") else book["bid"]
                shares = min(p["shares"], sum(s for pr, s in book["bids"] if pr >= px))
                if shares > 0:
                    self.record_sale(p, self.b.sell(p["token"], shares, px), px, reason)

    def record_sale(self, p, sold, px, reason):
        if sold <= 0:
            log(f"{reason}: nothing sold  {p['question'][:60]}")
            return
        left = p["shares"] - sold
        cost_sold = p["cost"] * sold / p["shares"]
        if left <= 0.01:
            self.s.x("update positions set status='closed', exit_value=?, closed=? where token=?",
                     sold * px, now_utc().isoformat(), p["token"])
        else:
            self.s.x("update positions set shares=?, cost=? where token=?", left, p["cost"] - cost_sold, p["token"])
            self.s.put("partial_pnl", self.s.get("partial_pnl", 0.0) + sold * px - cost_sold)
        log(f"{reason}: sold {sold:.1f} @ {px:.3f}  pnl={sold * px - cost_sold:+.2f}"
            f"{f'  ({left:.1f} left)' if left > 0.01 else ''}  {p['question'][:60]}")

    def close(self, which, min_price=None):
        """Manual exit: cancel orders and sell positions for a market id (or 'all') at the bid."""
        for o in self.s.open_orders():
            if which in ("all", o["market_id"]):
                self.b.cancel(o["id"])
                self.s.x("update orders set status='cancelled' where id=?", o["id"])
                log(f"cancelled order {o['id'][:18]}  {o['question'][:60]}")
        targets = [p for p in self.s.open_positions() if which in ("all", p["market_id"])]
        if not targets:
            log(f"no open position for '{which}' -- see --report for market ids")
        for p in targets:
            book = fetch_book(p["token"])
            if not book or book["bid"] is None:
                log(f"no bids right now, not sold  {p['question'][:60]}")
                continue
            px = book["bid"] if min_price is None else max(book["bid"], min_price)
            depth = sum(s for pr, s in book["bids"] if pr >= px)
            if depth <= 0:
                log(f"no bids at >= {px:.3f}, not sold (best bid {book['bid']:.3f})  {p['question'][:60]}")
                continue
            self.record_sale(p, self.b.sell(p["token"], min(p["shares"], depth), px), px, "CLOSE")

    def candidates(self):
        if not self.tags:
            self.tags = resolve_tags(self.a.tags)
            log(f"tags resolved: {', '.join(self.tags)}")
        seen, out = set(), []
        for slug, tid in self.tags.items():
            cat, k = CATEGORIES.get(slug, ("other", 0.94))
            markets = scan_open(tid, self.a.max_days)
            log(f"scanned {slug}: {len(markets)} open markets")
            for m in markets:
                if m.get("id") in seen or not m.get("enableOrderBook", True):
                    continue
                seen.add(m.get("id"))
                toks = tokens_of(m)
                end = parse_iso(m.get("endDate"))
                vol = to_float(m.get("volumeNum") or m.get("volume")) or 0
                prices = parse_list(m.get("outcomePrices"))
                if not (toks and end and prices) or vol < self.a.min_volume:
                    continue
                days = (end - now_utc()).total_seconds() / 86400
                if not (self.a.min_days <= days <= self.a.max_days):
                    continue
                last_yes = to_float(prices[toks[2]])
                if last_yes is None or not (self.a.min_yes <= last_yes < self.a.max_yes):
                    continue
                deribit = None
                if cat == "crypto":
                    deribit = self.crypto_parse(m.get("question") or "")
                    if deribit is None and not self.a.crypto_any:
                        continue  # memecoins, token launches: no fair value, don't guess
                ev = (m.get("events") or [{}])[0]
                out.append({"market_id": str(m["id"]), "event_id": str(ev.get("id", m["id"])),
                            "n_children": len(ev.get("markets") or []) or 1, "category": cat, "k": k,
                            "question": m.get("question") or "", "yes_tok": toks[0], "no_tok": toks[1],
                            "end": end.isoformat(), "days": days, "vol": vol, "deribit": deribit,
                            "tick": to_float(m.get("orderPriceMinTickSize")) or 0.01,
                            "min_size": to_float(m.get("orderMinSize")) or 5})
        # many-child events first (paper: their longshots are the most overpriced)
        out.sort(key=lambda c: (-c["n_children"], -c["vol"]))
        return out

    def quote(self, c):
        """Maker NO bid for a candidate from live books, or None if no edge."""
        yb, nb = fetch_book(c["yes_tok"]), fetch_book(c["no_tok"])
        if not (yb and nb and yb["bid"] is not None and yb["ask"] is not None and nb["ask"] is not None):
            return None
        yes_mid = (yb["bid"] + yb["ask"]) / 2
        if not (self.a.min_yes <= yes_mid < self.a.max_yes):
            return None
        if c.get("deribit"):
            fair = self.deribit_fair_no(c["deribit"], c["days"])
            if fair is None:
                return None
        else:
            fair = self.fair_no(yes_mid, c["k"])
        tick = c["tick"]
        best_bid = nb["bid"] or 0.0
        price = min(round_down(fair - self.a.min_edge, tick), round(nb["ask"] - tick, 6),
                    round(best_bid + tick, 6))
        if price <= 0 or price < best_bid or fair - price < self.a.min_edge:
            return None
        # capital is locked until resolution: rank by expected return per year locked
        ann = (fair - price) / price * 365 / max(c["days"], 0.5)
        return {**c, "yes_mid": yes_mid, "fair": fair, "price": price, "ann": ann,
                "src": "deribit" if c.get("deribit") else "bias"}

    def enter(self, equity):
        if self.available() - self.a.cash_buffer * equity < 5 * (1 - self.a.max_yes):
            return  # not enough free cash for even a minimum order; skip the book scan
        by_mkt, by_evt, by_cat, gross = self.exposure()
        held = {p["market_id"] for p in self.s.open_positions()} | {o["market_id"] for o in self.s.open_orders()}
        pool = sorted((c for c in self.candidates() if c["market_id"] not in held), key=lambda c: c["days"])
        quotes = [q for q in map(self.quote, pool[:self.a.max_quotes]) if q]
        quotes.sort(key=lambda q: -q["ann"])
        placed = 0
        for q in quotes:
            if placed >= self.a.max_new:
                break
            price = q["price"]
            f = self.kelly_frac(q["fair"], price)
            budget = min(f * equity,
                         self.a.cap_market * equity - by_mkt.get(q["market_id"], 0),
                         self.a.cap_event * equity - by_evt.get(q["event_id"], 0),
                         self.a.cap_category * equity - by_cat.get(q["category"], 0),
                         self.a.cap_gross * equity - gross,
                         self.available() - self.a.cash_buffer * equity)
            size = math.floor(budget / price)
            if size < q["min_size"]:
                continue
            oid = self.b.place_bid(q["no_tok"], price, size)
            if not oid:
                continue
            self.s.x("insert into orders values(?,?,?,?,?,?,?,?,?,0,'open',?,?,?)", oid, q["no_tok"],
                     q["market_id"], q["event_id"], q["category"], q["question"], "BUY_NO", price, size,
                     now_utc().isoformat(), q["yes_mid"], q["end"])
            cost = price * size
            by_mkt[q["market_id"]] = by_mkt.get(q["market_id"], 0) + cost
            by_evt[q["event_id"]] = by_evt.get(q["event_id"], 0) + cost
            by_cat[q["category"]] = by_cat.get(q["category"], 0) + cost
            gross += cost
            placed += 1
            log(f"BID   {size:>5} NO @ {price:.3f} (fair {q['fair']:.3f}, yes {q['yes_mid']:.3f}, "
                f"{q['days']:.1f}d, {q['ann']:+.0%}/yr, {q['category']}/{q['src']})  {q['question'][:60]}")

    def cycle(self):
        self.refresh_surfaces()
        self.sync_orders()
        self.reconcile()
        self.settle_and_exit()
        total, halted = self.guard()
        if not halted:
            self.enter(total)
        cash, pv = self.equity()
        log(f"equity {cash + pv:.2f}  (cash {cash:.2f}, positions {pv:.2f}, "
            f"{len(self.s.open_positions())} pos, {len(self.s.open_orders())} orders)"
            + (f"  HALTED: {self.s.get('halted')}" if halted else ""))


def report(store):
    closed = store.q("select * from positions where status='closed'")
    opn = store.open_positions()
    pnl = sum(p["exit_value"] - p["cost"] for p in closed) + store.get("partial_pnl", 0.0)
    wins = sum(1 for p in closed if p["exit_value"] > p["cost"])
    print(f"closed {len(closed)}  wins {wins}  realized pnl {pnl:+.2f}  "
          f"return on cost {pnl / max(sum(p['cost'] for p in closed), 1e-9):+.2%}")
    by = {}
    for p in closed:
        b = by.setdefault(p["category"], [0, 0.0, 0.0])
        b[0] += 1
        b[1] += p["cost"]
        b[2] += p["exit_value"] - p["cost"]
    for c, (n, cost, pl) in sorted(by.items()):
        print(f"  {c:<10} n={n:<4} cost={cost:>9.2f}  pnl={pl:+9.2f}  ({pl / cost:+.2%})")
    print(f"\nopen positions: {len(opn)}")
    for p in opn:
        print(f"  market {p['market_id']:<9} {p['shares']:>7.1f} NO  cost {p['cost']:>8.2f}  entry_yes {p['entry_yes']:.3f}  "
              f"ends {p['end_date'][:10]}  {p['question'][:60]}")
    eq = store.q("select * from equity order by ts desc limit 1")
    if eq:
        print(f"\nlast equity {eq[0]['total']:.2f} at {eq[0]['ts'][:16]}  peak {store.get('peak')}"
              + (f"  HALTED: {store.get('halted')}" if store.get("halted") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="send real orders (default: paper)")
    ap.add_argument("--db", default=None, help="state file (default engine_paper.db / engine_live.db)")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--bankroll", type=float, default=1000.0, help="paper starting cash")
    ap.add_argument("--poll", type=int, default=300, help="seconds between cycles")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--resume", action="store_true", help="clear a kill-switch halt and continue")
    ap.add_argument("--close", metavar="MARKET_ID", help="sell this position (or 'all') at the bid and exit")
    ap.add_argument("--close-min-price", type=float, help="with --close: don't sell below this price")
    ap.add_argument("--diagnose", action="store_true", help="live: show signer and pUSD balance per signature type")
    ap.add_argument("--tags", nargs="+", default=list(CATEGORIES))
    # universe
    ap.add_argument("--min-yes", type=float, default=0.03)
    ap.add_argument("--max-yes", type=float, default=0.25)
    ap.add_argument("--min-days", type=float, default=1.0)
    ap.add_argument("--max-days", type=float, default=45.0)
    ap.add_argument("--min-volume", type=float, default=25000)
    # pricing
    ap.add_argument("--k-haircut", type=float, default=0.10, help="inflate fair YES by this for model error")
    ap.add_argument("--min-edge", type=float, default=0.01, help="min fair NO minus bid, in $")
    ap.add_argument("--deribit-vol-bump", type=float, default=0.10,
                    help="crypto: also price at IV x (1+bump) and use the higher YES probability")
    ap.add_argument("--crypto-any", action="store_true",
                    help="also trade crypto markets Deribit can't price (memecoins etc.) on the bias model")
    ap.add_argument("--reprice-min", type=float, default=30, help="cancel/requote orders older than this")
    # sizing / risk
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--cap-market", type=float, default=0.03)
    ap.add_argument("--cap-event", type=float, default=0.06)
    ap.add_argument("--cap-category", type=float, default=0.30)
    ap.add_argument("--cap-gross", type=float, default=0.75)
    ap.add_argument("--cash-buffer", type=float, default=0.10)
    ap.add_argument("--max-new", type=int, default=10, help="max new orders per cycle")
    ap.add_argument("--max-quotes", type=int, default=150,
                    help="soonest-ending candidates to price each cycle (then ranked by edge per year)")
    ap.add_argument("--max-mark-spread", type=float, default=0.10,
                    help="value/stop on the book mid only if its spread is at most this")
    ap.add_argument("--stop-mult", type=float, default=3.0)
    ap.add_argument("--stop-yes", type=float, default=0.35)
    ap.add_argument("--recycle-at", type=float, default=0.99, help="sell NO at/above this bid (0=off)")
    ap.add_argument("--edge-exit", type=float, default=0.0,
                    help="crypto: sell once the NO bid is within this of Deribit fair value (edge captured)")
    ap.add_argument("--hold-to-resolution", dest="edge_exit", action="store_const", const=None,
                    help="disable --edge-exit")
    ap.add_argument("--daily-stop", type=float, default=0.05)
    ap.add_argument("--max-dd", type=float, default=0.15)
    args = ap.parse_args()

    load_env(args.env_file)
    store = Store(args.db or ("engine_live.db" if args.live else "engine_paper.db"))
    if args.resume:
        store.put("halted", None)
        store.put("peak", None)
        store.put("day", None)
        store.put("breach_strikes", 0)
        print("kill switch cleared")
    if args.report:
        report(store)
        return
    if args.diagnose:
        for st in (0, 1, 2, 3):
            try:
                b = LiveBroker(store, sig_type=st)
                print(f"signature_type {st}: signer {b.c.get_address()}  pUSD balance {b.balance():.2f}")
            except Exception as e:
                print(f"signature_type {st}: error {type(e).__name__}: {e}")
        print(f"funder (POLY_FUNDER): {os.environ.get('POLY_FUNDER')}")
        print("Set POLY_SIGNATURE_TYPE in .env to the type that shows your balance.")
        return
    broker = LiveBroker(store) if args.live else PaperBroker(store, args.bankroll)
    eng = Engine(args, store, broker)
    if args.close:
        eng.close(args.close, args.close_min_price)
        report(store)
        return
    log(f"engine start: {'LIVE' if args.live else 'PAPER'}  cash {broker.balance():.2f}")
    if not args.once:  # `systemctl stop` sends SIGTERM: leave no orders resting unattended
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            try:
                eng.cycle()
            except Exception as e:  # keep running through transient API failures
                log(f"cycle error: {type(e).__name__}: {e}")
            if args.once:
                break
            time.sleep(args.poll)
    finally:
        if not args.once and store.open_orders():
            log(f"shutdown: cancelling {len(store.open_orders())} open order(s)")
            eng.cancel_all()


if __name__ == "__main__":
    main()
