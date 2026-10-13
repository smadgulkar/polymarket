#!/usr/bin/env python3
"""
kalshi_backtest.py  --  event-driven backtest of the Kalshi engine's quoting rule on real trades

Validates the strategy kalshi_engine.py runs, on Kalshi's own history, with the frictions a live
bot actually faces:

  QUOTES     at every public trade, the engine's resting prices are recomputed exactly as the
             engine would: cautious Deribit fair value (IV x {0.85, 1, 1.1}) minus --edge, NO
             (= YES ask) on 3-25c YES, YES bids on price ranges with >= --yes-min-hours left
  STALENESS  fair value uses the spot price --lag-min minutes BEFORE the trade (a live bot
             requotes on a cycle, so its resting price lags the market; trades that hit us
             right after a move are the adverse selection this charges for)
  FILLS      a trade fills us only if the taker crossed our price on our side (taker bought YES
             at >= our ask / sold YES at <= our bid), at OUR price, for --share of the size
             (queue competition with other makers)
  BANKROLL   cash, open exposure, per-market / per-event / per-cluster / gross caps and
             Kelly sizing like the engine; positions held to settlement, paid on the real result
  FEES       from each series' fee_type (quadratic = no maker fee; *_with_maker_fees charged)

Output: equity curve by day, total return, max drawdown, worst day, P&L by series / time left,
and the same flow without bankroll limits for reference. Trades are cached in
kalshi_cache/ so re-running with different settings is instant.

Run:  python kalshi_backtest.py --days 7                   # fetch (once) + simulate
      python kalshi_backtest.py --days 7 --lag-min 5 --share 0.1 --min-hours 1
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from curl_cffi import requests

import polymarket_crypto_longshots as CL
from polymarket_capacity import spot_history, spot_at, fair_band
from polymarket_engine import get_json

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"KXBTCD": "BTC", "KXETHD": "ETH", "KXBTC": "BTC", "KXETH": "ETH"}
CACHE = "kalshi_cache"
MAKER_FEE_RATE = {"quadratic": 0.0, "quadratic_with_maker_fees": 0.0175, "quadratic_with_combo_maker_fees": 0.035}


def session():
    return requests.Session(impersonate="chrome")


def get(sess, path, params, retries=6):
    for attempt in range(retries):
        try:
            r = sess.get(API + path, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            return r.json() if r.status_code == 200 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def paged(sess, path, key, params, max_pages=100):
    out, cursor = [], None
    for _ in range(max_pages):
        d = get(sess, path, dict(params, **({"cursor": cursor} if cursor else {}))) or {}
        out += d.get(key) or []
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parsed_of(m, asset):
    st, lo, hi = m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")
    if st == "greater" and lo is not None:
        return {"asset": asset, "kind": "above", "k1": float(lo), "k2": None}
    if st == "less" and hi is not None:
        return {"asset": asset, "kind": "below", "k1": float(hi), "k2": None}
    if st == "between" and lo is not None and hi is not None:
        return {"asset": asset, "kind": "between", "k1": float(lo), "k2": float(hi)}
    return None


def maker_fee(rate, price, n):
    return math.ceil(rate * n * price * (1 - price) * 100) / 100 if rate else 0.0


# ----------------------------------------------------------------- data (cached)
def load_markets(days, refresh, series=None, end_days_ago=0):
    """Settled markets closing between `days` and `end_days_ago` days ago (end_days_ago > 0 keeps a
    recent stretch out of the sample, for out-of-sample tests)."""
    os.makedirs(CACHE, exist_ok=True)
    series = series or list(SERIES)
    path = os.path.join(CACHE, f"markets_{days}d_{end_days_ago}_{'-'.join(sorted(series))}.json")
    if os.path.exists(path) and not refresh:
        return json.load(open(path))
    now = int(time.time())
    sess, out, fees = session(), [], {}
    for ser, asset in ((s_, SERIES[s_]) for s_ in series):
        s = (get(sess, f"/series/{ser}", {}) or {}).get("series", {})
        fees[ser] = MAKER_FEE_RATE.get(s.get("fee_type"), 0.0) * float(s.get("fee_multiplier") or 1)
        ms = paged(sess, "/markets", "markets", {"series_ticker": ser, "min_close_ts": now - days * 86400,
                                                  "max_close_ts": now - end_days_ago * 86400, "limit": 1000})
        keep = [m for m in ms if m.get("result") in ("yes", "no") and float(m.get("volume_fp") or 0) > 0
                and parsed_of(m, asset)]
        out += [{"ticker": m["ticker"], "event": m["event_ticker"], "series": ser, "asset": asset,
                 "close": ts(m["close_time"]), "result": m["result"], "parsed": parsed_of(m, asset)} for m in keep]
        print(f"  {ser}: {len(keep)} settled, traded markets (maker fee rate {fees[ser]})", flush=True)
    data = {"markets": out, "fees": fees, "fetched": now}
    json.dump(data, open(path, "w"))
    return data


def load_trades(markets, workers):
    """{ticker: [(ts, yes_price, count, taker_outcome_side)]} with a per-market disk cache."""
    todo = [m for m in markets if not os.path.exists(os.path.join(CACHE, m["ticker"] + ".json"))]
    print(f"  trades: {len(markets) - len(todo)} markets cached, fetching {len(todo)} ...", flush=True)

    def fetch(m):
        sess = session()
        tr = paged(sess, "/markets/trades", "trades", {"ticker": m["ticker"], "limit": 1000}, 50)
        rows = [(ts(t["created_time"]), float(t["yes_price_dollars"]), float(t["count_fp"]),
                 t.get("taker_outcome_side") or t.get("taker_side")) for t in tr]
        json.dump(rows, open(os.path.join(CACHE, m["ticker"] + ".json"), "w"))
        return 1

    done, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(fetch, todo):
            done += 1
            if done % 250 == 0:
                print(f"    {done}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return {m["ticker"]: json.load(open(os.path.join(CACHE, m["ticker"] + ".json"))) for m in markets}


# ----------------------------------------------------------------- simulation
def cluster(p, side, spot):
    # BTC and ETH move together (2026-09-21: +7% / +6% in a day), so up/down is one cluster across assets
    a = p["asset"]
    if p["kind"] == "above":
        return "crypto:down" if side == "YES" else "crypto:up"
    if p["kind"] == "below":
        return "crypto:up" if side == "YES" else "crypto:down"
    if side == "YES":
        return f"{a}:range"
    return "crypto:up" if p["k1"] > spot else "crypto:down" if p["k2"] < spot else f"{a}:pin"


def dvol_history(asset, since_ms, until_ms):
    """[(ts_seconds, dvol)] hourly closes of Deribit's implied-vol index."""
    out, end = [], until_ms
    while end > since_ms:
        d = get_json("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                     params={"currency": asset, "start_timestamp": since_ms, "end_timestamp": end,
                             "resolution": "3600"})
        r = (d or {}).get("result") or {}
        out += [(int(row[0]) // 1000, float(row[4])) for row in r.get("data") or []]
        end = r.get("continuation")
        if not end:
            break
    return sorted(set(out))


def opportunities(data, trades, hist, surfaces, a, dvol=None):
    """Every fill our resting quotes would have received, before bankroll limits."""
    opps = []
    for m in data["markets"]:
        if m["series"] not in a.series:
            continue
        p, surf, h = m["parsed"], surfaces[m["asset"]], hist[m["asset"]]
        for t, y, n, taker in trades.get(m["ticker"], []):
            hours = (m["close"] - t) / 3600
            if not (a.min_hours <= hours <= a.max_hours):
                continue
            spot_q = spot_at(h, t - a.lag_min * 60)          # what our resting quote was based on
            scale = 1.0
            if dvol and dvol.get(m["asset"]):
                dv = dvol[m["asset"]]
                then = spot_at(dv, t - a.lag_min * 60)
                scale = then / dv[-1][1] if then else 1.0
            band = fair_band(surf, p, hours / 24, spot_q, a.vol_haircut, a.vol_bump, scale)
            if not band:
                continue
            lo, mid, hi = band
            if taker == "yes" and a.min_yes <= y < a.max_yes:
                ask = math.ceil((hi + a.edge) * 100) / 100   # our resting YES ask = buy NO at 1-ask
                if y >= ask:
                    opps.append({"t": t, "m": m, "side": "NO", "price": round(1 - ask, 2), "q": 1 - hi,
                                 "n": n * a.share, "hours": hours, "spot": spot_q})
            elif taker == "no" and p["kind"] == "between" and hours >= a.yes_min_hours \
                    and a.yes_min <= y < a.yes_max:
                bid = math.floor((lo - a.edge) * 100) / 100
                if 0 < bid and y <= bid:
                    opps.append({"t": t, "m": m, "side": "YES", "price": bid, "q": lo,
                                 "n": n * a.share, "hours": hours, "spot": spot_q})
    opps.sort(key=lambda o: o["t"])
    return opps


def simulate(data, opps, a):
    cash, events = a.bankroll, []
    for o in opps:
        events.append((o["t"], 0, o))
    for m in data["markets"]:
        events.append((m["close"], 1, m))
    events.sort(key=lambda e: (e[0], e[1]))
    pos = {}                      # (ticker, side) -> [contracts, cost, market, cluster]
    exp_mkt, exp_evt, exp_clu = defaultdict(float), defaultdict(float), defaultdict(float)
    gross, fills, closed = 0.0, [], []
    day_eq, peak, max_dd = {}, a.bankroll, 0.0
    for t, kind, x in events:
        if kind == 1:            # settlement
            for side in ("NO", "YES"):
                k = (x["ticker"], side)
                if k not in pos:
                    continue
                n, cost, m, clu = pos.pop(k)
                payout = n * (1.0 if (x["result"] == "yes") == (side == "YES") else 0.0)
                cash += payout
                gross -= cost
                exp_mkt[m["ticker"]] -= cost
                exp_evt[m["event"]] -= cost
                exp_clu[clu] -= cost
                closed.append({"ticker": x["ticker"], "series": x["series"], "side": side, "cost": cost,
                               "pnl": payout - cost, "t": t})
        else:
            o, m = x, x["m"]
            equity = cash + gross
            price, q = o["price"], o["q"]
            fee_rate = data["fees"].get(m["series"], 0.0)
            clu = cluster(m["parsed"], o["side"], o["spot"])
            kelly = a.kelly * max(q - price, 0) / max(1 - price, 1e-9)
            budget = min(kelly * equity,
                         a.cap_market * equity - exp_mkt[m["ticker"]],
                         a.cap_event * equity - exp_evt[m["event"]],
                         a.cap_cluster * equity - exp_clu[clu],
                         a.cap_gross * equity - gross,
                         cash)
            n = math.floor(min(o["n"], budget / price))
            if n < 1:
                continue
            fee = maker_fee(fee_rate, price, n)
            cost = n * price + fee
            cash -= cost
            gross += cost
            exp_mkt[m["ticker"]] += cost
            exp_evt[m["event"]] += cost
            exp_clu[clu] += cost
            p = pos.setdefault((m["ticker"], o["side"]), [0, 0.0, m, clu])
            p[0] += n
            p[1] += cost
            fills.append({"t": t, "ticker": m["ticker"], "side": o["side"], "n": n, "price": price,
                          "hours": o["hours"]})
        eq = cash + gross
        day = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
        day_eq[day] = eq
        peak = max(peak, eq)
        max_dd = max(max_dd, 1 - eq / peak)
    return cash + gross, fills, closed, day_eq, max_dd


def report(a, data, opps, final, fills, closed, day_eq, max_dd):
    print(f"\n=== bankroll ${a.bankroll:,.0f}  edge {a.edge * 100:.0f}c  lag {a.lag_min:.0f}min  share {a.share:.0%}  "
          f"hours [{a.min_hours}, {a.max_hours}] ===")
    days = sorted(day_eq)
    prev = a.bankroll
    rets = []
    for d in days:
        r = day_eq[d] / prev - 1
        rets.append(r)
        print(f"  {d}  equity ${day_eq[d]:>12,.2f}   day {r:+7.2%}")
        prev = day_eq[d]
    pnl = sum(c["pnl"] for c in closed)
    staked = sum(c["cost"] for c in closed)
    wins = sum(1 for c in closed if c["pnl"] > 0)
    print(f"\n  final ${final:,.2f}  return {final / a.bankroll - 1:+.1%}  max drawdown {max_dd:.1%}  "
          f"worst day {min(rets) if rets else 0:+.1%}  best day {max(rets) if rets else 0:+.1%}")
    print(f"  {len(fills)} fills, {len(closed)} settled positions ({wins} won), P&L {pnl:+,.2f} on "
          f"${staked:,.0f} staked = {pnl / staked if staked else 0:+.2%} per $")
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for c in closed:
        for k in (c["series"], c["side"]):
            by[k][0] += 1
            by[k][1] += c["cost"]
            by[k][2] += c["pnl"]
    for k, (n, c, p) in sorted(by.items()):
        print(f"    {k:<8} {n:>6} positions  staked ${c:>11,.0f}  P&L {p:>+11,.2f}  ({p / c if c else 0:+.2%})")
    # unconstrained flow by time left, settled on the real result
    hb = defaultdict(lambda: [0.0, 0.0])
    for o in opps:
        won = (o["m"]["result"] == "yes") == (o["side"] == "YES")
        cost = o["n"] * o["price"]
        b = "<1h" if o["hours"] < 1 else "1-6h" if o["hours"] < 6 else "6-12h" if o["hours"] < 12 else "12h+"
        for k in (b, f"{o['side']} {b}"):
            hb[k][0] += cost
            hb[k][1] += o["n"] * (1.0 if won else 0.0) - cost
    print("  flow at our prices (no bankroll limit), real outcomes, by time left and side:")
    for b in ("<1h", "1-6h", "6-12h", "12h+", "NO <1h", "NO 1-6h", "NO 6-12h", "NO 12h+", "YES 12h+"):
        c, p = hb[b]
        if c:
            print(f"    {b:<9} ${c:>12,.0f}   P&L {p:>+12,.0f}   {p / c:+.2%} per $")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--end-days-ago", type=int, default=0,
                    help="leave the most recent N days out (out-of-sample tests on older data)")
    ap.add_argument("--refresh", action="store_true", help="re-list markets (trades stay cached)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bankroll", type=float, default=1000)
    ap.add_argument("--series", nargs="+", default=list(SERIES), help="series to trade (data for all is cached)")
    ap.add_argument("--edge", type=float, default=0.02)
    ap.add_argument("--lag-min", type=float, default=2.0, help="minutes our resting quote lags the market")
    ap.add_argument("--share", type=float, default=0.25, help="share of each crossing trade we'd get")
    ap.add_argument("--min-hours", type=float, default=0.0)
    ap.add_argument("--max-hours", type=float, default=72.0)
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
    ap.add_argument("--cap-cluster", type=float, default=0.15)
    ap.add_argument("--no-dvol", action="store_true",
                    help="price past days with today's IV as-is (default: scale it by DVOL then / DVOL now)")
    ap.add_argument("--cap-gross", type=float, default=0.90)
    a = ap.parse_args()

    print(f"loading {a.days} days of settled Kalshi crypto markets ...", flush=True)
    data = load_markets(a.days, a.refresh, a.series, a.end_days_ago)
    trades = load_trades(data["markets"], a.workers)
    since = min(m["close"] for m in data["markets"]) - 3 * 86400
    surfaces = {x: CL.Surface(x) for x in set(SERIES.values())}
    hist = {x: spot_history(x, int(since * 1000), int(time.time() * 1000)) for x in surfaces}
    print(f"  {sum(len(v) for v in trades.values()):,} trades in {len(trades)} markets; "
          f"spot history {', '.join(f'{k} {len(v)}' for k, v in hist.items())} candles", flush=True)
    dvol = None
    if not a.no_dvol:
        dvol = {x: dvol_history(x, int(since * 1000), int(time.time() * 1000)) for x in surfaces}
        print("  DVOL " + ", ".join(f"{k} {min(r[1] for r in v):.0f}-{max(r[1] for r in v):.0f} (now {v[-1][1]:.0f})"
                                    for k, v in dvol.items() if v), flush=True)
    opps = opportunities(data, trades, hist, surfaces, a, dvol)
    final, fills, closed, day_eq, max_dd = simulate(data, opps, a)
    report(a, data, opps, final, fills, closed, day_eq, max_dd)
    print("\nnote: today's IV surface is applied to past days; a bigger --lag-min and smaller --share are the "
          "conservative stresses.")


if __name__ == "__main__":
    main()
