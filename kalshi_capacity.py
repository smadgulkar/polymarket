#!/usr/bin/env python3
"""
kalshi_capacity.py  --  the engine's edge on Kalshi crypto markets, measured from real fills

Same strategy as the Polymarket engine (resting bids on the side the options market says is
underpriced), on Kalshi -- CFTC-regulated and legal for US residents. Kalshi's public trade
feed records the AGGRESSOR (taker_side), so unlike Polymarket this counts only trades a
resting order on our side would actually have filled:

  NO side   taker BOUGHT YES at y (3-25c) above its cautious Deribit fair value + edge
            -> the maker who sold that YES (= bought NO at 1-y) earned the edge
  YES side  taker SOLD YES at y (10-70c) below cautious fair value - edge, on a price
            RANGE with >= --yes-min-hours left -> the maker who bought that YES earned it

Markets: Kalshi BTC/ETH daily above/below (KXBTCD, KXETHD) and ranges (KXBTC, KXETH), which
settle on the 60-second average of the CF Benchmarks real-time index before the stated time.
Fair value at each trade's moment: Deribit options surface (today's), Deribit 5-minute spot
history applied sticky-moneyness, cautious IV band x {0.85, 1, 1.1} -- same as the engine.

Run (public data, no account needed):
  python kalshi_capacity.py                     # last 24h
  python kalshi_capacity.py --hours 48 --share 0.3
"""

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

from curl_cffi import requests

import polymarket_crypto_longshots as CL
from polymarket_capacity import PERPS, spot_history, spot_at, fair_band

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"KXBTCD": "BTC", "KXETHD": "ETH", "KXBTC": "BTC", "KXETH": "ETH"}
OUT = "kalshi_capacity_trades.csv"
S = requests.Session(impersonate="chrome")


def get(path, params, retries=5):
    for attempt in range(retries):
        try:
            r = S.get(API + path, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            return r.json() if r.status_code == 200 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


def paged(path, key, params, max_pages=50):
    out, cursor = [], None
    for _ in range(max_pages):
        p = dict(params, **({"cursor": cursor} if cursor else {}))
        d = get(path, p) or {}
        out += d.get(key) or []
        cursor = d.get("cursor")
        if not cursor:
            break
        time.sleep(0.05)
    return out


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parsed_of(m, asset):
    """Kalshi strike fields -> the same parsed shape the Deribit model uses."""
    st, lo, hi = m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")
    if st == "greater" and lo is not None:
        return {"asset": asset, "kind": "above", "k1": float(lo), "k2": None}
    if st == "less" and hi is not None:
        return {"asset": asset, "kind": "below", "k1": float(hi), "k2": None}
    if st == "between" and lo is not None and hi is not None:
        return {"asset": asset, "kind": "between", "k1": float(lo), "k2": float(hi)}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--yes-min-hours", type=float, default=12)
    ap.add_argument("--vol-haircut", type=float, default=0.15)
    ap.add_argument("--vol-bump", type=float, default=0.10)
    ap.add_argument("--share", type=float, default=0.25, help="assumed share of qualifying maker fills we'd win")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    since = int((now - timedelta(hours=args.hours)).timestamp())
    surfaces = {a: CL.Surface(a) for a in set(SERIES.values())}
    hist = {a: spot_history(a, since * 1000 - 3600_000, int(now.timestamp() * 1000)) for a in surfaces}
    print("deribit: " + ", ".join(f"{a} {s.spot:,.0f} ({len(hist[a])} candles)" for a, s in surfaces.items()))

    markets = []
    for ser, asset in SERIES.items():
        ms = paged("/markets", "markets", {"series_ticker": ser, "min_close_ts": since, "limit": 1000})
        ms = [m for m in ms if float(m.get("volume_fp") or 0) > 0 and parsed_of(m, asset)]
        markets += [(m, asset) for m in ms]
        print(f"  {ser}: {len(ms)} traded markets closing after the window start")

    rows, every, t0 = [], 0.0, time.time()
    for i, (m, asset) in enumerate(markets):
        p, close = parsed_of(m, asset), ts(m["close_time"])
        for t in paged("/markets/trades", "trades", {"ticker": m["ticker"], "min_ts": since, "limit": 1000}, 20):
            tts, y, n = ts(t["created_time"]), float(t["yes_price_dollars"]), float(t["count_fp"])
            every += n * min(y, 1 - y)
            hours_left = (close - tts) / 3600
            if hours_left <= 0:
                continue
            band = fair_band(surfaces[asset], p, hours_left / 24, spot_at(hist[asset], tts),
                             args.vol_haircut, args.vol_bump)
            if not band:
                continue
            lo, mid, hi = band
            side = None
            if t["taker_side"] == "yes" and 0.03 <= y < 0.25 and y - hi >= args.min_edge:
                side, edge, cost = "NO", y - hi, n * (1 - y)
            elif t["taker_side"] == "no" and p["kind"] == "between" and hours_left >= args.yes_min_hours \
                    and 0.10 <= y < 0.70 and lo - y >= args.min_edge:
                side, edge, cost = "YES", lo - y, n * y
            if side:
                rows.append({"ts": int(tts), "ticker": m["ticker"], "kind": p["kind"], "side": side,
                             "yes_price": y, "fair_mid": round(mid, 4), "edge_per_contract": round(edge, 4),
                             "contracts": n, "cost_usd": round(cost, 2), "edge_usd": round(edge * n, 2),
                             "hours_left": round(hours_left, 2)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(markets)} markets, {len(rows)} qualifying maker fills ({time.time() - t0:.0f}s)")

    if rows:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    per_day = 24 / args.hours
    print(f"\n=== Kalshi crypto: maker fills on our side, last {args.hours:.0f}h, per day "
          f"(edge >= {args.min_edge * 100:.0f}c, aggressor-verified) ===")
    print(f"{'group':<24}{'fills/day':>10}{'$ /day':>11}{'edge $/day':>12}{'edge/$':>8}")

    def line(label, rs):
        c, e = sum(r["cost_usd"] for r in rs), sum(r["edge_usd"] for r in rs)
        print(f"{label:<24}{len(rs) * per_day:>10,.0f}{c * per_day:>11,.0f}{e * per_day:>12,.0f}{(e / c if c else 0):>8.1%}")
    line("ALL", rows)
    for s in ("NO", "YES"):
        line(f"  side {s}", [r for r in rows if r["side"] == s])
    for ser in SERIES:
        line(f"  {ser}", [r for r in rows if r["ticker"].startswith(ser + "-")])
    for a, b in ((0, 1), (1, 6), (6, 12), (12, 72)):
        line(f"  {a}-{b}h left", [r for r in rows if a <= r["hours_left"] < b])
    edge = sum(r["edge_usd"] for r in rows) * per_day
    flow = sum(r["cost_usd"] for r in rows) * per_day
    hold = statistics.median([r["hours_left"] for r in rows]) / 24 if rows else 0
    print(f"\ncontext: all trading in these markets ~${every * per_day:,.0f}/day (cheaper side of each trade)")
    print(f"at {args.share:.0%} capture: ~${flow * args.share:,.0f}/day of fills, ~${edge * args.share:,.0f}/day edge "
          f"(~${edge * args.share * 30:,.0f}/month); median {hold * 24:.1f}h to close -> "
          f"useful bankroll ~${flow * args.share * max(hold, 0.25):,.0f}")
    print(f"rows in {OUT}. Kalshi taker fees don't apply to resting orders on most markets; check yours.")


if __name__ == "__main__":
    main()
