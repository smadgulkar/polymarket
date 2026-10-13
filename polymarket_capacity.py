#!/usr/bin/env python3
"""
polymarket_capacity.py  --  how much money can the engine's edge absorb?

The engine earns when someone trades a crypto contract at a price the options market says
is wrong and our resting bid is on the other side. So capacity is not order-book depth; it
is the FLOW of mispriced trades:

  NO side   a trade with YES at y (3-25c) above its cautious fair value + edge
            -> a lottery buyer we could have sold to (we buy NO at 1-y)
  YES side  a trade with YES at y (10-70c) below its cautious fair value - edge,
            on a price RANGE with >= 12h left -> a seller we could have bought from

Method: every Deribit-priceable crypto market active in the last --hours (open and
already-resolved), all its trades from Polymarket's data API, each valued at the moment
it happened: Deribit 5-minute price history gives the spot then, and fair value is the
current options surface applied sticky-moneyness (strikes scaled by spot_now/spot_then)
over the time that was left -- the same IV x {0.85, 1, 1.1} cautious band as the engine.

Output: mispriced $ per day and edge $ per day, by side / asset / time-to-expiry, the
top markets, and what that implies for bankroll at an assumed capture share.

Caveats stated once: the data API doesn't say which side was the aggressor, so this
counts all volume at mispriced prices -- an UPPER BOUND on fills a maker could get (other
makers compete for the same flow; --share sets the assumed capture); the IV surface is
today's, applied to past trades; Polymarket's API is unreachable from some networks
(run it where the engine runs).

Run:  python polymarket_capacity.py                  # last 48h
      python polymarket_capacity.py --hours 96 --share 0.3
"""

import argparse
import bisect
import csv
import math
import statistics
import sys
import time
from datetime import timedelta

import polymarket_crypto_longshots as CL
from polymarket_engine import GAMMA, DATA, get_json, parse_iso, parse_list, to_float, now_utc, resolve_tags, tokens_of

PERPS = {"BTC": "BTC-PERPETUAL", "ETH": "ETH-PERPETUAL", "SOL": "SOL_USDC-PERPETUAL",
         "XRP": "XRP_USDC-PERPETUAL", "HYPE": "HYPE_USDC-PERPETUAL", "AVAX": "AVAX_USDC-PERPETUAL"}
OUT = "capacity_trades.csv"


def gamma_markets(tag_id, end_min, end_max, closed):
    """All markets under a tag with endDate in [end_min, end_max]; walks past Gamma's offset cap."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    lo, out, seen = end_min.strftime(fmt), [], set()
    while True:
        page, offset = [], 0
        while offset < 1000:
            page = get_json(f"{GAMMA}/markets", params={
                "closed": "true" if closed else "false", "limit": 100, "offset": offset, "tag_id": tag_id,
                "end_date_min": lo, "end_date_max": end_max.strftime(fmt), "order": "endDate", "ascending": "true"})
            if not isinstance(page, list) or not page:
                return out
            for m in page:
                if m.get("id") not in seen:
                    seen.add(m.get("id"))
                    out.append(m)
            if len(page) < 100:
                return out
            offset += 100
            time.sleep(0.1)
        latest = max((e for e in (parse_iso(m.get("endDate")) for m in page) if e), default=None)
        if latest is None:
            return out
        nxt = latest.strftime(fmt)
        lo = nxt if nxt > lo else (latest + timedelta(seconds=1)).strftime(fmt)


def spot_history(asset, since_ms, until_ms):
    """[(ts_seconds, close)] from Deribit 5-minute candles."""
    out, start = [], since_ms
    while start < until_ms:
        end = min(start + 3 * 86400 * 1000, until_ms)
        d = get_json("https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
                     params={"instrument_name": PERPS[asset], "start_timestamp": start, "end_timestamp": end,
                             "resolution": "5"})
        r = (d or {}).get("result") or {}
        out += [(t // 1000, c) for t, c in zip(r.get("ticks") or [], r.get("close") or []) if c]
        start = end
    return sorted(set(out))


def spot_at(hist, ts):
    i = bisect.bisect_right(hist, (ts, float("inf"))) - 1
    return hist[i][1] if i >= 0 else None


def market_trades(cid, since_ts, max_pages):
    out = []
    for page in range(max_pages):
        d = get_json(f"{DATA}/trades", params={"market": cid, "limit": 500, "offset": page * 500})
        if not isinstance(d, list) or not d:
            break
        out += d
        oldest = min(int(float(t.get("timestamp", 0))) for t in d)
        if len(d) < 500 or oldest < since_ts:
            break
        time.sleep(0.1)
    return [t for t in out if int(float(t.get("timestamp", 0))) >= since_ts]


def fair_band(surf, parsed, days_left, spot_then, vh, vb, vol_scale=1.0):
    """Cautious P(YES) band at a past moment: sticky moneyness on today's surface, with the whole
    surface scaled by vol_scale (e.g. DVOL then / DVOL now)."""
    if days_left <= 0 or not spot_then:
        return None
    scale = surf.spot / spot_then  # move the strikes, not the surface: K/spot_then == (K*scale)/spot_now
    p = dict(parsed, k1=parsed["k1"] * scale, k2=parsed["k2"] * scale if parsed["k2"] else None)
    ps = [CL.model_prob(surf, p, days_left / 365, m * vol_scale) for m in (1 - vh, 1.0, 1 + vb)]
    return min(ps), ps[1], max(ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=48)
    ap.add_argument("--min-edge", type=float, default=0.01)
    ap.add_argument("--no-band", nargs=2, type=float, default=[0.03, 0.25], metavar=("MIN", "MAX"))
    ap.add_argument("--yes-band", nargs=2, type=float, default=[0.10, 0.70], metavar=("MIN", "MAX"))
    ap.add_argument("--yes-min-days", type=float, default=0.5)
    ap.add_argument("--max-days", type=float, default=3, help="the engine only trades markets <= this many days out")
    ap.add_argument("--vol-haircut", type=float, default=0.15)
    ap.add_argument("--vol-bump", type=float, default=0.10)
    ap.add_argument("--share", type=float, default=0.25, help="assumed share of mispriced flow our bids would get")
    ap.add_argument("--max-pages", type=int, default=10, help="trade pages (500 each) per market")
    args = ap.parse_args()

    now = now_utc()
    since = now - timedelta(hours=args.hours)
    since_ts = int(since.timestamp())
    surfaces = {a: s for a in PERPS if (s := CL.Surface(a)).ok}
    print("deribit: " + ", ".join(f"{a} {s.spot:,.2f}" for a, s in surfaces.items()))
    hist = {a: spot_history(a, since_ts * 1000 - 3600_000, int(now.timestamp() * 1000)) for a in surfaces}
    print("spot history: " + ", ".join(f"{a} {len(h)} candles" for a, h in hist.items()))

    tag = resolve_tags(["crypto"]).get("crypto")
    if not tag:
        sys.exit("could not reach Polymarket (gamma-api) -- run this where the engine runs")
    mkts = gamma_markets(tag, since, now + timedelta(days=args.max_days), closed=False) + \
        gamma_markets(tag, since, now, closed=True)
    cands = []
    for m in mkts:
        p = CL.parse_question(m.get("question") or "")
        toks, end, cid = tokens_of(m), parse_iso(m.get("endDate")), m.get("conditionId")
        if p and p["asset"] in surfaces and toks and end and cid:
            cands.append((m, p, toks, end, cid))
    print(f"polymarket: {len(mkts)} crypto markets ending in the window, {len(cands)} priceable by Deribit")

    rows, t0 = [], time.time()
    for i, (m, p, toks, end, cid) in enumerate(cands):
        seen = set()
        for t in market_trades(cid, since_ts, args.max_pages):
            ts, px, sz = int(float(t.get("timestamp", 0))), to_float(t.get("price")), to_float(t.get("size"))
            if not (px and sz):
                continue
            y = px if str(t.get("asset")) == str(toks[0]) else 1 - px          # YES-equivalent price
            key = (t.get("transactionHash"), round(y, 4), round(sz, 2))         # maker+taker records -> one trade
            if key in seen:
                continue
            seen.add(key)
            days_left = (end.timestamp() - ts) / 86400
            if days_left <= 0 or days_left > args.max_days:
                continue
            band = fair_band(surfaces[p["asset"]], p, days_left, spot_at(hist[p["asset"]], ts),
                             args.vol_haircut, args.vol_bump)
            if not band:
                continue
            lo, mid, hi = band
            side = edge = None
            if args.no_band[0] <= y < args.no_band[1] and y - hi >= args.min_edge:
                side, edge, cost = "NO", y - hi, sz * (1 - y)
            elif p["kind"] == "between" and days_left >= args.yes_min_days and \
                    args.yes_band[0] <= y < args.yes_band[1] and lo - y >= args.min_edge:
                side, edge, cost = "YES", lo - y, sz * y
            if side:
                rows.append({"ts": ts, "asset": p["asset"], "kind": p["kind"], "question": m.get("question"),
                             "side": side, "yes_price": round(y, 4), "fair_mid": round(mid, 4),
                             "edge_per_share": round(edge, 4), "shares": sz, "cost_usd": round(cost, 2),
                             "edge_usd": round(edge * sz, 2), "days_left": round(days_left, 2)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cands)} markets, {len(rows)} mispriced trades ({time.time() - t0:.0f}s)")

    if rows:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    per_day = 24 / args.hours
    print(f"\n=== mispriced flow, last {args.hours:.0f}h, scaled to per day (edge >= {args.min_edge * 100:.0f}c) ===")
    print(f"{'group':<28}{'trades/day':>11}{'$ /day':>11}{'edge $/day':>12}{'edge/$':>8}")

    def line(label, rs):
        cost, edge = sum(r["cost_usd"] for r in rs), sum(r["edge_usd"] for r in rs)
        print(f"{label:<28}{len(rs) * per_day:>11,.0f}{cost * per_day:>11,.0f}{edge * per_day:>12,.0f}"
              f"{(edge / cost if cost else 0):>8.1%}")
    line("ALL", rows)
    for key, label in (("side", "side"), ("asset", "asset"), ("kind", "kind")):
        for v in sorted({r[key] for r in rows}):
            line(f"  {label} {v}", [r for r in rows if r[key] == v])
    for lo_d, hi_d in ((0, 0.5), (0.5, 1), (1, 3)):
        line(f"  {lo_d}-{hi_d} days left", [r for r in rows if lo_d <= r["days_left"] < hi_d])
    by_q = {}
    for r in rows:
        by_q.setdefault((r["question"], r["side"]), []).append(r)
    print("\ntop markets by edge $:")
    for (q, s), rs in sorted(by_q.items(), key=lambda kv: -sum(r["edge_usd"] for r in kv[1]))[:12]:
        print(f"  {s:<3} ${sum(r['cost_usd'] for r in rs):>8,.0f} traded  ${sum(r['edge_usd'] for r in rs):>7,.0f} edge  {q[:70]}")

    flow = sum(r["cost_usd"] for r in rows) * per_day
    edge = sum(r["edge_usd"] for r in rows) * per_day
    hold = statistics.median([r["days_left"] for r in rows]) if rows else 0
    print(f"\n=== what it implies (assuming our bids capture {args.share:.0%} of this flow) ===")
    print(f"  capturable fills    ${flow * args.share:,.0f}/day")
    print(f"  expected edge       ${edge * args.share:,.0f}/day  (~${edge * args.share * 30:,.0f}/month)")
    print(f"  median days held    {hold:.2f}  (positions resolve or hit fair value sooner)")
    print(f"  useful bankroll     ~${flow * args.share * max(hold, 0.25):,.0f}  -- money beyond this sits unfilled")
    print(f"\nrows in {OUT}. Upper bound: aggressor side unknown, other makers compete for the same flow.")


if __name__ == "__main__":
    main()
