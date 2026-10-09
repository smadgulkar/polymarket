#!/usr/bin/env python3
"""
polymarket_weather_accuracy.py  --  does the weather forecast beat the market, or is our 'edge' model error?

The capacity scan found half of all weather ranges priced outside our cautious forecast
band. Either the market is badly wrong, or our forecast is (wrong station, grid-vs-airport
bias, coastal cells). This settles it on RESOLVED markets, before any money:

  for every resolved temperature event in the last --days days, at T = 12h before the
  station-local day starts:
    MARKET  each range's price at T (Polymarket price history), normalized to sum to 1
    MODEL   the day-ahead forecast available at T (Open-Meteo Previous Runs API: ECMWF IFS
            0.25 + GFS, hourly values as predicted ~24h before, -> that local day's high/low),
            spread with a Gaussian, two versions:
              raw         no bias correction, sigma from --sigma-f / --sigma-c
              calibrated  per-station bias and sigma learned LEAVE-ONE-DAY-OUT (only from the
                          station's other days, so a day never grades itself)
  scores   log loss on the outcome that happened (lower = better), market vs model
  TRADES   replays the engine rule at T prices: NO when YES (3-25c) >= model + edge,
           YES when YES (10-70c) <= model - edge; P&L per $ on the real outcome

Verdict per station and overall: a station is tradeable only if the calibrated model's
simulated trades made money there AND its log loss beat the market's.

Run (where Polymarket is reachable):  python polymarket_weather_accuracy.py --days 7
"""

import argparse
import csv
import math
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

from polymarket_capacity import gamma_markets
from polymarket_engine import CLOB, get_json, parse_iso, parse_list, to_float, now_utc, resolve_tags, tokens_of
from polymarket_weather_capacity import (Forecasts, parse_bucket, parse_date, parse_station, temp_kind,
                                         is_weather_temp, ncdf)

PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = ("ecmwf_ifs025", "gfs_seamless")
OUT = "weather_accuracy_trades.csv"


def price_at(token, t_end):
    """Last price-history point at or before t_end (unix seconds), or None."""
    d = get_json(f"{CLOB}/prices-history", params={"market": token, "startTs": t_end - 3 * 86400,
                                                   "endTs": t_end, "fidelity": 60})
    pts = [(x["t"], x["p"]) for x in (d or {}).get("history", []) if x.get("t", 0) <= t_end]
    return max(pts)[1] if pts else None


class DayAhead:
    """Day-ahead daily high/low per station-local date from previous_day1 hourly forecasts."""

    def __init__(self, fc, days):
        self.fc, self.days, self.cache = fc, days, {}

    def get(self, station, unit, day, kind):
        key = (station, unit)
        if key not in self.cache:
            lat, lon = self.fc.coords.get(station, (None, None))
            d = None
            if lat is not None:
                d = get_json(PREV, params={"latitude": lat, "longitude": lon, "timezone": "auto",
                                           "hourly": "temperature_2m_previous_day1", "models": ",".join(MODELS),
                                           "temperature_unit": "fahrenheit" if unit == "F" else "celsius",
                                           "past_days": self.days + 3, "forecast_days": 1})
                time.sleep(0.2)
            self.cache[key] = d
        d = self.cache[key]
        if not d or "hourly" not in d:
            return None, None
        h = d["hourly"]
        series = [h[k] for k in h if k.startswith("temperature_2m_previous_day1")]
        vals = []
        for s in series:
            day_vals = [v for t, v in zip(h["time"], s) if t.startswith(day.isoformat()) and v is not None]
            if len(day_vals) >= 20:
                vals.append(max(day_vals) if kind == "max" else min(day_vals))
        offset = d.get("utc_offset_seconds", 0)
        return (statistics.mean(vals) if vals else None), offset


def bucket_p(mu, sigma, lo, hi):
    a = -math.inf if lo is None else lo - 0.5
    b = math.inf if hi is None else hi + 0.5
    return (ncdf((b - mu) / sigma) if b != math.inf else 1.0) - (ncdf((a - mu) / sigma) if a != -math.inf else 0.0)


def bucket_center(lo, hi):
    if lo is not None and hi is not None:
        return (lo + hi) / 2
    return (hi - 1) if lo is None else (lo + 1)  # open-ended tails: a degree past the edge


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--hours-before", type=float, default=12, help="price/forecast taken this long before the day")
    ap.add_argument("--sigma-f", type=float, default=2.0, help="raw model spread, degF")
    ap.add_argument("--sigma-c", type=float, default=1.2, help="raw model spread, degC")
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--max-events", type=int, default=2000)
    args = ap.parse_args()

    now = now_utc()
    tag = resolve_tags(["weather"]).get("weather")
    if not tag:
        sys.exit("could not reach Polymarket (gamma-api) -- run this where the engine runs")
    mkts = [m for m in gamma_markets(tag, now - timedelta(days=args.days), now, closed=True) if is_weather_temp(m)]
    print(f"resolved temperature markets in the last {args.days} days: {len(mkts)}")

    # group ranges into events: (station, kind, unit, date)
    events, fc = {}, Forecasts()
    for m in mkts:
        b = parse_bucket(m.get("groupItemTitle") or "") or parse_bucket(m.get("question"))
        st = parse_station((m.get("description") or "") + " " + (m.get("resolutionSource") or ""))
        end = parse_iso(m.get("endDate"))
        day = parse_date(m.get("question"), end)
        toks, fin = tokens_of(m), parse_list(m.get("outcomePrices"))
        if not (b and st and day and toks and fin):
            continue
        won = to_float(fin[toks[2]])
        if won is None or 0.02 < won < 0.98:
            continue  # not cleanly resolved
        events.setdefault((st, temp_kind(m.get("question")), b[2], day), []).append(
            {"lo": b[0], "hi": b[1], "yes_tok": toks[0], "won": won > 0.5, "q": m.get("question")})
    events = {k: v for k, v in events.items() if sum(r["won"] for r in v) == 1 and len(v) >= 5}
    fc.load_stations([k[0] for k in events])
    keys = sorted(events, key=lambda k: (k[3], k[0]))[-args.max_events:]
    print(f"{len(events)} complete events (one winning range); scoring {len(keys)} "
          f"across {len({k[0] for k in keys})} stations")

    # forecasts and market prices at T
    da, recs = DayAhead(fc, args.days), []
    t0 = time.time()
    for i, key in enumerate(keys):
        st, kind, unit, day = key
        mu, offset = da.get(st, unit, day, kind)
        if mu is None:
            continue
        day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(seconds=offset)
        t_price = int((day_start - timedelta(hours=args.hours_before)).timestamp())
        rows = events[key]
        prices = [price_at(r["yes_tok"], t_price) for r in rows]
        if any(p is None for p in prices) or sum(prices) <= 0:
            continue
        tot = sum(prices)
        winner = next(r for r in rows if r["won"])
        recs.append({"station": fc.label.get(st, st), "kind": kind, "unit": unit, "day": day, "mu": mu,
                     "actual": bucket_center(winner["lo"], winner["hi"]), "rows": rows,
                     "prices": prices, "mkt_norm": [p / tot for p in prices]})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(keys)} events priced ({time.time() - t0:.0f}s)")
    print(f"{len(recs)} events with forecast + market prices at T-{args.hours_before:.0f}h")
    if not recs:
        return

    def sigma0(unit):
        return args.sigma_f if unit == "F" else args.sigma_c

    def calibrated(rec):
        """Bias and spread for this station+kind from its OTHER days only."""
        errs = [r["actual"] - r["mu"] for r in recs if r["station"] == rec["station"] and r["kind"] == rec["kind"]
                and r["day"] != rec["day"]]
        if len(errs) < 3:
            return 0.0, sigma0(rec["unit"])
        return statistics.mean(errs), max(statistics.pstdev(errs), 0.6 * sigma0(rec["unit"]))

    trades, score = [], {}
    for rec in recs:
        for variant in ("raw", "calibrated"):
            bias, sig = (0.0, sigma0(rec["unit"])) if variant == "raw" else calibrated(rec)
            ps = [max(bucket_p(rec["mu"] + bias, sig, r["lo"], r["hi"]), 1e-4) for r in rec["rows"]]
            tot = sum(ps)
            ps = [p / tot for p in ps]
            wi = next(i for i, r in enumerate(rec["rows"]) if r["won"])
            s = score.setdefault((rec["station"], variant), {"model": 0.0, "market": 0.0, "n": 0})
            s["model"] -= math.log(ps[wi])
            s["market"] -= math.log(max(rec["mkt_norm"][wi], 1e-4))
            s["n"] += 1
            for r, y, p in zip(rec["rows"], rec["prices"], ps):
                side = None
                if 0.03 <= y < 0.25 and y - p >= args.min_edge:
                    side, cost, pay = "NO", 1 - y, (0.0 if r["won"] else 1.0)
                elif 0.10 <= y < 0.70 and p - y >= args.min_edge:
                    side, cost, pay = "YES", y, (1.0 if r["won"] else 0.0)
                if side:
                    trades.append({"variant": variant, "station": rec["station"], "kind": rec["kind"],
                                   "day": rec["day"].isoformat(), "range": f"{r['lo']}-{r['hi']}", "side": side,
                                   "yes_price": round(y, 4), "model_p": round(p, 4), "cost": round(cost, 4),
                                   "pnl": round(pay - cost, 4), "won": r["won"]})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trades[0].keys()) if trades else ["variant"])
        w.writeheader()
        w.writerows(trades)

    for variant in ("raw", "calibrated"):
        tv = [t for t in trades if t["variant"] == variant]
        m = sum(s["model"] for (st, v), s in score.items() if v == variant)
        k = sum(s["market"] for (st, v), s in score.items() if v == variant)
        n = sum(s["n"] for (st, v), s in score.items() if v == variant)
        cost, pnl = sum(t["cost"] for t in tv), sum(t["pnl"] for t in tv)
        print(f"\n=== {variant.upper()} model: {n} events ===")
        print(f"  log loss per event  model {m / n:.3f}  vs  market {k / n:.3f}   "
              f"({'MODEL better' if m < k else 'MARKET better'})")
        print(f"  simulated trades    {len(tv)} trades, ${cost:,.2f} staked per 1 share each, "
              f"P&L {pnl:+,.2f}  ->  {pnl / cost if cost else 0:+.1%} per $")
        for side in ("NO", "YES"):
            ts = [t for t in tv if t["side"] == side]
            c, p = sum(t["cost"] for t in ts), sum(t["pnl"] for t in ts)
            if ts:
                print(f"    {side:<3} {len(ts):>5} trades  {p / c:+.1%} per $  "
                      f"(won {sum(1 for t in ts if t['pnl'] > 0)}/{len(ts)})")

    print("\n=== per station (calibrated model) -- tradeable only if P&L > 0 AND model log loss < market ===")
    print(f"  {'station':<8}{'events':>7}{'model LL':>10}{'market LL':>11}{'trades':>8}{'P&L/$':>8}  verdict")
    good = []
    for st in sorted({k[0] for k in score}):
        s = score.get((st, "calibrated"))
        if not s:
            continue
        tv = [t for t in trades if t["variant"] == "calibrated" and t["station"] == st]
        c, p = sum(t["cost"] for t in tv), sum(t["pnl"] for t in tv)
        ok = tv and p > 0 and s["model"] < s["market"] and s["n"] >= 5
        good += [st] if ok else []
        print(f"  {st:<8}{s['n']:>7}{s['model'] / s['n']:>10.3f}{s['market'] / s['n']:>11.3f}{len(tv):>8}"
              f"{(p / c if c else 0):>+8.1%}  {'TRADEABLE' if ok else ''}")
    print(f"\ntradeable stations: {', '.join(good) if good else 'none'}")
    print(f"trades in {OUT}. Small samples: treat a station as a candidate, not a conclusion, below ~20 events.")


if __name__ == "__main__":
    main()
