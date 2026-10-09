#!/usr/bin/env python3
"""
polymarket_weather_capacity.py  --  is there an engine-sized edge in Polymarket's weather markets?

Same question as polymarket_capacity.py, new fair-value source. The 588M-trade study
(arXiv 2609.12878) found weather longshots the MOST overpriced category (-25%), and
weather has a free, strong forecast: ensemble NWP.

Markets: "Highest temperature in <city> on <date>?" -- ~11 ranges per event ("41°F or
below", "52-53°F", "62°F or higher"), settled on NOAA's whole-degree daily high at a named
airport station (Weather Underground as fallback).

Fair value: Open-Meteo ensemble daily max at the station (ECMWF IFS 51 + GFS 31 members,
station-local day). Each member is smoothed with a Gaussian (ensembles are under-dispersed)
and the probability of each range computed. CAUTIOUS band = min/max over smoothing
sigma {1.0, 1.75, 2.5}°F and a station-vs-grid bias {-1, 0, +1}°F, so a trade only counts
if it's mispriced under all of them.

Measures:
  SNAPSHOT  open markets now: last price vs fair, how many ranges sit outside the cautious band
  FLOW      trades in the last --hours on markets still >= 12h from their date, valued against
            the CURRENT forecast: NO-side (YES 3-25c above cautious fair + edge) and YES-side
            (YES 10-70c below cautious fair - edge), $/day and edge $/day, by city

Caveats stated once: past trades are valued against today's forecast (forecasts move; this
can over- or under-state edge for trades many hours old -- keep --hours short); the grid-cell
forecast isn't the airport sensor (the +/-1°F bias band is a guess, not a calibration);
aggressor side unknown, so flow is an upper bound. If this shows real money, the next step is
calibrating against resolved markets, not trading.

Run (where Polymarket is reachable):
  python polymarket_weather_capacity.py                 # last 24h of flow + snapshot
  python polymarket_weather_capacity.py --hours 12 --unparsed
"""

import argparse
import csv
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone

from polymarket_capacity import gamma_markets, market_trades
from polymarket_engine import SESSION, get_json, parse_iso, parse_list, to_float, now_utc, resolve_tags, tokens_of

ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                      "august", "september", "october", "november", "december"], 1)}
SIGMAS = (1.0, 1.75, 2.5)
BIASES = (-1.0, 0.0, 1.0)
OUT = "weather_capacity_trades.csv"


# ----------------------------------------------------------------- parsing
def parse_bucket(text):
    """'52-53°F' -> (52, 53, 'F'); '41°F or below' -> (None, 41, 'F'); '62°F or higher' -> (62, None, 'F')."""
    t = (text or "").replace("º", "°").replace("–", "-")
    m = re.search(r"(-?\d+)\s*°?\s*-\s*(-?\d+)\s*°\s*([FC])", t)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    m = re.search(r"(-?\d+)\s*°\s*([FC])\s*or\s*(below|lower|less)", t, re.I)
    if m:
        return None, int(m.group(1)), m.group(2)
    m = re.search(r"(-?\d+)\s*°\s*([FC])\s*or\s*(higher|above|more)", t, re.I)
    if m:
        return int(m.group(1)), None, m.group(2)
    m = re.search(r"\b(-?\d+)\s*°\s*([FC])\b", t)
    if m:
        return int(m.group(1)), int(m.group(1)), m.group(2)
    return None


def parse_station(text):
    """Settlement station: an ICAO code from URLs (weather.gov ...site=KLGA, wunderground .../KLGA),
    'name:<airport name>' from 'NOAA at the <name> Station', or 'HKO' for the Hong Kong Observatory."""
    for pat in (r"site=([A-Z][A-Z0-9]{3})\b", r"/history/daily/[^\s\"')]*?/([A-Z][A-Z0-9]{3})\b",
                r"\(([A-Z][A-Z0-9]{3})\)", r"\b(K[A-Z]{3})\b"):
        m = re.search(pat, text or "")
        if m:
            return m.group(1)
    m = re.search(r"NOAA at the (.+?) Station\b", text or "")
    if m:
        return "name:" + m.group(1).strip()
    if "weather.gov.hk" in (text or "") or "Hong Kong Observatory" in (text or ""):
        return "HKO"
    return None


def norm_name(s):
    """'Adolfo Suárez Madrid–Barajas Airport' -> 'adolfo suarez madrid barajas'."""
    s = re.sub(r"[^\w\s]", " ", s)  # punctuation incl. en-dashes -> space, before accents are stripped
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(international|intl|airport|arpt|airfield|aeroporto|aeropuerto|aeroport)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Airports:
    """OurAirports public dataset: ICAO codes, names and coordinates for every airport."""
    URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    CACHE = "ourairports.csv"
    HKO = (22.3019, 114.1742)  # Hong Kong Observatory HQ, Tsim Sha Tsui

    def __init__(self):
        if not os.path.exists(self.CACHE):
            print("  downloading airport list (one time, ~12 MB) ...", flush=True)
            r = SESSION.get(self.URL, timeout=120)
            r.raise_for_status()
            open(self.CACHE, "w", encoding="utf-8").write(r.text)
        # stream the 86k rows and keep only airports with scheduled-size traffic: holding the whole
        # file as dicts doesn't fit next to the engine on a 512 MB droplet
        self.by_code, self.by_name = {}, {}
        with open(self.CACHE, encoding="utf-8") as f:
            for a in csv.DictReader(f):
                if a["type"] not in ("large_airport", "medium_airport"):
                    continue
                code = a["icao_code"] or a["gps_code"] or a["ident"]
                ll = (float(a["latitude_deg"]), float(a["longitude_deg"]))
                if code:
                    self.by_code.setdefault(code, ll)
                self.by_name.setdefault(norm_name(a["name"]), (code, ll))
        print(f"  airport list: {len(self.by_name):,} airports indexed", flush=True)

    def locate(self, station):
        """-> (label, (lat, lon)) or None."""
        if station == "HKO":
            return "HKO", self.HKO
        if not station.startswith("name:"):
            return (station, self.by_code[station]) if station in self.by_code else None
        key = norm_name(station[5:])
        if key in self.by_name:
            return self.by_name[key]
        words = set(key.split())
        # every word of the settlement name inside an airport's name ('malpensa' in 'milan malpensa'):
        # take the shortest such name, i.e. the closest fit
        subset = [(k, v) for k, v in self.by_name.items() if words and words <= set(k.split())]
        if subset:
            return min(subset, key=lambda kv: len(kv[0]))[1]
        best = max(self.by_name.items(), key=lambda kv: len(words & set(kv[0].split())) /
                   max(len(words | set(kv[0].split())), 1), default=None)
        if best and len(words & set(best[0].split())) / max(len(words | set(best[0].split())), 1) >= 0.6:
            return best[1]
        return None


def parse_date(question, end):
    m = re.search(r"\bon\s+([A-Za-z]+)\s+(\d{1,2})", question or "")
    if not m or m.group(1).lower() not in MONTHS or not end:
        return None
    y = end.year
    d = datetime(y, MONTHS[m.group(1).lower()], int(m.group(2))).date()
    if (d - end.date()).days > 200:  # Dec market ending in Jan
        d = d.replace(year=y - 1)
    return d


def is_weather_temp(m):
    return "temperature" in (m.get("question") or "").lower()


def temp_kind(question):
    """'max' for daily-high markets, 'min' for daily-low ones ('lowest temperature in London')."""
    q = (question or "").lower()
    return "min" if ("lowest" in q or "low temperature" in q or "minimum" in q) else "max"


# ----------------------------------------------------------------- forecasts
class Forecasts:
    def __init__(self):
        self.coords, self.cache = {}, {}

    def load_stations(self, ids):
        """Resolve station ids/names to coordinates; self.label maps them to an ICAO-style label."""
        ap = Airports()
        self.label = getattr(self, "label", {})
        for s in set(ids) - set(self.coords):
            hit = ap.locate(s)
            if hit:
                self.label[s], self.coords[s] = hit

    def members(self, station, unit, day, kind="max"):
        """Ensemble daily-max (or daily-min) values for that station-local date, or None."""
        var = f"temperature_2m_{kind}"
        key = (station, unit, kind)
        if key not in self.cache:
            lat, lon = self.coords.get(station, (None, None))
            d = None
            if lat is not None:
                d = get_json(ENSEMBLE, params={"latitude": lat, "longitude": lon, "daily": var,
                                               "temperature_unit": "fahrenheit" if unit == "F" else "celsius",
                                               "timezone": "auto", "forecast_days": 7,
                                               "models": "ecmwf_ifs025,gfs025"})
                time.sleep(0.2)
            self.cache[key] = (d or {}).get("daily")
        daily = self.cache[key]
        if not daily or day.isoformat() not in daily.get("time", []):
            return None
        i = daily["time"].index(day.isoformat())
        vals = [daily[k][i] for k in daily if k.startswith(f"{var}_member") and daily[k][i] is not None]
        return vals or None


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bucket_prob(vals, lo, hi, sigma, bias):
    """P(rounded max in [lo, hi]) with each member smoothed by N(bias, sigma)."""
    a = -math.inf if lo is None else lo - 0.5
    b = math.inf if hi is None else hi + 0.5
    tot = 0.0
    for v in vals:
        mu = v + bias
        tot += (ncdf((b - mu) / sigma) if b != math.inf else 1.0) - (ncdf((a - mu) / sigma) if a != -math.inf else 0.0)
    return tot / len(vals)


def band(vals, lo, hi, unit):
    scale = 1.0 if unit == "F" else 5 / 9  # sigmas/biases are in °F
    ps = [bucket_prob(vals, lo, hi, s * scale, b * scale) for s in SIGMAS for b in BIASES]
    return min(ps), bucket_prob(vals, lo, hi, SIGMAS[1] * scale, 0.0), max(ps)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--min-hours-left", type=float, default=12, help="skip trades closer than this to the market's day")
    ap.add_argument("--share", type=float, default=0.25)
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--unparsed", action="store_true")
    args = ap.parse_args()

    now = now_utc()
    tag = resolve_tags(["weather"]).get("weather")
    if not tag:
        sys.exit("could not reach Polymarket (gamma-api) -- run this where the engine runs")
    since = now - timedelta(hours=args.hours)
    mkts = gamma_markets(tag, since, now + timedelta(days=5), closed=False) + \
        gamma_markets(tag, since, now, closed=True)
    temps = [m for m in mkts if is_weather_temp(m)]
    print(f"polymarket: {len(mkts)} weather markets in the window, {len(temps)} temperature markets")

    fc, cands, unparsed = Forecasts(), [], []
    for m in temps:
        b = parse_bucket(m.get("groupItemTitle") or "") or parse_bucket(m.get("question"))
        st = parse_station((m.get("description") or "") + " " + (m.get("resolutionSource") or ""))
        end = parse_iso(m.get("endDate"))
        day = parse_date(m.get("question"), end)
        toks = tokens_of(m)
        if not (b and st and day and toks and m.get("conditionId")):
            unparsed.append((m, b, st, day))
            continue
        cands.append((m, b, st, day, toks))
    fc.load_stations([c[2] for c in cands])
    print(f"parsed {len(cands)} range markets across {len({c[2] for c in cands})} stations "
          f"({len(fc.coords)} located); {len(unparsed)} unparsed "
          f"({sum(1 for c in cands if temp_kind(c[0].get('question')) == 'min')} of the parsed are daily-low markets)")
    if args.unparsed or not cands:
        # one example per city: show where the settlement station is named, so the parser can learn it
        shown = set()
        for m, b, st, day in unparsed:
            city = re.sub(r".*temperature in (.+?) (be|on) .*", r"\1", m.get("question", ""))
            if city in shown or len(shown) >= 15:
                continue
            shown.add(city)
            text = (m.get("description") or "") + " || " + (m.get("resolutionSource") or "")
            hit = re.search(r"(?i)(station|airport|wunderground|weather\.gov|noaa).{0,160}", text)
            print(f"  {city[:28]:<28} bucket={b} station={st} day={day}\n"
                  f"      {(hit.group(0) if hit else text[:200])!r}")

    # snapshot: last prices of open markets vs fair
    snap, out_band = [], 0
    for m, (lo, hi, unit), st, day, toks in cands:
        if m.get("closed") or day <= now.date():  # same-day markets already know part of the day
            continue
        vals = fc.members(st, unit, day, temp_kind(m.get("question")))
        prices = parse_list(m.get("outcomePrices"))
        if not (vals and prices):
            continue
        y = to_float(prices[toks[2]])
        p_lo, p_mid, p_hi = band(vals, lo, hi, unit)
        if y is None:
            continue
        snap.append((y, p_mid))
        if y - p_hi >= args.min_edge or p_lo - y >= args.min_edge:
            out_band += 1
    if snap:
        err = statistics.mean(abs(y - p) for y, p in snap)
        print(f"\nsnapshot: {len(snap)} open ranges, mean |price - forecast| {err:.3f}, "
              f"{out_band} outside the cautious band by >= {args.min_edge * 100:.0f}c")

    # flow
    rows, t0 = [], time.time()
    for i, (m, (lo, hi, unit), st, day, toks) in enumerate(cands):
        vals = fc.members(st, unit, day, temp_kind(m.get("question")))
        if not vals:
            continue
        p_lo, p_mid, p_hi = band(vals, lo, hi, unit)
        day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)  # rough; local day starts later
        seen = set()
        for t in market_trades(m["conditionId"], int(since.timestamp()), args.max_pages):
            ts, px, sz = int(float(t.get("timestamp", 0))), to_float(t.get("price")), to_float(t.get("size"))
            if not (px and sz) or (day_start.timestamp() - ts) / 3600 < args.min_hours_left:
                continue
            y = px if str(t.get("asset")) == str(toks[0]) else 1 - px
            key = (t.get("transactionHash"), round(y, 4), round(sz, 2))
            if key in seen:
                continue
            seen.add(key)
            side = None
            if 0.03 <= y < 0.25 and y - p_hi >= args.min_edge:
                side, edge, cost = "NO", y - p_hi, sz * (1 - y)
            elif 0.10 <= y < 0.70 and p_lo - y >= args.min_edge:
                side, edge, cost = "YES", p_lo - y, sz * y
            if side:
                rows.append({"ts": ts, "station": fc.label.get(st, st), "kind": temp_kind(m.get("question")), "date": day.isoformat(), "bucket": m.get("groupItemTitle"),
                             "question": m.get("question"), "side": side, "yes_price": round(y, 4),
                             "fair_mid": round(p_mid, 4), "edge_per_share": round(edge, 4), "shares": sz,
                             "cost_usd": round(cost, 2), "edge_usd": round(edge * sz, 2)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cands)} markets, {len(rows)} mispriced trades ({time.time() - t0:.0f}s)")

    if rows:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    per_day = 24 / args.hours
    print(f"\n=== mispriced weather flow, last {args.hours:.0f}h, per day (edge >= {args.min_edge * 100:.0f}c, "
          f">= {args.min_hours_left:.0f}h before the day) ===")
    print(f"{'group':<26}{'trades/day':>11}{'$ /day':>10}{'edge $/day':>12}{'edge/$':>8}")

    def line(label, rs):
        c, e = sum(r["cost_usd"] for r in rs), sum(r["edge_usd"] for r in rs)
        print(f"{label:<26}{len(rs) * per_day:>11,.0f}{c * per_day:>10,.0f}{e * per_day:>12,.0f}{(e / c if c else 0):>8.1%}")
    line("ALL", rows)
    for s in ("NO", "YES"):
        line(f"  side {s}", [r for r in rows if r["side"] == s])
    by_st = {}
    for r in rows:
        by_st.setdefault(r["station"], []).append(r)
    for st, rs in sorted(by_st.items(), key=lambda kv: -sum(r["edge_usd"] for r in kv[1]))[:12]:
        line(f"  {st}", rs)
    edge = sum(r["edge_usd"] for r in rows) * per_day
    flow = sum(r["cost_usd"] for r in rows) * per_day
    print(f"\nat {args.share:.0%} capture: ~${flow * args.share:,.0f}/day of fills, ~${edge * args.share:,.0f}/day edge "
          f"(~${edge * args.share * 30:,.0f}/month); held ~1-2 days -> useful bankroll ~${flow * args.share * 1.5:,.0f}")
    print(f"rows in {OUT}. Upper bound; next step if it's material: calibrate against resolved markets.")


if __name__ == "__main__":
    main()
