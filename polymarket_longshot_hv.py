#!/usr/bin/env python3
"""
polymarket_longshot_hv.py  --  Measurement #1

Question: in Polymarket markets with REAL books (>= $50k volume), do contracts
priced 2-10c at 7/14/21/30 days before scheduled end resolve YES less often
than their price says?

Method: scan every closed market under a set of non-sports tags (both past and
future scheduled endDates, since long-dated markets often resolve early), keep
only those with volume >= --min-volume that lived >= --min-lifetime-days,
pull the YES token's full price history, sample the last-CHANGED price at each
horizon (only if the market was still open then), and tabulate realized YES
frequency by price bucket.

Setup:
    pip install curl_cffi
Run:
    python polymarket_longshot_hv.py
Overnight, deeper:
    python polymarket_longshot_hv.py --per-tag 30000

Output: one calibration table per horizon + a POOLED [0.02,0.10) line with
95% CI, split by tag. Full samples in longshot_hv_samples.csv.

Decision rule (agreed): pooled 2-10c at 14d and 21d with n >= 150,
realized YES <= 1.5% and CI upper below mean price  =>  edge is real at size.
Realized >= mean price  =>  dead.

Caveats stated once: prices are last-trade, not the ask; depth is unknown;
longshots in the same event resolve together.
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta

try:
    from curl_cffi import requests
except ImportError:
    sys.exit("curl_cffi not installed. Run:  pip install curl_cffi")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SESSION = requests.Session(impersonate="chrome")

DEFAULT_TAGS = ["politics", "crypto", "business", "science", "pop-culture", "world",
                "tech", "economy", "geopolitics", "elections", "finance"]

BUCKETS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35),
           (0.35, 0.65), (0.65, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.01)]


# ----------------------------------------------------------------- helpers
def get_json(url, params=None, retries=4, backoff=1.5):
    last_err = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1) * 2)
                continue
            if r.status_code in (400, 404, 422):
                return {"_http_error": r.status_code}
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (attempt + 1) * 2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    return {"_http_error": "exhausted", "_body": str(last_err)[:120]}


def parse_iso(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00").replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def parse_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - hw, c + hw)


# ----------------------------------------------------------------- gamma
def resolve_tags(slugs):
    out = {}
    for slug in slugs:
        t = get_json(f"{GAMMA}/tags/slug/{slug}")
        if isinstance(t, dict) and "id" in t:
            out[slug] = t["id"]
        else:
            print(f"  tag '{slug}' not found -- skipping")
        time.sleep(0.15)
    return out


def scan_closed_markets(tag_id, max_markets, direction, page_size=100, offset_cap=1000):
    """Walk closed markets under a tag. 'past': endDate <= now, backwards.
    'future': endDate > now but already closed (resolved early), forwards.
    Gamma rejects deep offsets, so we page in endDate windows."""
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_max = now_s if direction == "past" else None
    end_min = now_s if direction == "future" else None
    markets, seen, offset = [], set(), 0
    while len(markets) < max_markets:
        params = {"closed": "true", "limit": page_size, "offset": offset, "tag_id": tag_id,
                  "order": "endDate", "ascending": "false" if direction == "past" else "true"}
        if end_max:
            params["end_date_max"] = end_max
        if end_min:
            params["end_date_min"] = end_min
        page = get_json(f"{GAMMA}/markets", params=params)
        if isinstance(page, dict):
            if offset > 0:
                page = []
            else:
                print(f"    gamma error {page.get('_http_error')} at offset 0; stopping with {len(markets)}",
                      file=sys.stderr)
                break
        for m in page:
            if m.get("id") not in seen:
                seen.add(m.get("id"))
                markets.append(m)
        offset += len(page)
        if (not page) or len(page) < page_size or offset >= offset_cap:
            eds = [parse_iso(m.get("endDate")) for m in markets]
            eds = [e for e in eds if e]
            if not eds:
                break
            offset = 0
            if direction == "past":
                new = (min(eds) - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                if new == end_max:
                    break
                end_max = new
            else:
                new = (max(eds) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                if new == end_min:
                    break
                end_min = new
        time.sleep(0.12)
    return markets[:max_markets]


# ----------------------------------------------------------------- clob
def fetch_history(token_id, fidelity_min):
    data = get_json(f"{CLOB}/prices-history",
                    params={"market": token_id, "interval": "max", "fidelity": fidelity_min})
    if isinstance(data, dict) and "_http_error" in data:
        return None, f"http_{data['_http_error']}"
    hist = data.get("history") if isinstance(data, dict) else None
    if hist is None:
        return None, "no_history"
    pts = []
    for h in hist:
        try:
            pts.append((int(h["t"]), float(h["p"])))
        except (KeyError, TypeError, ValueError):
            pass
    pts.sort()
    return pts, None


def last_changed_price(points, ts):
    """(t_last_change, price) at or before ts. The fidelity series is carried
    forward, so the time of the last CHANGE is the only honest staleness."""
    price, last_change, prev = None, None, None
    for t, p in points:
        if t > ts:
            break
        if prev is None or p != prev:
            last_change = t
        prev, price = p, p
    return None if price is None else (last_change, price)


# ----------------------------------------------------------------- core
def evaluate(m, args):
    base = {"market_id": m.get("id"), "tag": m.get("_tag"), "question": m.get("question"),
            "neg_risk": m.get("negRisk"), "volume": to_float(m.get("volumeNum") or m.get("volume")),
            "created": None, "scheduled_end": None, "resolution_time": None,
            "outcome_yes": None, "skip_reason": None}
    samples = []

    outcomes, final, tokens = parse_list(m.get("outcomes")), parse_list(m.get("outcomePrices")), parse_list(m.get("clobTokenIds"))
    if not (outcomes and final and tokens and len(outcomes) == len(final) == len(tokens) == 2):
        base["skip_reason"] = "not_binary"
        return base, samples
    fp = [to_float(x) for x in final]
    labels = [str(o).strip().lower() for o in outcomes]
    yes_idx = labels.index("yes") if "yes" in labels else 0
    if fp[yes_idx] is None:
        base["skip_reason"] = "bad_final_prices"
        return base, samples
    if fp[yes_idx] >= 0.999:
        base["outcome_yes"] = 1
    elif fp[yes_idx] <= 0.001:
        base["outcome_yes"] = 0
    else:
        base["skip_reason"] = "not_cleanly_resolved"
        return base, samples

    res = parse_iso(m.get("closedTime")) or parse_iso(m.get("umaEndDate")) or parse_iso(m.get("endDate"))
    sched = parse_iso(m.get("endDate"))
    created = parse_iso(m.get("createdAt")) or parse_iso(m.get("startDate"))
    if not (res and sched and created):
        base["skip_reason"] = "missing_dates"
        return base, samples
    if res > datetime.now(timezone.utc):
        base["skip_reason"] = "resolution_in_future"
        return base, samples
    base.update(created=created.isoformat(), scheduled_end=sched.isoformat(), resolution_time=res.isoformat())

    pts, err = fetch_history(tokens[yes_idx], args.fidelity)
    if err:
        base["skip_reason"] = f"history_{err}"
        return base, samples
    pts = [(t, p) for t, p in pts if t <= int(res.timestamp())]
    if len(pts) < 3:
        base["skip_reason"] = "too_few_points"
        return base, samples

    for h in args.horizons:
        target = int((sched - timedelta(days=h)).timestamp())
        if target >= int(res.timestamp()):
            samples.append({**base, "horizon_days": h, "sample_reason": "already_resolved"})
            continue
        if created.timestamp() > target:
            samples.append({**base, "horizon_days": h, "sample_reason": "not_yet_created"})
            continue
        lp = last_changed_price(pts, target)
        if lp is None:
            samples.append({**base, "horizon_days": h, "sample_reason": "no_trade_yet"})
            continue
        stale_h = (target - lp[0]) / 3600.0
        if stale_h > args.max_staleness_hours:
            samples.append({**base, "horizon_days": h, "sample_reason": "stale", "price": lp[1],
                            "staleness_hours": round(stale_h, 1)})
            continue
        samples.append({**base, "horizon_days": h, "sample_reason": None, "price": lp[1],
                        "staleness_hours": round(stale_h, 1)})
    return base, samples


def table(samples, h):
    rows = [s for s in samples if s.get("horizon_days") == h and s.get("sample_reason") is None]
    uns = {}
    for s in samples:
        if s.get("horizon_days") == h and s.get("sample_reason"):
            uns[s["sample_reason"]] = uns.get(s["sample_reason"], 0) + 1
    print(f"\n=== {h} days before SCHEDULED end   (n={len(rows)} usable)   unsampled: {uns} ===")
    if not rows:
        return
    print(f"  {'bucket':>12} {'n':>5} {'mean_p':>7} {'real_yes':>8} {'gap':>7} {'95% CI':>15} {'NO_yield':>9}")
    for lo, hi in BUCKETS:
        b = [s for s in rows if lo <= s["price"] < hi]
        if not b:
            continue
        n, k = len(b), sum(s["outcome_yes"] for s in b)
        mp, ry = sum(s["price"] for s in b) / n, k / n
        ci = wilson(k, n)
        noy = (1 - ry) / (1 - mp) - 1 if mp < 1 else float("nan")
        print(f"  [{lo:.2f},{hi:.2f}) {n:5d} {mp:7.3f} {ry:8.3f} {ry-mp:+7.3f} [{ci[0]:.3f},{ci[1]:.3f}] {noy:+9.2%}")
    low = [s for s in rows if 0.02 <= s["price"] < 0.10]
    if low:
        n, k = len(low), sum(s["outcome_yes"] for s in low)
        mp = sum(s["price"] for s in low) / n
        ci = wilson(k, n)
        print(f"  POOLED [0.02,0.10): n={n} yes={k} mean_p={mp:.3f} realized={k/n:.3f} "
              f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  NO_yield point {((1-k/n)/(1-mp)-1):+.2%}  "
              f"at CI upper {((1-ci[1])/(1-mp)-1):+.2%}")
        bt = {}
        for s in low:
            bt.setdefault(s.get("tag"), []).append(s)
        print("    by tag: " + ", ".join(f"{t}: n={len(v)} yes={sum(x['outcome_yes'] for x in v)}"
                                        for t, v in sorted(bt.items(), key=lambda kv: -len(kv[1]))))
        wins = [s for s in low if s["outcome_yes"] == 1]
        for s in wins:
            print(f"    YES hit: {s['price']:.3f}  ${s['volume']:,.0f}  {s['question'][:80]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--per-tag", type=int, default=20000, help="closed markets to scan per tag per direction")
    ap.add_argument("--min-volume", type=float, default=50000.0)
    ap.add_argument("--min-lifetime-days", type=int, default=21)
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 21, 14, 7])
    ap.add_argument("--fidelity", type=int, default=60)
    ap.add_argument("--max-staleness-hours", type=float, default=72.0)
    ap.add_argument("--out", default="longshot_hv_samples.csv")
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    tags = resolve_tags(args.tags)
    if not tags:
        sys.exit("no valid tags")
    print(f"tags: {tags}")

    # ---- scan (cheap) ----
    scanned, seen = [], set()
    for slug, tid in tags.items():
        for direction in ("past", "future"):
            got = scan_closed_markets(tid, args.per_tag, direction)
            new = 0
            for m in got:
                if m.get("id") in seen:
                    continue
                seen.add(m.get("id"))
                m["_tag"] = slug
                scanned.append(m)
                new += 1
            print(f"  {slug:12s} {direction:6s} fetched={len(got):6d} new={new:6d} total={len(scanned)}")
    print(f"scanned {len(scanned)} unique closed markets")

    # ---- pre-filter before any CLOB call ----
    keep, drop = [], {"sports": 0, "low_volume": 0, "short_life": 0, "no_dates": 0}
    for m in scanned:
        if m.get("gameStartTime") or m.get("sportsMarketType"):
            drop["sports"] += 1
            continue
        vol = to_float(m.get("volumeNum") or m.get("volume"))
        if vol is None or vol < args.min_volume:
            drop["low_volume"] += 1
            continue
        created = parse_iso(m.get("createdAt")) or parse_iso(m.get("startDate"))
        closed = parse_iso(m.get("closedTime")) or parse_iso(m.get("umaEndDate")) or parse_iso(m.get("endDate"))
        if not (created and closed):
            drop["no_dates"] += 1
            continue
        if (closed - created) < timedelta(days=args.min_lifetime_days):
            drop["short_life"] += 1
            continue
        keep.append(m)
    print(f"pre-filter: {drop}  ->  {len(keep)} markets with volume >= ${args.min_volume:,.0f} "
          f"and lifetime >= {args.min_lifetime_days}d")
    if not keep:
        sys.exit("nothing to evaluate")

    # ---- price + sample ----
    samples, rows = [], []
    for i, m in enumerate(keep, 1):
        try:
            base, smp = evaluate(m, args)
        except Exception as e:
            base = {"market_id": m.get("id"), "tag": m.get("_tag"), "question": m.get("question"),
                    "skip_reason": f"exception_{type(e).__name__}"}
            smp = []
        rows.append(base)
        samples.extend(smp)
        if i % 100 == 0:
            ok = sum(1 for s in samples if s.get("sample_reason") is None)
            low = sum(1 for s in samples if s.get("sample_reason") is None and 0.02 <= s["price"] < 0.10)
            print(f"  {i}/{len(keep)} priced, {ok} samples, {low} in 2-10c")
        time.sleep(args.sleep)

    skips = {}
    for r in rows:
        if r.get("skip_reason"):
            skips[r["skip_reason"]] = skips.get(r["skip_reason"], 0) + 1
    print(f"\nevaluated {sum(1 for r in rows if not r.get('skip_reason'))} of {len(rows)}   skips: {skips}")
    for h in args.horizons:
        table(samples, h)

    fields = ["market_id", "tag", "question", "neg_risk", "volume", "created", "scheduled_end",
              "resolution_time", "outcome_yes", "horizon_days", "price", "staleness_hours",
              "sample_reason", "skip_reason"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow({k: s.get(k) for k in fields})
        for r in rows:
            if r.get("skip_reason"):
                w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()