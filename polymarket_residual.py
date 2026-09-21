#!/usr/bin/env python3
"""
polymarket_residual.py  --  the "Other" lottery test

Identity: in a mutually-exclusive, exhaustive multi-outcome event,
    P(Other) = 1 - sum P(named)
If named longshots are collectively overpriced (we measured they are), the
residual contract must be underpriced by the same amount. This script checks
that on every resolved Polymarket multi-outcome event that has an
Other / someone-else / none-of-the-above contract.

For each event and horizon H (days before the event's scheduled end):
    sum_named   = sum of last-changed YES prices of all named contracts at t
    other_price = last-changed YES price of the Other contract at t
    implied_residual = 1 - sum_named        (what Other SHOULD be, if named were right)
    realized_other   = 1 if Other resolved YES

Output: calibration of Other by sum_named bucket, and by other_price bucket:
    n, mean other price, realized rate, 95% CI, EV per $1 spent on Other
    (= realized/price - 1). Positive EV with CI lower bound > price = lottery.

Setup:  pip install curl_cffi
Run:    python polymarket_residual.py
Deeper: python polymarket_residual.py --per-tag 20000

Caveats stated once: last-trade prices not asks; candidates added after t
are detected and the sample dropped (flag --keep-added to keep them);
'Other' definitions vary by event -- read the CSV's other_question column.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
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

OTHER_RE = re.compile(
    r"\b(?:other|another|someone else|somebody else|anyone else|none of the above|no one|nobody|"
    r"the field|field|any other|other person|other candidate|other team|other company|other country|"
    r"not listed|unlisted|else)\b", re.I)


def is_other(m):
    title = (m.get("groupItemTitle") or "").strip()
    q = m.get("question") or ""
    if title and OTHER_RE.search(title):
        return True
    if not title and OTHER_RE.search(q):
        return True
    return False


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


# ----------------------------------------------------------------- gamma events
def resolve_tags(slugs):
    out = {}
    for slug in slugs:
        t = get_json(f"{GAMMA}/tags/slug/{slug}")
        if isinstance(t, dict) and "id" in t:
            out[slug] = t["id"]
        else:
            print(f"  tag '{slug}' not found -- skipping")
        time.sleep(0.1)
    return out


def scan_closed_events(tag_id, max_events, direction, page_size=100, offset_cap=1000):
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_max = now_s if direction == "past" else None
    end_min = now_s if direction == "future" else None
    events, seen, offset = [], set(), 0
    while len(events) < max_events:
        params = {"closed": "true", "limit": page_size, "offset": offset, "tag_id": tag_id,
                  "order": "endDate", "ascending": "false" if direction == "past" else "true"}
        if end_max:
            params["end_date_max"] = end_max
        if end_min:
            params["end_date_min"] = end_min
        page = get_json(f"{GAMMA}/events", params=params)
        if isinstance(page, dict):
            if offset > 0:
                page = []
            else:
                print(f"    gamma error {page.get('_http_error')} at offset 0", file=sys.stderr)
                break
        for e in page:
            if e.get("id") not in seen:
                seen.add(e.get("id"))
                events.append(e)
        offset += len(page)
        if (not page) or len(page) < page_size or offset >= offset_cap:
            eds = [parse_iso(e.get("endDate")) for e in events]
            eds = [d for d in eds if d]
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
    return events[:max_events]


# ----------------------------------------------------------------- clob
def fetch_history(token_id, fidelity_min):
    data = get_json(f"{CLOB}/prices-history",
                    params={"market": token_id, "interval": "max", "fidelity": fidelity_min})
    if isinstance(data, dict) and "_http_error" in data:
        return None
    hist = data.get("history") if isinstance(data, dict) else None
    if hist is None:
        return None
    pts = []
    for h in hist:
        try:
            pts.append((int(h["t"]), float(h["p"])))
        except (KeyError, TypeError, ValueError):
            pass
    pts.sort()
    return pts


def last_changed_price(points, ts):
    price, last_change, prev = None, None, None
    for t, p in points:
        if t > ts:
            break
        if prev is None or p != prev:
            last_change = t
        prev, price = p, p
    return None if price is None else (last_change, price)


# ----------------------------------------------------------------- event record
def event_record(e, args):
    """Parse a closed multi-outcome event into named contracts + the Other contract."""
    mkts = e.get("markets") or []
    if len(mkts) < args.min_names + 1:
        return None, "too_few_markets"
    if not e.get("negRisk") and not args.allow_non_negrisk:
        return None, "not_negrisk"
    ev_end = parse_iso(e.get("endDate"))
    ev_created = parse_iso(e.get("createdAt")) or parse_iso(e.get("startDate"))
    if not ev_end:
        return None, "no_endDate"
    recs, others = [], []
    for m in mkts:
        outcomes, final, tokens = parse_list(m.get("outcomes")), parse_list(m.get("outcomePrices")), parse_list(m.get("clobTokenIds"))
        if not (outcomes and final and tokens and len(outcomes) == len(final) == len(tokens) == 2):
            continue
        labels = [str(o).strip().lower() for o in outcomes]
        yi = labels.index("yes") if "yes" in labels else 0
        fp = to_float(final[yi])
        if fp is None:
            continue
        if fp >= 0.999:
            outcome = 1
        elif fp <= 0.001:
            outcome = 0
        else:
            return None, "market_not_cleanly_resolved"
        rec = {
            "id": m.get("id"), "title": (m.get("groupItemTitle") or m.get("question") or "")[:80],
            "question": m.get("question") or "", "tok": tokens[yi], "outcome": outcome,
            "created": parse_iso(m.get("createdAt")) or ev_created,
            "res": parse_iso(m.get("closedTime")) or parse_iso(m.get("umaEndDate")) or parse_iso(m.get("endDate")) or ev_end,
            "volume": to_float(m.get("volumeNum") or m.get("volume")) or 0.0, "pts": None,
        }
        if rec["created"] is None:
            return None, "market_no_created"
        (others if is_other(m) else recs).append(rec)
    if len(others) != 1:
        return None, f"other_count_{len(others)}"
    if len(recs) < args.min_names:
        return None, "too_few_named"
    winners = sum(r["outcome"] for r in recs) + others[0]["outcome"]
    if winners != 1:
        return None, f"winners_{winners}"
    return {"id": e.get("id"), "title": e.get("title") or "", "tag": e.get("_tag"), "end": ev_end,
            "named": recs, "other": others[0], "volume": to_float(e.get("volume")) or 0.0}, None


def sample_event(ev, args):
    rows = []
    o = ev["other"]
    for H in args.horizons:
        t = int((ev["end"] - timedelta(days=H)).timestamp())
        # every contract must exist, be open, and have a fresh-enough price at t
        prices, added_later, ok, why = [], 0, True, ""
        for r in ev["named"] + [o]:
            if r["created"].timestamp() > t:
                added_later += 1
                continue
            if r["res"].timestamp() <= t:
                ok, why = False, "resolved_before_t"
                break
            if r["pts"] is None:
                # never traded at all: a dead candidate is worth ~0; Other must have a price
                if r is o:
                    ok, why = False, "other_no_history"
                    break
                prices.append((r, 0.0))
                continue
            lp = last_changed_price(r["pts"], t)
            if lp is None:
                if r is o:
                    ok, why = False, "other_no_trade_yet"
                    break
                prices.append((r, 0.0))
                continue
            stale_h = (t - lp[0]) / 3600.0
            if stale_h > args.max_staleness_hours and (r is o or lp[1] > 0.05):
                ok, why = False, ("other_stale" if r is o else "named_stale")
                break
            prices.append((r, lp[1]))
        if not ok:
            rows.append({"event_id": ev["id"], "title": ev["title"], "tag": ev["tag"], "horizon_days": H,
                         "sample_reason": why or "unsampled"})
            continue
        if added_later and args.drop_added:
            rows.append({"event_id": ev["id"], "title": ev["title"], "tag": ev["tag"], "horizon_days": H,
                         "sample_reason": f"candidates_added_later_{added_later}"})
            continue
        other_hit = [p for r, p in prices if r is o]
        if not other_hit:
            rows.append({"event_id": ev["id"], "title": ev["title"], "tag": ev["tag"], "horizon_days": H,
                         "sample_reason": "other_not_yet_created"})
            continue
        other_p = other_hit[0]
        named_p = [p for r, p in prices if r is not o]
        rows.append({
            "event_id": ev["id"], "title": ev["title"], "tag": ev["tag"], "horizon_days": H,
            "sample_reason": None, "n_named": len(named_p), "added_later": added_later,
            "sum_named": round(sum(named_p), 4), "sum_all": round(sum(named_p) + other_p, 4),
            "implied_residual": round(1 - sum(named_p), 4), "other_price": other_p,
            "other_question": o["question"][:90], "other_volume": round(o["volume"]),
            "event_volume": round(ev["volume"]), "realized_other": o["outcome"],
            "winner": next((r["title"] for r in ev["named"] if r["outcome"] == 1), "OTHER"),
        })
    return rows


# ----------------------------------------------------------------- report
def block(label, rs):
    if not rs:
        return
    n = len(rs)
    k = sum(r["realized_other"] for r in rs)
    mp = sum(r["other_price"] for r in rs) / n
    ms = sum(r["sum_named"] for r in rs) / n
    ci = wilson(k, n)
    ev = (k / n) / mp - 1 if mp > 0 else float("nan")
    ev_lo = ci[0] / mp - 1 if mp > 0 else float("nan")
    print(f"  {label:26s} n={n:4d}  sum_named={ms:.3f}  other_price={mp:.3f}  realized={k/n:.3f} "
          f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  EV/$1 point {ev:+.0%}  at CI lower {ev_lo:+.0%}")


def report(rows, args):
    for H in args.horizons:
        rs = [r for r in rows if r["horizon_days"] == H and r.get("sample_reason") is None]
        uns = defaultdict(int)
        for r in rows:
            if r["horizon_days"] == H and r.get("sample_reason"):
                uns[r["sample_reason"].split("_")[0]] += 1
        print(f"\n=== {H} days before event end: n={len(rs)} events   unsampled: {dict(uns)} ===")
        if not rs:
            continue
        block("ALL", rs)
        print("  -- by sum of named prices:")
        for lo, hi in [(0, 0.90), (0.90, 0.97), (0.97, 1.00), (1.00, 1.03), (1.03, 1.08), (1.08, 1.15), (1.15, 9)]:
            block(f"    sum [{lo:.2f},{hi:.2f})", [r for r in rs if lo <= r["sum_named"] < hi])
        print("  -- by Other price:")
        for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]:
            block(f"    other [{lo:.2f},{hi:.2f})", [r for r in rs if lo <= r["other_price"] < hi])
        print("  -- over-summed AND cheap Other (the lottery):")
        block("    sum>=1.03 & other<0.05", [r for r in rs if r["sum_named"] >= 1.03 and r["other_price"] < 0.05])
        block("    sum>=1.00 & other<0.10", [r for r in rs if r["sum_named"] >= 1.00 and r["other_price"] < 0.10])
        hits = [r for r in rs if r["realized_other"] == 1]
        if hits:
            print("  Other WON in:")
            for r in sorted(hits, key=lambda r: r["other_price"])[:15]:
                print(f"    other={r['other_price']:.3f} sum_named={r['sum_named']:.3f} n={r['n_named']}  {r['title'][:70]}")
    print("\nEV/$1 = realized/price - 1. The lottery exists if the 'sum>=1.03 & other<0.05' line has EV > 0 at CI lower.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--per-tag", type=int, default=8000)
    ap.add_argument("--min-names", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 14, 7, 3])
    ap.add_argument("--fidelity", type=int, default=60)
    ap.add_argument("--max-staleness-hours", type=float, default=96.0)
    ap.add_argument("--drop-added", action="store_true", help="drop samples where candidates were added after t (default: keep; it is real risk)")
    ap.add_argument("--allow-non-negrisk", action="store_true")
    ap.add_argument("--out", default="residual_samples.csv")
    ap.add_argument("--sleep", type=float, default=0.08)
    args = ap.parse_args()

    tags = resolve_tags(args.tags)
    if not tags:
        sys.exit("no valid tags")
    print(f"tags: {tags}")

    scanned, seen = [], set()
    for slug, tid in tags.items():
        for direction in ("past", "future"):
            got = scan_closed_events(tid, args.per_tag, direction)
            new = 0
            for e in got:
                if e.get("id") in seen:
                    continue
                seen.add(e.get("id"))
                e["_tag"] = slug
                scanned.append(e)
                new += 1
            print(f"  {slug:12s} {direction:6s} fetched={len(got):6d} new={new:6d} total={len(scanned)}")
    print(f"scanned {len(scanned)} unique closed events")

    events, drop = [], defaultdict(int)
    for e in scanned:
        rec, why = event_record(e, args)
        if rec is None:
            drop[why] += 1
        else:
            events.append(rec)
    print(f"multi-outcome events with exactly one Other contract: {len(events)}   dropped: {dict(drop)}")
    if not events:
        sys.exit("nothing to test")

    todo = [r for ev in events for r in ev["named"] + [ev["other"]]]
    cache_path = f"history_cache_f{args.fidelity}.json"
    cache = {}
    try:
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"loaded {len(cache)} cached histories from {cache_path}")
    except (OSError, ValueError):
        pass
    print(f"pricing {len(todo)} contracts ({sum(1 for r in todo if r['tok'] in cache)} cached) ...")
    fetched = 0
    for i, r in enumerate(todo, 1):
        if r["tok"] in cache:
            pts = cache[r["tok"]]
        else:
            pts = fetch_history(r["tok"], args.fidelity) or []
            cache[r["tok"]] = pts
            fetched += 1
            time.sleep(args.sleep)
            if fetched % 500 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
        if pts:
            r["pts"] = [(t, p) for t, p in pts if t <= int(r["res"].timestamp())]
        if i % 2000 == 0:
            print(f"  {i}/{len(todo)}")
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"cache saved: {len(cache)} histories -> {cache_path}")

    rows = []
    for ev in events:
        rows.extend(sample_event(ev, args))
    report(rows, args)

    fields = ["event_id", "title", "tag", "horizon_days", "sample_reason", "n_named", "added_later", "sum_named",
              "sum_all", "implied_residual", "other_price", "other_question", "other_volume", "event_volume",
              "realized_other", "winner"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()