#!/usr/bin/env python3
"""
polymarket_term_structure.py  --  Event term structure on Polymarket

Idea: the same event is listed at several deadlines ("...by Sep 30?",
"...by Oct 31?", "...by Dec 31?"). Their YES prices p1 <= p2 <= p3 are points
on a survival curve. Each adjacent pair implies a "between" probability
    b_i = p_{i+1} - p_i          (market-implied P(event in (d_i, d_{i+1}]))
and a conditional hazard
    h_i = b_i / (1 - p_i)        (P(happens in interval | not yet by d_i))

Hypothesis: lottery buyers crowd the NEAREST deadline, so near-interval
hazard is overpriced relative to far intervals and the market-implied b_i
is too small. The tradeable expression is a calendar spread:
    SELL near YES at p_i, BUY far YES at p_{i+1}   (cost = p_{i+1} - p_i)
Payoff = +1 if the event lands in (d_i, d_{i+1}], else 0. So the spread is
mispriced in your favour iff  realized P(between) > b_i.
A near price ABOVE the far price (p_i > p_{i+1}) is a riskless violation.

What this script does, from resolved markets only:
  1. scans closed non-sports markets (same Gamma walk as before)
  2. normalises each question by stripping its deadline phrase -> "stem"
  3. groups markets by stem; keeps families with >= 2 distinct scheduled
     endDates (>= 3 days apart) and MONOTONE outcomes (if the near one
     resolved YES, every later one must have too; otherwise the markets
     resolve on different rules and are dropped)
  4. for each family and horizon H, samples every deadline's last-changed
     YES price at t = (earliest deadline - H days), only if every market in
     the family was open at t
  5. realizes which interval the event landed in from the outcomes
  6. reports: violations, calibration of implied "between" prob vs realized
     for adjacent pairs, split near-pair vs later pairs, and by volume tier

Setup:  pip install curl_cffi
Run:    python polymarket_term_structure.py
Deeper: python polymarket_term_structure.py --per-tag 30000

Output: family_samples CSV (one row per adjacent pair per horizon) and a
summary. Read the 'ADJACENT PAIRS' block: if realized_between > implied
with CI excluding it for the NEAR pair, the hypothesis holds.

Caveats: last-trade prices, unknown depth, and stems are matched by text so
some families are noise (the monotone-outcome check removes most of it;
inspect the CSV's stem column for the rest).
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

MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DAY = r"(?:\d{1,2}(?:st|nd|rd|th)?)"
YEAR = r"(?:20\d{2})"
DEADLINE_PATTERNS = [
    rf"\b(?:by|before|on or before|prior to|until|through|as of|on)\s+(?:the\s+)?(?:end\s+of\s+)?{MONTHS}\.?\s*{DAY}?,?\s*{YEAR}?\b",
    rf"\b(?:by|before|on or before|prior to|until|through|as of)\s+{DAY}\s+{MONTHS}\.?,?\s*{YEAR}?\b",
    rf"\b(?:by|before|until|through)\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(?:year|month|week|q[1-4]|h[12]|{YEAR}|{MONTHS})\b",
    rf"\b(?:by|before|until|through|in|during)\s+(?:q[1-4]|h[12])\s*{YEAR}?\b",
    rf"\b(?:by|before|until|through|in|during)\s+{YEAR}\b",
    rf"\b(?:in|during)\s+{MONTHS}\.?\s*{YEAR}?\b",
    rf"\b(?:by|before)\s+(?:the\s+)?(?:end\s+of\s+)?(?:this|next)\s+(?:year|month|week)\b",
    rf"\bthis\s+(?:year|month|week)\b",
    rf"\b{MONTHS}\.?\s+{DAY},?\s*{YEAR}?\b",
    rf"\b{DAY}\s+{MONTHS}\.?,?\s*{YEAR}?\b",
    rf"\b{YEAR}\b",
]
DEADLINE_RE = [re.compile(p, re.I) for p in DEADLINE_PATTERNS]


CUMULATIVE_RE = re.compile(
    rf"\b(?:by|before|on or before|prior to|until|through|no later than)\s+(?:the\s+)?(?:end\s+of\s+)?"
    rf"(?:{MONTHS}|{DAY}\s+{MONTHS}|{YEAR}|(?:the\s+)?(?:year|month|week)|q[1-4]|h[12]|this|next)", re.I)
WINDOW_RE = re.compile(rf"\b(?:in|during|on)\s+(?:{MONTHS}|{YEAR}|q[1-4]|h[12]|(?:the\s+)?week|(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b", re.I)


def deadline_kind(question):
    """'cumulative' = happens by/before a date (nested, forms a survival curve)
       'window'     = happens in/during a period (disjoint, NOT nested)
       'none'       = no deadline phrase found"""
    if CUMULATIVE_RE.search(question):
        return "cumulative"
    if WINDOW_RE.search(question):
        return "window"
    return "none"


def stem_of(question):
    q = question.lower().strip()
    q = q.replace("’", "'").replace("“", '"').replace("”", '"')
    for rx in DEADLINE_RE:
        q = rx.sub(" ", q)
    q = re.sub(r"[^\w\s$%.'-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ?.,-")
    return q


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
    price, last_change, prev = None, None, None
    for t, p in points:
        if t > ts:
            break
        if prev is None or p != prev:
            last_change = t
        prev, price = p, p
    return None if price is None else (last_change, price)


# ----------------------------------------------------------------- market record
def market_record(m):
    """Extract what we need; returns dict or (None, reason)."""
    outcomes, final, tokens = parse_list(m.get("outcomes")), parse_list(m.get("outcomePrices")), parse_list(m.get("clobTokenIds"))
    if not (outcomes and final and tokens and len(outcomes) == len(final) == len(tokens) == 2):
        return None, "not_binary"
    labels = [str(o).strip().lower() for o in outcomes]
    if "yes" not in labels:
        return None, "not_yes_no"
    yi = labels.index("yes")
    fp = to_float(final[yi])
    if fp is None:
        return None, "bad_final"
    if fp >= 0.999:
        outcome = 1
    elif fp <= 0.001:
        outcome = 0
    else:
        return None, "not_cleanly_resolved"
    res = parse_iso(m.get("closedTime")) or parse_iso(m.get("umaEndDate")) or parse_iso(m.get("endDate"))
    sched = parse_iso(m.get("endDate"))
    created = parse_iso(m.get("createdAt")) or parse_iso(m.get("startDate"))
    if not (res and sched and created):
        return None, "missing_dates"
    if res > datetime.now(timezone.utc):
        return None, "resolution_in_future"
    kind = deadline_kind(m.get("question", ""))
    if kind != "cumulative":
        return None, f"deadline_{kind}"
    return {
        "id": m.get("id"), "tag": m.get("_tag"), "question": m.get("question", ""),
        "stem": stem_of(m.get("question", "")), "yes_token": tokens[yi], "outcome": outcome,
        "res": res, "sched": sched, "created": created,
        "volume": to_float(m.get("volumeNum") or m.get("volume")) or 0.0,
        "neg_risk": m.get("negRisk"), "pts": None,
    }, None


# ----------------------------------------------------------------- analysis
def build_families(records, min_gap_days):
    fams = defaultdict(list)
    for r in records:
        if len(r["stem"]) >= 12:
            fams[r["stem"]].append(r)
    out = {}
    for stem, ms in fams.items():
        ms = sorted(ms, key=lambda r: r["sched"])
        # dedupe deadlines that are basically the same day (keep higher volume)
        kept = []
        for r in ms:
            if kept and (r["sched"] - kept[-1]["sched"]) < timedelta(days=min_gap_days):
                if r["volume"] > kept[-1]["volume"]:
                    kept[-1] = r
                continue
            kept.append(r)
        if len(kept) >= 2:
            out[stem] = kept
    return out


def monotone(fam):
    """Once YES, stays YES for later deadlines."""
    seen_yes = False
    for r in fam:
        if seen_yes and r["outcome"] == 0:
            return False
        if r["outcome"] == 1:
            seen_yes = True
    return True


def realized_interval(fam):
    """Index i of first deadline whose market resolved YES; len(fam) if never."""
    for i, r in enumerate(fam):
        if r["outcome"] == 1:
            return i
    return len(fam)


def analyse(families, args):
    """Sample each ADJACENT PAIR independently at t = d_i - H. Only the two
    markets of the pair must be created before t and still open at t."""
    pair_rows = []
    fam_stats = {"families": len(families), "non_monotone": 0, "pair_samples": 0, "pairs_unsampled": 0}
    for stem, fam in families.items():
        if not monotone(fam):
            fam_stats["non_monotone"] += 1
            continue
        r_int = realized_interval(fam)
        for i in range(len(fam) - 1):
            near, far = fam[i], fam[i + 1]
            for H in args.horizons:
                t = int((near["sched"] - timedelta(days=H)).timestamp())
                prices = []
                for r in (near, far):
                    if r["created"].timestamp() > t or r["res"].timestamp() <= t or r["pts"] is None:
                        prices = None
                        break
                    lp = last_changed_price(r["pts"], t)
                    if lp is None or (t - lp[0]) / 3600.0 > args.max_staleness_hours:
                        prices = None
                        break
                    prices.append(lp[1])
                if prices is None:
                    fam_stats["pairs_unsampled"] += 1
                    continue
                fam_stats["pair_samples"] += 1
                p_near, p_far = prices
                implied_between = p_far - p_near
                pair_rows.append({
                    "stem": stem, "tag": near["tag"], "horizon_days": H, "pair_index": i,
                    "n_deadlines": len(fam),
                    "near_q": near["question"], "far_q": far["question"],
                    "near_deadline": near["sched"].date().isoformat(),
                    "far_deadline": far["sched"].date().isoformat(),
                    "gap_days": (far["sched"] - near["sched"]).days,
                    "p_near": p_near, "p_far": p_far,
                    "implied_between": round(implied_between, 4),
                    "implied_hazard_next": round(implied_between / max(1e-9, 1 - p_near), 4),
                    "realized_between": 1 if r_int == i + 1 else 0,
                    "realized_by_near": 1 if r_int <= i else 0,
                    "realized_by_far": 1 if r_int <= i + 1 else 0,
                    "violation": 1 if p_near > p_far + 1e-9 else 0,
                    "min_volume": min(near["volume"], far["volume"]),
                    "spread_pnl": (1 if r_int == i + 1 else 0) - implied_between,
                })
    return pair_rows, fam_stats


def summarise(pair_rows, args):
    print(f"\n================ ADJACENT PAIRS ================")
    for H in args.horizons:
        rows = [r for r in pair_rows if r["horizon_days"] == H]
        if not rows:
            print(f"\n-- horizon {H}d: no samples")
            continue
        print(f"\n-- horizon {H}d before FIRST deadline: {len(rows)} pairs from {len({r['stem'] for r in rows})} families")
        viol = [r for r in rows if r["violation"]]
        print(f"   violations (near priced above far): {len(viol)}  "
              f"mean size {sum(r['p_near']-r['p_far'] for r in viol)/len(viol):+.3f}" if viol else
              "   violations (near priced above far): 0")

        for r in sorted(viol, key=lambda r: r["p_far"] - r["p_near"])[:8]:
            print(f"      {r['p_near']:.3f} > {r['p_far']:.3f}  ${r['min_volume']:,.0f}  "
                  f"{r['near_q'][:60]}  ||  {r['far_q'][:60]}")

        def block(label, rs):
            if not rs:
                return
            n = len(rs)
            k = sum(r["realized_between"] for r in rs)
            imp = sum(r["implied_between"] for r in rs) / n
            ci = wilson(k, n)
            pnl = sum(r["spread_pnl"] for r in rs) / n
            # per-dollar-risked: long spread costs implied_between (when positive)
            cost = sum(max(r["implied_between"], 0) for r in rs) / n
            roi = pnl / cost if cost > 0 else float("nan")
            print(f"   {label:28s} n={n:4d}  implied_between={imp:.3f}  realized={k/n:.3f} "
                  f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  gap={k/n-imp:+.3f}  "
                  f"long-spread EV/pair={pnl:+.3f}  ROI on cost={roi:+.1%}")

        block("ALL pairs", rows)
        block("NEAR pair (i=0)", [r for r in rows if r["pair_index"] == 0])
        block("LATER pairs (i>=1)", [r for r in rows if r["pair_index"] >= 1])
        # near-deadline calibration: is p_near itself overpriced?
        near = [r for r in rows if r["pair_index"] == 0]
        if near:
            n = len(near)
            k = sum(r["realized_by_near"] for r in near)
            mp = sum(r["p_near"] for r in near) / n
            ci = wilson(k, n)
            print(f"   {'NEAR deadline itself':28s} n={n:4d}  price={mp:.3f}  realized_by_near={k/n:.3f} "
                  f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  gap={k/n-mp:+.3f}")
        for name, lo, hi in [("<2k", 0, 2000), ("2-10k", 2000, 10000), ("10-50k", 10000, 50000), (">50k", 50000, 1e15)]:
            block(f"  min_vol {name}", [r for r in rows if lo <= r["min_volume"] < hi])
        # low-price subset: the longshot term structure
        block("  p_near < 0.15 (longshots)", [r for r in rows if r["p_near"] < 0.15])
    print("\nlong-spread = sell near YES, buy far YES; pays 1 if event lands between deadlines.")
    print("Hypothesis holds if realized_between > implied_between (positive gap, CI above implied),")
    print("especially for the NEAR pair and the longshot subset.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--per-tag", type=int, default=15000)
    ap.add_argument("--min-volume", type=float, default=1000.0, help="per-market floor before pricing")
    ap.add_argument("--min-gap-days", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 14, 7, 3])
    ap.add_argument("--fidelity", type=int, default=60)
    ap.add_argument("--max-staleness-hours", type=float, default=72.0)
    ap.add_argument("--out", default="term_structure_pairs.csv")
    ap.add_argument("--families-out", default="term_structure_families.csv")
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    tags = resolve_tags(args.tags)
    if not tags:
        sys.exit("no valid tags")
    print(f"tags: {tags}")

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

    records, drop = [], defaultdict(int)
    for m in scanned:
        if m.get("gameStartTime") or m.get("sportsMarketType"):
            drop["sports"] += 1
            continue
        rec, why = market_record(m)
        if rec is None:
            drop[why] += 1
            continue
        if rec["volume"] < args.min_volume:
            drop["low_volume"] += 1
            continue
        records.append(rec)
    print(f"records: {len(records)}   dropped: {dict(drop)}")

    families = build_families(records, args.min_gap_days)
    n_mono = sum(1 for f in families.values() if monotone(f))
    print(f"families with >=2 deadlines: {len(families)}   monotone outcomes: {n_mono}   "
          f"markets in families: {sum(len(f) for f in families.values())}")
    if not families:
        sys.exit("no deadline families found; check stems in --families-out")

    # write families for inspection before pricing
    with open(args.families_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "n_deadlines", "monotone", "deadline", "question", "outcome", "volume", "market_id"])
        for stem, fam in families.items():
            for r in fam:
                w.writerow([stem, len(fam), monotone(fam), r["sched"].date().isoformat(), r["question"],
                            r["outcome"], r["volume"], r["id"]])
    print(f"wrote {args.families_out} (inspect stems)")

    # price only markets in monotone families
    todo = [r for fam in families.values() if monotone(fam) for r in fam]
    print(f"pricing {len(todo)} markets ...")
    fails = 0
    for i, r in enumerate(todo, 1):
        pts, err = fetch_history(r["yes_token"], args.fidelity)
        if err or not pts:
            fails += 1
        else:
            r["pts"] = [(t, p) for t, p in pts if t <= int(r["res"].timestamp())]
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} priced ({fails} failed)")
        time.sleep(args.sleep)

    pair_rows, fam_stats = analyse(families, args)
    print(f"\nfamilies={fam_stats['families']} non_monotone={fam_stats['non_monotone']} "
          f"pair samples={fam_stats['pair_samples']} unsampled={fam_stats['pairs_unsampled']}")
    summarise(pair_rows, args)

    if pair_rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()