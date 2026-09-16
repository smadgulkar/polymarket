#!/usr/bin/env python3
"""
polymarket_term_live.py  --  live nested-deadline scanner

Every run: pull OPEN markets under non-sports tags, keep cumulative-deadline
questions ("... by Sep 30?", "... before 2027?"), group by stem, and for each
adjacent deadline pair read the REAL order books of both YES tokens.

Reports, per pair (near deadline d_i, far deadline d_{i+1}):
  VIOLATION   near YES bid  >  far YES ask         -> riskless: buy far YES at ask,
                                                     sell near YES at bid (i.e. buy near NO)
  SPREAD      long calendar spread cost = far_ask - near_bid
              pays 1 if the event lands in (d_i, d_{i+1}], 0 otherwise
              implied P(between) = cost; implied hazard/day for that interval
              vs implied hazard/day for the first interval  (shape inversion =
              near interval priced hotter than the next one)
  depth       size available at those prices (USDC), so you know what fills

Everything is appended to term_live_log.csv with a timestamp so a month of
runs becomes a paper track record. Nothing is traded.

Setup:  pip install curl_cffi
Run:    python polymarket_term_live.py
        python polymarket_term_live.py --min-depth 50 --tags geopolitics politics tech
Loop:   python polymarket_term_live.py --every 180      (minutes; Ctrl-C to stop)
"""

import argparse
import csv
import json
import math
import os
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
    rf"(?:{MONTHS}\b|{DAY}\s+{MONTHS}\b|{YEAR}\b|(?:the\s+)?(?:year|month|week)\b|q[1-4]\b|h[12]\b|this\b|next\b)", re.I)
# Families whose YES probability FALLS with a later deadline: negations and
# persistence-of-state ("continues through", "remains", "still"). These are
# evaluated on the NO side, which turns them into ordinary rising curves.
DECREASING_RE = re.compile(
    r"\b(?:not|never|without|no\s+(?!longer\b)\w+|continues?|remains?|stays?|still|maintains?|survives?|persists?)\b", re.I)


def polarity(q):
    """'no longer X' is an event (rising). 'no next X', 'not X', 'continues through' fall."""
    return "decreasing" if DECREASING_RE.search(q) else "increasing"


def is_cumulative(q):
    return bool(CUMULATIVE_RE.search(q))


def stem_of(question):
    q = question.lower().strip().replace("’", "'")
    for rx in DEADLINE_RE:
        q = rx.sub(" ", q)
    q = re.sub(r"[^\w\s$%.'-]", " ", q)
    return re.sub(r"\s+", " ", q).strip(" ?.,-")


# ----------------------------------------------------------------- http
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


# ----------------------------------------------------------------- gamma (open markets)
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


def scan_open_markets(tag_id, max_markets=5000, page_size=100, offset_cap=1000):
    """Open markets ordered by endDate ascending, paged in end_date_min windows."""
    end_min = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    markets, seen, offset = [], set(), 0
    while len(markets) < max_markets:
        params = {"closed": "false", "active": "true", "limit": page_size, "offset": offset,
                  "tag_id": tag_id, "order": "endDate", "ascending": "true", "end_date_min": end_min}
        page = get_json(f"{GAMMA}/markets", params=params)
        if isinstance(page, dict):
            if offset > 0:
                page = []
            else:
                print(f"    gamma error {page.get('_http_error')} at offset 0", file=sys.stderr)
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
            new = (max(eds) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if new == end_min:
                break
            end_min, offset = new, 0
        time.sleep(0.1)
    return markets[:max_markets]


# ----------------------------------------------------------------- clob (order book)
def fetch_book(token_id):
    """Returns (best_bid, bid_size, best_ask, ask_size, bid_depth_5c, ask_depth_5c) in price/USDC.
    depth_5c = size available within 5c of the best price."""
    data = get_json(f"{CLOB}/book", params={"token_id": token_id})
    if not isinstance(data, dict) or "_http_error" in data:
        return None
    bids = [(to_float(b.get("price")), to_float(b.get("size"))) for b in data.get("bids", [])]
    asks = [(to_float(a.get("price")), to_float(a.get("size"))) for a in data.get("asks", [])]
    bids = [(p, s) for p, s in bids if p is not None and s is not None]
    asks = [(p, s) for p, s in asks if p is not None and s is not None]
    if not bids or not asks:
        return {"bid": None, "bid_sz": 0.0, "ask": None, "ask_sz": 0.0, "bid_d5": 0.0, "ask_d5": 0.0,
                "n_bids": len(bids), "n_asks": len(asks)}
    bids.sort(key=lambda x: -x[0])
    asks.sort(key=lambda x: x[0])
    bb, ba = bids[0], asks[0]
    bid_d5 = sum(p * s for p, s in bids if p >= bb[0] - 0.05)
    ask_d5 = sum(p * s for p, s in asks if p <= ba[0] + 0.05)
    return {"bid": bb[0], "bid_sz": round(bb[0] * bb[1], 2), "ask": ba[0], "ask_sz": round(ba[0] * ba[1], 2),
            "bid_d5": round(bid_d5, 2), "ask_d5": round(ask_d5, 2), "n_bids": len(bids), "n_asks": len(asks)}


# ----------------------------------------------------------------- core
def yes_token(m):
    outcomes, tokens = parse_list(m.get("outcomes")), parse_list(m.get("clobTokenIds"))
    if not (outcomes and tokens and len(outcomes) == len(tokens) == 2):
        return None
    labels = [str(o).strip().lower() for o in outcomes]
    if "yes" not in labels:
        return None
    return tokens[labels.index("yes")]


def build_pairs(markets, min_gap_days):
    fams = defaultdict(list)
    for m in markets:
        q = m.get("question") or ""
        if m.get("gameStartTime") or m.get("sportsMarketType") or not is_cumulative(q):
            continue
        tok = yes_token(m)
        ed = parse_iso(m.get("endDate"))
        if not tok or not ed:
            continue
        st = stem_of(q)
        if len(st) < 12:
            continue
        st = st + " ||" + polarity(q)  # never mix polarities in one family
        fams[st].append({"id": m.get("id"), "q": q, "slug": m.get("slug"), "tok": tok, "end": ed, "pol": polarity(q),
                         "vol": to_float(m.get("volumeNum") or m.get("volume")) or 0.0, "tag": m.get("_tag")})
    pairs = []
    for st, ms in fams.items():
        ms = sorted(ms, key=lambda r: r["end"])
        kept = []
        for r in ms:
            if kept and (r["end"] - kept[-1]["end"]) < timedelta(days=min_gap_days):
                if r["vol"] > kept[-1]["vol"]:
                    kept[-1] = r
                continue
            kept.append(r)
        for i in range(len(kept) - 1):
            pairs.append((st, i, kept[i], kept[i + 1], len(kept)))
    return pairs


def no_side(b):
    """Convert a YES book to the NO book: NO bid = 1 - YES ask, NO ask = 1 - YES bid."""
    if not b:
        return b
    return {"bid": (1 - b["ask"]) if b["ask"] is not None else None, "bid_sz": b["ask_sz"],
            "ask": (1 - b["bid"]) if b["bid"] is not None else None, "ask_sz": b["bid_sz"],
            "bid_d5": b["ask_d5"], "ask_d5": b["bid_d5"], "n_bids": b["n_asks"], "n_asks": b["n_bids"]}


def evaluate_pair(st, i, near, far, n_dead, books, now):
    bn, bf = books.get(near["tok"]), books.get(far["tok"])
    if not bn or not bf:
        return None
    pol = near.get("pol", "increasing")
    if pol == "decreasing":
        bn, bf = no_side(bn), no_side(bf)
    st = st.split(" ||")[0]
    days_near = max(0.01, (near["end"] - now).total_seconds() / 86400)
    gap_days = max(0.01, (far["end"] - near["end"]).total_seconds() / 86400)
    row = {
        "ts": now.strftime("%Y-%m-%d %H:%M"), "stem": st, "polarity": pol, "side": "NO" if pol == "decreasing" else "YES",
        "pair_index": i, "n_deadlines": n_dead, "tag": near["tag"],
        "near_q": near["q"], "far_q": far["q"], "near_slug": near["slug"], "far_slug": far["slug"],
        "near_end": near["end"].date().isoformat(), "far_end": far["end"].date().isoformat(),
        "days_to_near": round(days_near, 1), "gap_days": round(gap_days, 1),
        "near_bid": bn["bid"], "near_ask": bn["ask"], "far_bid": bf["bid"], "far_ask": bf["ask"],
        "near_bid_$": bn["bid_sz"], "far_ask_$": bf["ask_sz"], "near_bid_d5_$": bn["bid_d5"], "far_ask_d5_$": bf["ask_d5"],
        "near_vol": round(near["vol"]), "far_vol": round(far["vol"]),
        "violation": 0, "violation_edge": None, "violation_fill_$": None,
        "spread_cost": None, "implied_between": None, "hazard_near_per_day": None, "hazard_next_per_day": None,
        "shape_inverted": None, "near_tok": near["tok"], "far_tok": far["tok"],
    }
    if bn["bid"] is None or bf["ask"] is None:
        return row
    # riskless violation: sell near YES at bid, buy far YES at ask
    if bn["bid"] > bf["ask"] + 1e-9:
        row["violation"] = 1
        row["violation_edge"] = round(bn["bid"] - bf["ask"], 4)
        row["violation_fill_$"] = round(min(bn["bid_sz"], bf["ask_sz"]), 2)
    # calendar spread at the touch
    cost = bf["ask"] - bn["bid"]
    row["spread_cost"] = round(cost, 4)
    row["implied_between"] = round(cost, 4)
    mid_n = (bn["bid"] + bn["ask"]) / 2 if bn["ask"] is not None else bn["bid"]
    mid_f = (bf["bid"] + bf["ask"]) / 2 if bf["bid"] is not None else bf["ask"]
    # hazard per day: first interval uses mid_n over days_to_near; next uses conditional prob over gap_days
    h_near = mid_n / days_near
    h_next = max(0.0, (mid_f - mid_n)) / max(1e-9, 1 - mid_n) / gap_days
    row["hazard_near_per_day"] = round(h_near, 5)
    row["hazard_next_per_day"] = round(h_next, 5)
    row["shape_inverted"] = 1 if h_near > 1.5 * h_next else 0
    return row


def print_report(rows, min_depth):
    viol = [r for r in rows if r["violation"]]
    print(f"\n=== {rows[0]['ts'] if rows else ''} UTC   pairs={len(rows)}   violations={len(viol)} ===")
    if viol:
        print("\nVIOLATIONS (near bid > far ask on the rising side; side shown): riskless if both legs fill")
        for r in sorted(viol, key=lambda r: -(r["violation_edge"] or 0)):
            flag = "" if (r["violation_fill_$"] or 0) >= min_depth else "  (thin)"
            print(f"  +{r['violation_edge']:.3f}  fill ${r['violation_fill_$']:>8,.0f}{flag}")
            print(f"     sell {r['side']} {r['near_q'][:66]}  @bid {r['near_bid']:.3f}  (${r['near_bid_$']:,.0f})")
            print(f"     buy  {r['side']} {r['far_q'][:66]}  @ask {r['far_ask']:.3f}  (${r['far_ask_$']:,.0f})")
    live = [r for r in rows if r["spread_cost"] is not None]
    inv = [r for r in live if r["shape_inverted"]]
    print(f"\nSHAPE INVERSIONS (near-interval hazard > 1.5x next interval): {len(inv)}")
    for r in sorted(inv, key=lambda r: -(r["hazard_near_per_day"] / max(1e-9, r["hazard_next_per_day"])))[:25]:
        depth = min(r["near_bid_d5_$"], r["far_ask_d5_$"])
        flag = "" if depth >= min_depth else "  (thin)"
        print(f"  [{r['side']}] {r['stem'][:52]:52s} d_near={r['near_end']} d_far={r['far_end']}  "
              f"near {r['near_bid']:.3f}/{r['near_ask']:.3f}  far {r['far_bid'] if r['far_bid'] is not None else float('nan'):.3f}/{r['far_ask']:.3f}  "
              f"spread cost {r['spread_cost']:+.3f}  h_near/day {r['hazard_near_per_day']:.4f} vs h_next/day {r['hazard_next_per_day']:.4f}  "
              f"depth ${depth:,.0f}{flag}")
    print(f"\nALL PAIRS ({len(live)} with two-sided books), cheapest spreads first:")
    for r in sorted(live, key=lambda r: r["spread_cost"])[:40]:
        print(f"  cost {r['spread_cost']:+.3f}  {r['near_end']}->{r['far_end']} ({r['gap_days']:.0f}d)  "
              f"near {r['near_bid']:.3f}/{r['near_ask']:.3f}  far {r['far_bid'] if r['far_bid'] is not None else float('nan'):.3f}/{r['far_ask']:.3f}  "
              f"depth ${min(r['near_bid_d5_$'], r['far_ask_d5_$']):,.0f}  [{r['side']}] {r['stem'][:50]}")


def run_once(args, tags):
    now = datetime.now(timezone.utc)
    scanned, seen = [], set()
    for slug, tid in tags.items():
        got = scan_open_markets(tid, args.per_tag)
        for m in got:
            if m.get("id") not in seen:
                seen.add(m.get("id"))
                m["_tag"] = slug
                scanned.append(m)
    pairs = build_pairs(scanned, args.min_gap_days)
    print(f"open markets: {len(scanned)}   nested-deadline pairs: {len(pairs)}")
    toks = {}
    for st, i, near, far, n in pairs:
        toks[near["tok"]] = 1
        toks[far["tok"]] = 1
    books = {}
    for k, tok in enumerate(toks, 1):
        b = fetch_book(tok)
        if b:
            books[tok] = b
        if k % 50 == 0:
            print(f"  {k}/{len(toks)} books")
        time.sleep(args.sleep)
    rows = []
    for st, i, near, far, n in pairs:
        r = evaluate_pair(st, i, near, far, n, books, now)
        if r:
            rows.append(r)
    if not rows:
        print("no pairs with books")
        return
    print_report(rows, args.min_depth)
    new_file = not os.path.exists(args.log)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended {len(rows)} rows to {args.log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--per-tag", type=int, default=5000)
    ap.add_argument("--min-gap-days", type=int, default=3)
    ap.add_argument("--min-depth", type=float, default=100.0, help="USDC; pairs below are flagged (thin)")
    ap.add_argument("--log", default="term_live_log.csv")
    ap.add_argument("--sleep", type=float, default=0.08)
    ap.add_argument("--every", type=int, default=0, help="minutes between runs; 0 = run once")
    args = ap.parse_args()
    tags = resolve_tags(args.tags)
    if not tags:
        sys.exit("no valid tags")
    while True:
        try:
            run_once(args, tags)
        except Exception as e:
            print(f"run failed: {type(e).__name__}: {e}", file=sys.stderr)
        if not args.every:
            break
        print(f"\nsleeping {args.every} min ...")
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()