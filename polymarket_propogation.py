#!/usr/bin/env python3
"""
polymarket_propagation.py  --  how fast do linked markets reprice?

Two kinds of linked families, both with an exact identity:
  negRisk events  : candidates of one race. sum of YES prices must stay ~1.
                    If candidate A jumps +D, the others must give back D.
  deadline curves : same event at several cumulative deadlines ("by Sep 30",
                    "by Dec 31"). far >= near must hold; when near jumps, far
                    must move at least as far as the identity requires.

Method:
  1. scan closed events/markets (same Gamma walk as before), build families,
     keep legs with volume >= --min-volume
  2. pull HOURLY history for every leg (cheap); find shocks = a leg moving
     >= --shock cents within one hour
  3. for each shock, pull MINUTE history for every leg in a window around it
     (CLOB accepts short startTs/endTs windows at fidelity=1)
  4. locate the shock start t0 at minute resolution, then record every leg at
     t0-1 (pre) and t0+1, +5, +15, +60, +180 minutes
  5. residual_k = mispricing of the family at k minutes after the shock:
        negRisk : sum(all legs at k) - sum(all legs pre)   (should be 0)
        curve   : max(0, near_k - far_k)  +  far's move as a fraction of the
                  move it eventually makes (fraction < 1 = lagging)
     overshoot_k on the shocked leg: (p_k - p_1) * sign(shock) (negative =
     reverted after minute 1)

The tradeable number is |residual_1| - |residual_60| (what converges between
minute 1 and minute 60) on legs with real books. Bots own minute 0; minute 1
onward is what this measures.

Setup:  pip install curl_cffi
Run:    python polymarket_propagation.py
        python polymarket_propagation.py --max-families 300 --shock 0.15
Output: propagation_shocks.csv (one row per shock per horizon) + summary.
"""

import argparse
import csv
import json
import math
import re
import statistics as st
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
HORIZONS = [1, 5, 15, 60, 180]

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
DECREASING_RE = re.compile(
    r"\b(?:not|never|without|no\s+(?!longer\b)\w+|continues?|remains?|stays?|still|maintains?|survives?|persists?)\b", re.I)


def stem_of(q):
    q = q.lower().strip().replace("’", "'")
    for rx in DEADLINE_RE:
        q = rx.sub(" ", q)
    q = re.sub(r"[^\w\s$%.'-]", " ", q)
    return re.sub(r"\s+", " ", q).strip(" ?.,-")


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


def scan_closed(path, tag_id, max_items, direction, page_size=100, offset_cap=1000):
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_max = now_s if direction == "past" else None
    end_min = now_s if direction == "future" else None
    items, seen, offset = [], set(), 0
    while len(items) < max_items:
        params = {"closed": "true", "limit": page_size, "offset": offset, "tag_id": tag_id,
                  "order": "endDate", "ascending": "false" if direction == "past" else "true"}
        if end_max:
            params["end_date_max"] = end_max
        if end_min:
            params["end_date_min"] = end_min
        page = get_json(f"{GAMMA}{path}", params=params)
        if isinstance(page, dict):
            if offset > 0:
                page = []
            else:
                break
        for e in page:
            if e.get("id") not in seen:
                seen.add(e.get("id"))
                items.append(e)
        offset += len(page)
        if (not page) or len(page) < page_size or offset >= offset_cap:
            eds = [parse_iso(e.get("endDate")) for e in items]
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
    return items[:max_items]


def yes_token_and_outcome(m):
    outcomes, final, tokens = parse_list(m.get("outcomes")), parse_list(m.get("outcomePrices")), parse_list(m.get("clobTokenIds"))
    if not (outcomes and final and tokens and len(outcomes) == len(final) == len(tokens) == 2):
        return None
    labels = [str(o).strip().lower() for o in outcomes]
    yi = labels.index("yes") if "yes" in labels else 0
    fp = to_float(final[yi])
    return tokens[yi], (1 if fp is not None and fp >= 0.999 else 0 if fp is not None and fp <= 0.001 else None)


def leg(m, kind_tag):
    tk = yes_token_and_outcome(m)
    if not tk:
        return None
    res = parse_iso(m.get("closedTime")) or parse_iso(m.get("umaEndDate")) or parse_iso(m.get("endDate"))
    created = parse_iso(m.get("createdAt")) or parse_iso(m.get("startDate"))
    if not (res and created):
        return None
    return {"id": m.get("id"), "q": m.get("question") or "", "title": m.get("groupItemTitle") or "",
            "tok": tk[0], "outcome": tk[1], "res": res, "created": created, "sched": parse_iso(m.get("endDate")),
            "vol": to_float(m.get("volumeNum") or m.get("volume")) or 0.0, "tag": kind_tag, "hourly": None}


# ----------------------------------------------------------------- history
def hist(token, fidelity, start=None, end=None):
    params = {"market": token, "fidelity": fidelity}
    if start is not None and end is not None:
        params.update({"startTs": int(start), "endTs": int(end)})
    else:
        params["interval"] = "max"
    data = get_json(f"{CLOB}/prices-history", params=params)
    if isinstance(data, dict) and "_http_error" in data:
        return None
    h = data.get("history") if isinstance(data, dict) else None
    if h is None:
        return None
    pts = []
    for x in h:
        try:
            pts.append((int(x["t"]), float(x["p"])))
        except (KeyError, TypeError, ValueError):
            pass
    pts.sort()
    return pts


def price_at(points, ts):
    best = None
    for t, p in points:
        if t <= ts:
            best = p
        else:
            break
    return best


def find_shocks(pts, shock):
    """Consecutive hourly points that differ by >= shock. Returns list of (t_prev, t_now, dp)."""
    out = []
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t1 - t0 <= 2 * 3600 and abs(p1 - p0) >= shock:
            out.append((t0, t1, p1 - p0))
    return out


def shock_start_minute(mpts, t_prev, t_now, dp, shock):
    """First minute in [t_prev, t_now] where cumulative move from p(t_prev) reaches shock/2 in the shock direction."""
    base = price_at(mpts, t_prev)
    if base is None:
        return None
    sgn = 1 if dp > 0 else -1
    for t, p in mpts:
        if t < t_prev:
            continue
        if t > t_now + 3600:
            break
        if sgn * (p - base) >= shock / 2:
            return t
    return None


# ----------------------------------------------------------------- core
def analyse_family(fam, kind, args, cache):
    """fam: list of legs (negRisk siblings or deadline-ordered curve). Returns shock rows."""
    rows = []
    legs = [l for l in fam if l["hourly"]]
    if len(legs) < 2:
        return rows
    # collect shocks across all legs; when several legs move in the same hour
    # (siblings reacting), attribute the shock to the leg that moved most
    cands = []
    for l in legs:
        for t_prev, t_now, dp in find_shocks(l["hourly"], args.shock):
            cands.append((t_now // 3600, abs(dp), l, t_prev, t_now, dp))
    by_hour = {}
    for hr, mag, l, t_prev, t_now, dp in cands:
        if hr not in by_hour or mag > by_hour[hr][0]:
            by_hour[hr] = (mag, l, t_prev, t_now, dp)
    for hr in sorted(by_hour):
        _, shocked, t_prev, t_now, dp = by_hour[hr]
        if True:
            # all legs must be open through the window
            if any(l["res"].timestamp() <= t_now + 4 * 3600 or l["created"].timestamp() > t_prev for l in legs):
                continue
            w0, w1 = t_prev - 3600, t_now + 4 * 3600
            mins = {}
            for l in legs:
                key = (l["tok"], w0, w1)
                if key not in cache:
                    cache[key] = hist(l["tok"], 1, w0, w1)
                    time.sleep(args.sleep)
                mins[l["id"]] = cache[key]
            if any(not v for v in mins.values()):
                continue
            t0 = shock_start_minute(mins[shocked["id"]], t_prev, t_now, dp, args.shock)
            if t0 is None:
                continue
            pre = {l["id"]: price_at(mins[l["id"]], t0 - 60) for l in legs}
            if any(v is None for v in pre.values()):
                continue
            sgn = 1 if dp > 0 else -1
            p_shock_1 = price_at(mins[shocked["id"]], t0 + 60)
            for k in HORIZONS:
                at = {l["id"]: price_at(mins[l["id"]], t0 + k * 60) for l in legs}
                if any(v is None for v in at.values()):
                    continue
                d_shock = at[shocked["id"]] - pre[shocked["id"]]
                others = [l for l in legs if l["id"] != shocked["id"]]
                d_others = sum(at[l["id"]] - pre[l["id"]] for l in others)
                row = {"kind": kind, "family": fam[0].get("fam_key", ""), "shocked_leg": shocked["q"][:80],
                       "n_legs": len(legs), "shock_dir": sgn, "t0": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
                       "k_min": k, "d_shock": round(d_shock, 4), "d_others_total": round(d_others, 4),
                       "min_vol_others": round(min(l["vol"] for l in others)),
                       "overshoot": round((at[shocked["id"]] - p_shock_1) * sgn, 4) if p_shock_1 is not None else None}
                if kind == "negrisk":
                    row["residual"] = round(d_shock + d_others, 4)  # sum drift; >0 means siblings haven't given it back
                else:
                    # curve: legs ordered by sched; identity far >= near. residual = max violation among adjacent pairs after k
                    viol = 0.0
                    for a, b in zip(legs, legs[1:]):
                        viol = max(viol, at[a["id"]] - at[b["id"]])
                    row["residual"] = round(viol, 4)
                    # how much of its eventual move have the OTHER legs made by k (vs 180 min)?
                rows.append(row)
    return rows


def summarise(rows):
    print("\n================ PROPAGATION ================")
    for kind in ("negrisk", "curve"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            print(f"\n{kind}: no shocks found")
            continue
        n_shock = len({(r["family"], r["t0"], r["shocked_leg"]) for r in rs})
        print(f"\n-- {kind}: {n_shock} shocks --")
        print(f"  {'k(min)':>6} {'n':>5} {'|d_shock| mean':>14} {'residual mean':>14} {'residual med':>13} {'|residual|>=0.03':>16} {'overshoot mean':>15}")
        for k in HORIZONS:
            b = [r for r in rs if r["k_min"] == k]
            if not b:
                continue
            res = [r["residual"] for r in b]
            ov = [r["overshoot"] for r in b if r["overshoot"] is not None]
            print(f"  {k:6d} {len(b):5d} {st.mean(abs(r['d_shock']) for r in b):14.3f} {st.mean(res):14.4f} "
                  f"{st.median(res):13.4f} {sum(1 for x in res if abs(x) >= 0.03)/len(res):16.0%} "
                  f"{st.mean(ov) if ov else float('nan'):15.4f}")
        # liquid subset
        liq = [r for r in rs if r["min_vol_others"] >= 10000]
        if liq:
            print(f"  liquid subset (all other legs >= $10k volume):")
            for k in HORIZONS:
                b = [r for r in liq if r["k_min"] == k]
                if b:
                    res = [r["residual"] for r in b]
                    print(f"    k={k:3d} n={len(b):4d} residual mean {st.mean(res):+.4f} median {st.median(res):+.4f}  share>=0.03: {sum(1 for x in res if abs(x)>=0.03)/len(res):.0%}")
        r1 = {(r["family"], r["t0"], r["shocked_leg"]): r["residual"] for r in rs if r["k_min"] == 1}
        r60 = {(r["family"], r["t0"], r["shocked_leg"]): r["residual"] for r in rs if r["k_min"] == 60}
        both = [(r1[key], r60[key]) for key in r1 if key in r60]
        if both:
            conv = [abs(a) - abs(b) for a, b in both]
            print(f"  convergence minute1 -> minute60: n={len(both)} mean {st.mean(conv):+.4f} median {st.median(conv):+.4f} "
                  f"(positive = mispricing at minute 1 that closed by minute 60 = tradeable)")
    print("\nresidual: negrisk = drift of the family's price sum after the shock (should be 0);")
    print("          curve   = largest near>far violation after the shock (should be 0).")
    print("overshoot: shocked leg's move from minute 1 to k, signed by shock direction (negative = reverted).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--per-tag", type=int, default=4000)
    ap.add_argument("--min-volume", type=float, default=5000.0, help="per-leg floor")
    ap.add_argument("--max-families", type=int, default=400)
    ap.add_argument("--shock", type=float, default=0.10, help="hourly move that counts as a shock")
    ap.add_argument("--out", default="propagation_shocks.csv")
    ap.add_argument("--sleep", type=float, default=0.08)
    args = ap.parse_args()

    tags = {}
    for slug in args.tags:
        t = get_json(f"{GAMMA}/tags/slug/{slug}")
        if isinstance(t, dict) and "id" in t:
            tags[slug] = t["id"]
        time.sleep(0.1)
    print(f"tags: {tags}")

    # ---- negRisk families from events ----
    families = []
    seen_ev = set()
    for slug, tid in tags.items():
        for direction in ("past", "future"):
            for e in scan_closed("/events", tid, args.per_tag, direction):
                if e.get("id") in seen_ev or not e.get("negRisk"):
                    continue
                seen_ev.add(e.get("id"))
                legs = [leg(m, slug) for m in (e.get("markets") or [])]
                legs = [l for l in legs if l and l["vol"] >= args.min_volume]
                if len(legs) >= 3:
                    for l in legs:
                        l["fam_key"] = f"event:{e.get('id')}:{(e.get('title') or '')[:50]}"
                    families.append(("negrisk", legs))
    print(f"negRisk families (>=3 legs with vol>={args.min_volume:,.0f}): {len(families)}")

    # ---- deadline curves from markets ----
    seen_m, stems = set(), defaultdict(list)
    for slug, tid in tags.items():
        for direction in ("past", "future"):
            for m in scan_closed("/markets", tid, args.per_tag, direction):
                if m.get("id") in seen_m or m.get("gameStartTime") or m.get("sportsMarketType"):
                    continue
                seen_m.add(m.get("id"))
                q = m.get("question") or ""
                if not CUMULATIVE_RE.search(q) or DECREASING_RE.search(q):
                    continue
                l = leg(m, slug)
                if l and l["vol"] >= args.min_volume and l["sched"]:
                    stems[stem_of(q)].append(l)
    n_curve = 0
    for stm, legs in stems.items():
        legs = sorted(legs, key=lambda l: l["sched"])
        if len(legs) >= 2:
            for l in legs:
                l["fam_key"] = f"curve:{stm[:60]}"
            families.append(("curve", legs))
            n_curve += 1
    print(f"deadline curves (>=2 legs): {n_curve}")
    families.sort(key=lambda f: -max(l["res"].timestamp() for l in f[1]))
    families = families[:args.max_families]
    print(f"analysing {len(families)} families ({sum(len(f[1]) for f in families)} legs) ...")

    # ---- hourly histories ----
    for i, (kind, legs) in enumerate(families, 1):
        for l in legs:
            l["hourly"] = hist(l["tok"], 60)
            time.sleep(args.sleep)
        if i % 25 == 0:
            print(f"  hourly: {i}/{len(families)} families")

    # ---- shocks + minute windows ----
    rows, cache = [], {}
    for i, (kind, legs) in enumerate(families, 1):
        rows.extend(analyse_family(legs, kind, args, cache))
        if i % 25 == 0:
            print(f"  shocks: {i}/{len(families)} families, {len(rows)} rows, {len(cache)} minute windows pulled")
    if not rows:
        sys.exit("no shocks found; lower --shock or raise --max-families")
    summarise(rows)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()