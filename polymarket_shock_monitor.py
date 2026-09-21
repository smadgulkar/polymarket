#!/usr/bin/env python3
"""
polymarket_shock_monitor.py  --  is the propagation residual sellable?

Backtest on last-trade prices showed: after one candidate in a negRisk event
jumps >=10c, the family's price sum sits ~17-22c above $1 at minute 1 and
closes to ~5c by minute 60. Last trades can be stale; bids cannot. This
monitor measures the same thing on live order books.

Loop (every --poll seconds):
  1. batch-fetch mid prices for every tracked leg (open negRisk events with
     >= 3 legs of >= --min-volume); keep a rolling 15-minute mid history
  2. shock = a leg whose mid moved >= --shock vs its mid --lookback minutes ago
  3. on a shock, snapshot the FULL BOOK of every leg at +0, +1, +5, +15, +60 min
     and record per leg: bid, ask, depth within 5c, and pre-shock mid
  4. per snapshot, family-level:
        sum_bid   = sum of best bids of ALL legs   (>1  => sell every leg at bid = riskless)
        sum_mid   = sum of mids
        sib_lag   = sum over siblings of (bid_now - premid) = how much of the
                    siblings' required give-back is STILL sellable at the bid
  Everything appends to shock_log.csv (one row per leg per snapshot, plus a
  FAMILY row) so two weeks of runs is the dataset.

Setup:  pip install curl_cffi
Run:    python polymarket_shock_monitor.py
        python polymarket_shock_monitor.py --shock 0.08 --max-events 300 --poll 30
Stop with Ctrl-C; restart any time, it re-scans events.

Reading it (after a week+): filter rows where kind == FAMILY, group by k_min.
  sib_lag at k=1 near zero      -> bids already collapsed; residual was never sellable; dead
  sib_lag at k=1 >= ~0.05       -> sellable mispricing survives the bots' window; build execution
  sum_bid > 1 at any k          -> outright riskless (rare, but log it)
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict, deque
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
SNAPSHOTS_MIN = [0, 1, 5, 15, 60]


# ----------------------------------------------------------------- http
def get_json(url, params=None, retries=3, backoff=1.0):
    last_err = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1) * 2)
                continue
            if r.status_code in (400, 404, 422):
                return {"_http_error": r.status_code, "_body": r.text[:120]}
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    return {"_http_error": "exhausted", "_body": str(last_err)[:120]}


def post_json(url, body, retries=3, backoff=1.0):
    last_err = None
    for attempt in range(retries):
        try:
            r = SESSION.post(url, json=body, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1) * 2)
                continue
            if r.status_code in (400, 404, 405, 422):
                return {"_http_error": r.status_code, "_body": r.text[:120]}
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (attempt + 1))
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
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
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


# ----------------------------------------------------------------- events
def resolve_tags(slugs):
    out = {}
    for slug in slugs:
        t = get_json(f"{GAMMA}/tags/slug/{slug}")
        if isinstance(t, dict) and "id" in t:
            out[slug] = t["id"]
        time.sleep(0.1)
    return out


def scan_open_events(tag_id, max_events=3000, page_size=100, offset_cap=1000):
    end_min = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    events, seen, offset = [], set(), 0
    while len(events) < max_events:
        params = {"closed": "false", "active": "true", "limit": page_size, "offset": offset, "tag_id": tag_id,
                  "order": "endDate", "ascending": "true", "end_date_min": end_min}
        page = get_json(f"{GAMMA}/events", params=params)
        if isinstance(page, dict):
            if offset > 0:
                page = []
            else:
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
            new = (max(eds) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if new == end_min:
                break
            end_min, offset = new, 0
        time.sleep(0.1)
    return events[:max_events]


def build_families(tags, args):
    fams = {}
    seen = set()
    for slug, tid in tags.items():
        for e in scan_open_events(tid):
            if e.get("id") in seen or not e.get("negRisk"):
                continue
            seen.add(e.get("id"))
            legs = []
            for m in e.get("markets") or []:
                if m.get("closed") or m.get("active") is False:
                    continue
                outcomes, tokens = parse_list(m.get("outcomes")), parse_list(m.get("clobTokenIds"))
                if not (outcomes and tokens and len(outcomes) == len(tokens) == 2):
                    continue
                labels = [str(o).strip().lower() for o in outcomes]
                yi = labels.index("yes") if "yes" in labels else 0
                vol = to_float(m.get("volumeNum") or m.get("volume")) or 0.0
                if vol < args.min_volume:
                    continue
                legs.append({"tok": tokens[yi], "title": (m.get("groupItemTitle") or m.get("question") or "")[:60],
                             "vol": vol, "mid": None})
            if len(legs) >= 3:
                fams[e.get("id")] = {"id": e.get("id"), "title": (e.get("title") or "")[:70], "tag": slug,
                                     "legs": legs, "vol": sum(l["vol"] for l in legs), "cooldown_until": 0}
    ordered = sorted(fams.values(), key=lambda f: -f["vol"])[:args.max_events]
    return {f["id"]: f for f in ordered}


# ----------------------------------------------------------------- prices / books
_batch_mode = {"ok": None}


def fetch_mids(tokens):
    """Batch midpoints. Tries POST /midpoints, falls back to GET /midpoint per token."""
    out = {}
    if _batch_mode["ok"] is not False:
        for i in range(0, len(tokens), 400):
            chunk = tokens[i:i + 400]
            data = post_json(f"{CLOB}/midpoints", [{"token_id": t} for t in chunk])
            if isinstance(data, dict) and "_http_error" not in data:
                for t in chunk:
                    v = data.get(t)
                    if isinstance(v, dict):
                        v = v.get("mid") or v.get("midpoint")
                    v = to_float(v)
                    if v is not None:
                        out[t] = v
                _batch_mode["ok"] = True
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                for d in data:
                    t, v = d.get("token_id") or d.get("asset_id"), to_float(d.get("mid") or d.get("midpoint"))
                    if t and v is not None:
                        out[t] = v
                _batch_mode["ok"] = True
            else:
                if _batch_mode["ok"] is None:
                    print(f"  batch /midpoints unavailable ({data.get('_http_error') if isinstance(data, dict) else 'shape'}); "
                          f"falling back to per-token GET (slower; reduce --max-events if polls lag)")
                _batch_mode["ok"] = False
                break
            time.sleep(0.1)
    if _batch_mode["ok"] is False:
        for t in tokens:
            d = get_json(f"{CLOB}/midpoint", params={"token_id": t})
            v = to_float(d.get("mid")) if isinstance(d, dict) else None
            if v is not None:
                out[t] = v
            time.sleep(0.03)
    return out


def fetch_book(token):
    d = get_json(f"{CLOB}/book", params={"token_id": token})
    if not isinstance(d, dict) or "_http_error" in d:
        return None
    bids = sorted([(to_float(b.get("price")), to_float(b.get("size"))) for b in d.get("bids", [])], key=lambda x: -(x[0] or 0))
    asks = sorted([(to_float(a.get("price")), to_float(a.get("size"))) for a in d.get("asks", [])], key=lambda x: (x[0] or 9))
    bids = [(p, s) for p, s in bids if p is not None and s is not None]
    asks = [(p, s) for p, s in asks if p is not None and s is not None]
    bb = bids[0][0] if bids else None
    ba = asks[0][0] if asks else None
    return {"bid": bb, "ask": ba,
            "bid_d5": round(sum(p * s for p, s in bids if bb is not None and p >= bb - 0.05), 2),
            "ask_d5": round(sum(p * s for p, s in asks if ba is not None and p <= ba + 0.05), 2),
            "mid": (bb + ba) / 2 if bb is not None and ba is not None else (bb or ba)}


# ----------------------------------------------------------------- snapshot
def snapshot(fam, shock, k, now_ts, writer, fh):
    rows = []
    books = {}
    for l in fam["legs"]:
        books[l["tok"]] = fetch_book(l["tok"])
        time.sleep(0.05)
    ts_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sum_bid = sum_ask = sum_mid = 0.0
    sib_lag_bid = sib_lag_mid = 0.0
    n_ok = 0
    for l in fam["legs"]:
        b = books[l["tok"]]
        premid = shock["premid"].get(l["tok"])
        is_shock = l["tok"] == shock["tok"]
        row = {"kind": "LEG", "ts": ts_iso, "event_id": fam["id"], "event": fam["title"], "tag": fam["tag"],
               "shock_id": shock["id"], "shock_leg": shock["title"], "shock_dir": shock["dir"], "shock_size": shock["size"],
               "k_min": k, "leg": l["title"], "is_shock_leg": int(is_shock), "leg_vol": round(l["vol"]),
               "premid": premid, "bid": None, "ask": None, "mid": None, "bid_d5": None, "ask_d5": None,
               "bid_minus_premid": None}
        if b and b["bid"] is not None and b["ask"] is not None:
            n_ok += 1
            row.update(bid=b["bid"], ask=b["ask"], mid=round(b["mid"], 4), bid_d5=b["bid_d5"], ask_d5=b["ask_d5"])
            sum_bid += b["bid"]
            sum_ask += b["ask"]
            sum_mid += b["mid"]
            if premid is not None:
                row["bid_minus_premid"] = round(b["bid"] - premid, 4)
                if not is_shock:
                    sib_lag_bid += b["bid"] - premid
                    sib_lag_mid += b["mid"] - premid
        rows.append(row)
    fam_row = {"kind": "FAMILY", "ts": ts_iso, "event_id": fam["id"], "event": fam["title"], "tag": fam["tag"],
               "shock_id": shock["id"], "shock_leg": shock["title"], "shock_dir": shock["dir"], "shock_size": shock["size"],
               "k_min": k, "leg": f"{n_ok}/{len(fam['legs'])} legs with two-sided books", "is_shock_leg": None,
               "leg_vol": round(fam["vol"]), "premid": round(sum(v for v in shock["premid"].values() if v is not None), 4),
               "bid": round(sum_bid, 4), "ask": round(sum_ask, 4), "mid": round(sum_mid, 4),
               "bid_d5": None, "ask_d5": None,
               "bid_minus_premid": round(sib_lag_bid, 4)}
    rows.append(fam_row)
    for r in rows:
        writer.writerow(r)
    fh.flush()
    riskless = sum_bid - 1.0 if n_ok == len(fam["legs"]) else None
    print(f"  [{ts_iso}] k={k:2d}m  {fam['title'][:45]:45s}  sum_bid={sum_bid:.3f} sum_mid={sum_mid:.3f}  "
          f"sibling bids still above pre-shock mids by {sib_lag_bid:+.3f}"
          + (f"  RISKLESS +{riskless:.3f}" if riskless is not None and riskless > 0.005 else ""))


# ----------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--min-volume", type=float, default=5000.0)
    ap.add_argument("--max-events", type=int, default=250, help="track the N highest-volume negRisk events")
    ap.add_argument("--poll", type=int, default=30, help="seconds between mid polls")
    ap.add_argument("--shock", type=float, default=0.10, help="mid move that counts as a shock")
    ap.add_argument("--lookback", type=int, default=10, help="minutes: compare mid now vs mid this long ago")
    ap.add_argument("--cooldown", type=int, default=90, help="minutes before the same family can re-trigger")
    ap.add_argument("--rescan", type=int, default=60, help="minutes between event re-scans")
    ap.add_argument("--log", default="shock_log.csv")
    args = ap.parse_args()

    tags = resolve_tags(args.tags)
    if not tags:
        sys.exit("no valid tags")
    fields = ["kind", "ts", "event_id", "event", "tag", "shock_id", "shock_leg", "shock_dir", "shock_size", "k_min",
              "leg", "is_shock_leg", "leg_vol", "premid", "bid", "ask", "mid", "bid_d5", "ask_d5", "bid_minus_premid"]
    new_file = not os.path.exists(args.log)
    fh = open(args.log, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if new_file:
        writer.writeheader()
        fh.flush()

    fams = build_families(tags, args)
    tok_to_fam = {l["tok"]: f["id"] for f in fams.values() for l in f["legs"]}
    print(f"tracking {len(fams)} negRisk events, {len(tok_to_fam)} legs; poll every {args.poll}s; shock >= {args.shock}")
    hist = defaultdict(lambda: deque(maxlen=200))  # tok -> deque of (ts, mid)
    pending = []  # (due_ts, fam_id, shock, k)
    last_scan = time.time()
    n_shocks = 0
    try:
        while True:
            t_start = time.time()
            if t_start - last_scan > args.rescan * 60:
                fams = build_families(tags, args)
                tok_to_fam = {l["tok"]: f["id"] for f in fams.values() for l in f["legs"]}
                last_scan = t_start
                print(f"re-scanned: {len(fams)} events, {len(tok_to_fam)} legs")
            mids = fetch_mids(list(tok_to_fam))
            now = time.time()
            for tok, mid in mids.items():
                hist[tok].append((now, mid))
            # detect shocks
            for fid, fam in fams.items():
                if fam["cooldown_until"] > now:
                    continue
                best = None
                for l in fam["legs"]:
                    h = hist[l["tok"]]
                    if len(h) < 2 or l["tok"] not in mids:
                        continue
                    cutoff = now - args.lookback * 60
                    past = [m for t, m in h if t <= cutoff]
                    if not past:
                        continue
                    ref = past[-1]
                    d = mids[l["tok"]] - ref
                    if abs(d) >= args.shock and (best is None or abs(d) > abs(best[1])):
                        best = (l, d, ref)
                if best:
                    l, d, ref = best
                    # pre-shock mids for every leg = their mid at the lookback reference time
                    premid = {}
                    for x in fam["legs"]:
                        hx = hist[x["tok"]]
                        pastx = [m for t, m in hx if t <= now - args.lookback * 60]
                        premid[x["tok"]] = pastx[-1] if pastx else (mids.get(x["tok"]))
                    n_shocks += 1
                    shock = {"id": n_shocks, "tok": l["tok"], "title": l["title"], "dir": 1 if d > 0 else -1,
                             "size": round(abs(d), 4), "premid": premid, "t0": now}
                    fam["cooldown_until"] = now + args.cooldown * 60
                    print(f"\nSHOCK #{n_shocks}  {fam['title'][:50]}  |  {l['title'][:40]}  {ref:.3f} -> {mids[l['tok']]:.3f} ({d:+.3f})")
                    for k in SNAPSHOTS_MIN:
                        pending.append((now + k * 60, fid, shock, k))
            # run due snapshots
            due = [p for p in pending if p[0] <= time.time()]
            pending = [p for p in pending if p[0] > time.time()]
            for _, fid, shock, k in sorted(due, key=lambda p: p[0]):
                if fid in fams:
                    snapshot(fams[fid], shock, k, time.time(), writer, fh)
            elapsed = time.time() - t_start
            if int(time.time()) % 600 < args.poll:
                print(f"  alive {datetime.now(timezone.utc).strftime('%H:%M')}Z  legs={len(tok_to_fam)} shocks={n_shocks} pending={len(pending)} poll={elapsed:.0f}s")
            time.sleep(max(1.0, args.poll - elapsed))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        fh.close()


if __name__ == "__main__":
    main()