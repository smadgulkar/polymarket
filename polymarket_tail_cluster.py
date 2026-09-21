#!/usr/bin/env python3
"""
polymarket_tail_cluster.py  --  geopolitical tail-cluster monitor

Thesis being traded (stated honestly): cheap geopolitical contracts are priced
off a CALM base rate, one market at a time. Escalation is a regime switch, not
a constant-rate process: when it starts, one real-world event resolves many
contracts at once. The crowd stays anchored to the calm rate for hours after
the regime has visibly changed. The edge, if it exists, is ENTRY TIMING --
being positioned in the first hours of a break -- not sitting long premium.

Measured EV on this bucket in our own backtest was ~zero (geopolitics 2/27,
1/27, 3/23 at ~5.5c, i.e. realized ~= price). This is a FAIR 20:1 lottery
with correlated upside, not a proven edge. Size accordingly.

What this does, every --poll minutes:
  1. pulls all OPEN markets in geo/conflict tags
  2. keeps CHEAP (ask <= --max-ask) NEAR-DATED (<= --max-days) contracts with a
     real book, plus a WATCH tier for cheap contracts of any date
  3. groups them into CLUSTERS by shared entities (countries, actors, straits,
     leaders) extracted from the question text -- these are contracts one
     real-world event would resolve together
  4. tracks each cluster's mean mid over time and fires a REGIME ALERT when
     the cluster moves together (mean mid up >= --cluster-move, or >= --n-move
     members each up >= --member-move) within --window minutes
  5. prints a ranked entry sheet: cost to buy N contracts across the cluster,
     payoff if the cluster resolves, breakeven probability, depth at the ask

Output: tail_watch.csv (every poll, every tracked contract) and
        tail_alerts.csv (cluster regime alerts with the entry sheet)

Setup:  pip install curl_cffi
Run:    python polymarket_tail_cluster.py
        python polymarket_tail_cluster.py --max-ask 0.08 --max-days 45 --poll 15
Stop with Ctrl-C.

Nothing here places orders. It tells you what to look at and what it costs.
"""

import argparse
import csv
import json
import os
import re
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

DEFAULT_TAGS = ["geopolitics", "world", "politics", "middle-east", "ukraine", "iran",
                "israel", "china", "russia", "economy", "crypto"]

# Entities that define a cluster: one real-world event resolves contracts sharing these.
ENTITIES = [
    # actors / states
    "iran", "israel", "hezbollah", "hamas", "houthi", "yemen", "lebanon", "syria", "iraq",
    "saudi", "uae", "qatar", "bahrain", "kuwait", "jordan", "egypt", "turkey", "turkiye",
    "russia", "ukraine", "nato", "belarus", "poland", "moldova", "kaliningrad",
    "china", "taiwan", "philippines", "japan", "north korea", "south korea", "pakistan",
    "india", "afghanistan", "venezuela", "colombia", "cuba", "nigeria", "sudan", "ethiopia",
    "libya", "mali", "niger", "serbia", "kosovo", "armenia", "azerbaijan", "georgia",
    # chokepoints / places
    "hormuz", "suez", "red sea", "bab el-mandeb", "black sea", "taiwan strait", "gaza",
    "west bank", "golan", "kharg", "crimea", "donetsk", "zaporizhzhia", "kursk",
    # leaders
    "khamenei", "netanyahu", "putin", "zelensky", "xi", "kim jong", "maduro", "erdogan",
    "trump", "modi", "assad", "mbs",
    # escalation nouns
    "nuclear", "ceasefire", "strike", "invade", "invasion", "airspace", "blockade",
    "mobilization", "martial law", "coup", "assassinat", "war",
]
ENTITY_RE = {e: re.compile(r"\b" + re.escape(e), re.I) for e in ENTITIES}


def entities_of(text):
    t = (text or "").lower()
    return frozenset(e for e, rx in ENTITY_RE.items() if rx.search(t))


# ----------------------------------------------------------------- http
def get_json(url, params=None, retries=3, backoff=1.0):
    last = None
    for a in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(backoff * (a + 1) * 2)
                continue
            if r.status_code in (400, 404, 422):
                return {"_http_error": r.status_code}
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(backoff * (a + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff * (a + 1))
    return {"_http_error": "exhausted", "_body": str(last)[:120]}


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


# ----------------------------------------------------------------- scan
def resolve_tags(slugs):
    out = {}
    for s in slugs:
        t = get_json(f"{GAMMA}/tags/slug/{s}")
        if isinstance(t, dict) and "id" in t:
            out[s] = t["id"]
        time.sleep(0.1)
    return out


def scan_open(tag_id, max_markets=4000, page_size=100, offset_cap=1000):
    end_min = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out, seen, offset = [], set(), 0
    while len(out) < max_markets:
        params = {"closed": "false", "active": "true", "limit": page_size, "offset": offset,
                  "tag_id": tag_id, "order": "endDate", "ascending": "true", "end_date_min": end_min}
        page = get_json(f"{GAMMA}/markets", params=params)
        if isinstance(page, dict):
            if offset > 0:
                page = []
            else:
                break
        for m in page:
            if m.get("id") not in seen:
                seen.add(m.get("id"))
                out.append(m)
        offset += len(page)
        if (not page) or len(page) < page_size or offset >= offset_cap:
            eds = [parse_iso(m.get("endDate")) for m in out]
            eds = [d for d in eds if d]
            if not eds:
                break
            new = (max(eds) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if new == end_min:
                break
            end_min, offset = new, 0
        time.sleep(0.1)
    return out[:max_markets]


def yes_token(m):
    outs, toks = parse_list(m.get("outcomes")), parse_list(m.get("clobTokenIds"))
    if not (outs and toks and len(outs) == len(toks) == 2):
        return None
    labels = [str(o).strip().lower() for o in outs]
    return toks[labels.index("yes")] if "yes" in labels else toks[0]


def fetch_book(token):
    d = get_json(f"{CLOB}/book", params={"token_id": token})
    if not isinstance(d, dict) or "_http_error" in d:
        return None
    bids = sorted([(to_float(b.get("price")), to_float(b.get("size"))) for b in d.get("bids", [])],
                  key=lambda x: -(x[0] or 0))
    asks = sorted([(to_float(a.get("price")), to_float(a.get("size"))) for a in d.get("asks", [])],
                  key=lambda x: (x[0] if x[0] is not None else 9))
    bids = [(p, s) for p, s in bids if p and s]
    asks = [(p, s) for p, s in asks if p and s]
    bb = bids[0][0] if bids else None
    ba = asks[0][0] if asks else None
    return {"bid": bb, "ask": ba, "mid": ((bb + ba) / 2 if bb and ba else (bb or ba)),
            "ask_sz": round(asks[0][0] * asks[0][1], 2) if asks else 0.0,
            "ask_d2": round(sum(p * s for p, s in asks if ba and p <= ba + 0.02), 2) if asks else 0.0}


# ----------------------------------------------------------------- clustering
def build_universe(tags, args):
    """Returns list of contract dicts (cheap-ish, open, with entities)."""
    seen, uni = set(), []
    now = datetime.now(timezone.utc)
    for slug, tid in tags.items():
        for m in scan_open(tid):
            mid_ = m.get("id")
            if mid_ in seen:
                continue
            seen.add(mid_)
            q = m.get("question") or ""
            ents = entities_of(q)
            if not ents:
                continue
            tok = yes_token(m)
            end = parse_iso(m.get("endDate"))
            if not tok or not end:
                continue
            days = (end - now).total_seconds() / 86400
            if days <= 0 or days > args.watch_days:
                continue
            vol = to_float(m.get("volumeNum") or m.get("volume")) or 0.0
            if vol < args.min_volume:
                continue
            uni.append({"id": mid_, "q": q[:90], "slug": m.get("slug"), "tok": tok, "tag": slug,
                        "end": end, "days": round(days, 1), "vol": vol, "ents": ents})
    return uni


def cluster(uni, args):
    """Group contracts by their most distinctive shared entity. A contract can
    sit in several clusters -- that is the point: one event resolves many."""
    by_ent = defaultdict(list)
    for c in uni:
        for e in c["ents"]:
            by_ent[e].append(c)
    out = {}
    for e, members in by_ent.items():
        if len(members) >= args.min_cluster:
            out[e] = sorted(members, key=lambda c: c["days"])
    return out


# ----------------------------------------------------------------- reporting
def entry_sheet(name, members, books, args):
    """Cost/payoff of buying 1 contract in each cheap near-dated member."""
    legs = []
    for c in members:
        b = books.get(c["tok"])
        if not b or b["ask"] is None:
            continue
        if b["ask"] > args.max_ask or c["days"] > args.max_days:
            continue
        if b["ask_d2"] < args.min_depth:
            continue
        legs.append((c, b))
    if not legs:
        return None
    cost = sum(b["ask"] for _, b in legs)
    n = len(legs)
    return {"cluster": name, "n_legs": n, "cost_per_set": round(cost, 4),
            "payoff_if_all": n, "payoff_if_half": n / 2,
            "breakeven_frac": round(cost / n, 4),
            "legs": legs}


def print_sheet(sheet, tag=""):
    if not sheet:
        return
    print(f"\n  [{sheet['cluster'].upper()}]{tag}  {sheet['n_legs']} cheap near-dated legs   "
          f"1 set costs ${sheet['cost_per_set']:.2f}  ->  pays ${sheet['payoff_if_all']:.0f} if all resolve YES, "
          f"${sheet['payoff_if_half']:.1f} if half do")
    print(f"     breakeven: {sheet['breakeven_frac']:.1%} of legs must hit")
    for c, b in sorted(sheet["legs"], key=lambda x: x[1]["ask"]):
        print(f"       {b['ask']:.3f}  ({c['days']:5.1f}d)  depth ${b['ask_d2']:>8,.0f}  vol ${c['vol']:>9,.0f}  {c['q'][:66]}")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--max-ask", type=float, default=0.10, help="only contracts this cheap are entry candidates")
    ap.add_argument("--max-days", type=int, default=60, help="entry candidates must resolve within this")
    ap.add_argument("--watch-days", type=int, default=400, help="tracked (for cluster movement) out to here")
    ap.add_argument("--min-volume", type=float, default=1000.0)
    ap.add_argument("--min-depth", type=float, default=20.0, help="USDC available at/near the ask")
    ap.add_argument("--min-cluster", type=int, default=3, help="contracts sharing an entity to count as a cluster")
    ap.add_argument("--poll", type=int, default=15, help="minutes between polls")
    ap.add_argument("--window", type=int, default=120, help="minutes: cluster move measured against this lookback")
    ap.add_argument("--cluster-move", type=float, default=0.02, help="mean mid rise that fires a regime alert")
    ap.add_argument("--member-move", type=float, default=0.03, help="per-member rise counted toward n-move")
    ap.add_argument("--n-move", type=int, default=3, help="members moving that fires a regime alert")
    ap.add_argument("--rescan", type=int, default=6, help="hours between universe re-scans")
    ap.add_argument("--show", type=int, default=8, help="standing sheets printed each poll")
    ap.add_argument("--watch-log", default="tail_watch.csv")
    ap.add_argument("--alert-log", default="tail_alerts.csv")
    args = ap.parse_args()

    tags = resolve_tags(args.tags)
    if not tags:
        sys.exit("no valid tags")
    print(f"tags: {list(tags)}")

    wf = open(args.watch_log, "a", newline="", encoding="utf-8")
    wfields = ["ts", "cluster_count", "market_id", "question", "tag", "days_to_end", "volume",
               "bid", "ask", "mid", "ask_depth2"]
    ww = csv.DictWriter(wf, fieldnames=wfields)
    if not os.path.exists(args.watch_log) or os.path.getsize(args.watch_log) == 0:
        ww.writeheader()
    af = open(args.alert_log, "a", newline="", encoding="utf-8")
    afields = ["ts", "cluster", "trigger", "mean_mid_now", "mean_mid_before", "cluster_move",
               "n_members_up", "n_legs", "cost_per_set", "breakeven_frac", "legs"]
    aw = csv.DictWriter(af, fieldnames=afields)
    if not os.path.exists(args.alert_log) or os.path.getsize(args.alert_log) == 0:
        aw.writeheader()

    uni = build_universe(tags, args)
    clusters = cluster(uni, args)
    last_scan = time.time()
    hist = defaultdict(lambda: deque(maxlen=400))   # cluster -> deque[(ts, mean_mid)]
    mhist = defaultdict(lambda: deque(maxlen=400))  # token   -> deque[(ts, mid)]
    fired = {}

    print(f"universe: {len(uni)} contracts, {len(clusters)} clusters "
          f"({', '.join(sorted(clusters, key=lambda k: -len(clusters[k]))[:12])})")

    try:
        while True:
            t0 = time.time()
            if t0 - last_scan > args.rescan * 3600:
                uni = build_universe(tags, args)
                clusters = cluster(uni, args)
                last_scan = t0
                print(f"re-scanned: {len(uni)} contracts, {len(clusters)} clusters")

            toks = {c["tok"]: c for c in uni}
            books = {}
            for i, (tok, c) in enumerate(toks.items(), 1):
                b = fetch_book(tok)
                if b and b["mid"] is not None:
                    books[tok] = b
                    mhist[tok].append((t0, b["mid"]))
                time.sleep(0.05)
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

            # log everything cheap
            for c in uni:
                b = books.get(c["tok"])
                if not b or b["ask"] is None or b["ask"] > args.max_ask:
                    continue
                ww.writerow({"ts": now_iso, "cluster_count": len(c["ents"]), "market_id": c["id"],
                             "question": c["q"], "tag": c["tag"], "days_to_end": c["days"],
                             "volume": round(c["vol"]), "bid": b["bid"], "ask": b["ask"],
                             "mid": round(b["mid"], 4), "ask_depth2": b["ask_d2"]})
            wf.flush()

            # cluster means + regime detection
            alerts = []
            for name, members in clusters.items():
                mids = [books[c["tok"]]["mid"] for c in members if c["tok"] in books]
                if len(mids) < args.min_cluster:
                    continue
                mean_mid = sum(mids) / len(mids)
                hist[name].append((t0, mean_mid))
                cutoff = t0 - args.window * 60
                past = [v for t, v in hist[name] if t <= cutoff]
                if not past:
                    continue
                before = past[-1]
                move = mean_mid - before
                n_up = 0
                for c in members:
                    h = mhist[c["tok"]]
                    p = [v for t, v in h if t <= cutoff]
                    if p and c["tok"] in books and books[c["tok"]]["mid"] - p[-1] >= args.member_move:
                        n_up += 1
                trigger = None
                if move >= args.cluster_move:
                    trigger = f"cluster mean +{move:.3f}"
                elif n_up >= args.n_move:
                    trigger = f"{n_up} members each +{args.member_move:.2f}"
                if trigger and fired.get(name, 0) < t0 - 3600:
                    fired[name] = t0
                    sheet = entry_sheet(name, members, books, args)
                    alerts.append((name, trigger, mean_mid, before, move, n_up, sheet))

            if alerts:
                print(f"\n{'='*70}\n REGIME ALERTS  {now_iso} UTC\n{'='*70}")
                for name, trig, mm, bf, mv, nu, sheet in alerts:
                    print(f"\n*** {name.upper()}  --  {trig}   (mean mid {bf:.3f} -> {mm:.3f} over {args.window}m)")
                    print_sheet(sheet)
                    aw.writerow({"ts": now_iso, "cluster": name, "trigger": trig,
                                 "mean_mid_now": round(mm, 4), "mean_mid_before": round(bf, 4),
                                 "cluster_move": round(mv, 4), "n_members_up": nu,
                                 "n_legs": sheet["n_legs"] if sheet else 0,
                                 "cost_per_set": sheet["cost_per_set"] if sheet else None,
                                 "breakeven_frac": sheet["breakeven_frac"] if sheet else None,
                                 "legs": " | ".join(f"{b['ask']:.3f} {c['q'][:40]}" for c, b in sheet["legs"]) if sheet else ""})
                af.flush()

            # standing sheet: the biggest cheap clusters, alert or not
            sheets = [entry_sheet(n, m, books, args) for n, m in clusters.items()]
            sheets = [s for s in sheets if s and s["n_legs"] >= args.min_cluster]
            sheets.sort(key=lambda s: -s["n_legs"])
            print(f"\n[{now_iso}] tracked {len(books)}/{len(uni)} books | "
                  f"{len(sheets)} clusters with >={args.min_cluster} cheap near-dated legs | "
                  f"poll {time.time()-t0:.0f}s")
            for s in sheets[:args.show]:
                print_sheet(s)

            time.sleep(max(5.0, args.poll * 60 - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        wf.close()
        af.close()


if __name__ == "__main__":
    main()