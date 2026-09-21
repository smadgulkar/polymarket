#!/usr/bin/env python3
"""
polymarket_copy_pipeline.py  --  find copyable wallets AND test copying them, in one run

Stage 1  HARVEST   high-volume resolved non-sports markets -> public trade tape
                   -> every wallet that traded there
Stage 2  PROFILE   pull each wallet's activity once (cached) and measure:
                     directional_share  two-sided book = market maker, not copyable
                     median_hold_h      seconds/minutes = cannot be followed
                     order % of volume  their own trade moved the price
                     pnl_concentration  one lucky bet is not a method
                     win rate / ROI on resolved directional markets
Stage 3  BACKTEST   for every wallet that survives, replay its BUYs and fill the
                   copier at the market price +1 / +5 / +15 minutes later
                   (minute history), hold to resolution, and compute what
                   fraction of the wallet's ROI survives the delay
Stage 4  RANK       one table: whale ROI vs copier ROI, sorted by copier ROI
                   weighted by sample size

Every stage caches to disk (activity, market info, minute history), so a
re-run or a crash costs minutes, not hours.

Setup:  pip install curl_cffi
Run:    python polymarket_copy_pipeline.py
        python polymarket_copy_pipeline.py --markets 20 --max-wallets 200 --backtest-top 15
        python polymarket_copy_pipeline.py --wallets-file addrs.txt   (skip harvesting)
        python polymarket_copy_pipeline.py --resume                   (reuse all caches)

Outputs:
    wallet_profiles.csv   every wallet profiled, with the reason it failed
    copy_results.csv      per-trade backtest rows for the tested wallets
    copy_ranking.csv      the final table

This does not predict the future. It tells you which wallets have a shape that
can be copied at all, and how much of their edge survives entering a minute late.
"""

import argparse
import csv
import json
import os
import statistics as stats
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

try:
    from curl_cffi import requests
except ImportError:
    sys.exit("curl_cffi not installed. Run:  pip install curl_cffi")

DATA = "https://data-api.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SESSION = requests.Session(impersonate="chrome")
DEFAULT_TAGS = ["politics", "geopolitics", "world", "economy", "tech", "business",
                "crypto", "elections", "science", "pop-culture", "finance"]

CACHE_DIR = "copy_cache"


# ----------------------------------------------------------------- infra
def get_json(url, params=None, retries=3, backoff=1.2):
    last = None
    for a in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (a + 1) * 3)
                continue
            if r.status_code in (400, 404, 422):
                return {"_http_error": r.status_code, "_body": r.text[:150]}
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(backoff * (a + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff * (a + 1))
    return {"_http_error": "exhausted", "_body": str(last)[:150]}


class Cache:
    """Dict-backed JSON cache, flushed periodically and on exit."""

    def __init__(self, name, every=200):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = os.path.join(CACHE_DIR, name)
        self.every = every
        self.n = 0
        self.d = {}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.d = json.load(f)
            except ValueError:
                self.d = {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v
        self.n += 1
        if self.n % self.every == 0:
            self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f)
        os.replace(tmp, self.path)

    def __len__(self):
        return len(self.d)


def first(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_ts(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v if v < 1e12 else v / 1000)
    s = str(v)
    if s.replace(".", "").isdigit():
        f = float(s)
        return int(f if f < 1e12 else f / 1000)
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
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


# ----------------------------------------------------------------- shared fetchers
def market_info(cid, cache):
    hit = cache.get(cid)
    if hit is not None:
        return hit if hit else None
    m = get_json(f"{GAMMA}/markets", params={"condition_ids": cid, "limit": 1})
    info = None
    if isinstance(m, list) and m:
        m = m[0]
        outs = parse_list(m.get("outcomes"))
        fps = parse_list(m.get("outcomePrices"))
        toks = parse_list(m.get("clobTokenIds"))
        fps = [to_float(x) for x in fps] if fps else None
        info = {"q": (m.get("question") or "")[:70], "closed": bool(m.get("closed")),
                "outcomes": outs, "final": fps, "tokens": toks,
                "volume": to_float(m.get("volumeNum") or m.get("volume")) or 0.0,
                "sports": bool(m.get("gameStartTime") or m.get("sportsMarketType")),
                "end": to_ts(m.get("closedTime") or m.get("umaEndDate") or m.get("endDate"))}
    cache.put(cid, info if info else {})
    time.sleep(0.07)
    return info


def wallet_activity(addr, cache, page_size=500, max_pages=12):
    hit = cache.get(addr)
    if hit is not None:
        return hit
    out, offset = [], 0
    for _ in range(max_pages):
        page = get_json(f"{DATA}/activity", params={"user": addr, "limit": page_size, "offset": offset})
        if isinstance(page, dict):
            if "_http_error" in page:
                break
            page = page.get("data") or []
        if not page:
            break
        out.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
        time.sleep(0.12)
    cache.put(addr, out)
    return out


def minute_history(token, t0, t1, cache):
    key = f"{token}:{int(t0)}:{int(t1)}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    d = get_json(f"{CLOB}/prices-history",
                 params={"market": token, "startTs": int(t0), "endTs": int(t1), "fidelity": 1})
    pts = []
    if isinstance(d, dict) and "history" in d:
        for h in d["history"]:
            try:
                pts.append([int(h["t"]), float(h["p"])])
            except (KeyError, TypeError, ValueError):
                pass
        pts.sort()
    cache.put(key, pts)
    time.sleep(0.06)
    return pts


def price_at_or_after(pts, ts, max_gap_s):
    for t, p in pts:
        if t >= ts:
            return p if (t - ts) <= max_gap_s else None
    return None


def trades_from_activity(act):
    out = []
    for a in act:
        if str(first(a, "type", "activityType", default="")).upper() != "TRADE":
            continue
        ts, cid = to_ts(first(a, "timestamp", "time")), first(a, "conditionId")
        price, size = to_float(first(a, "price")), to_float(first(a, "size", "shares"))
        if not (ts and cid and price and size):
            continue
        out.append({"ts": ts, "cid": cid, "outcome": str(first(a, "outcome", default="")),
                    "oidx": first(a, "outcomeIndex"), "side": str(first(a, "side", default="")).upper(),
                    "price": price, "shares": size,
                    "usd": to_float(first(a, "usdcSize")) or price * size,
                    "title": first(a, "title", default="")})
    out.sort(key=lambda t: t["ts"])
    return out


def outcome_index(info, outcome, oidx):
    outs = info.get("outcomes") or []
    if outcome and outcome in outs:
        return outs.index(outcome)
    try:
        i = int(oidx)
        if 0 <= i < len(outs):
            return i
    except (TypeError, ValueError):
        pass
    return None


# ----------------------------------------------------------------- stage 1
def resolve_tags(slugs):
    out = {}
    for s in slugs:
        t = get_json(f"{GAMMA}/tags/slug/{s}")
        if isinstance(t, dict) and "id" in t:
            out[s] = t["id"]
        time.sleep(0.1)
    return out


def top_resolved_markets(tags, n_per_tag, min_volume):
    out, seen = [], set()
    for slug, tid in tags.items():
        got, offset = 0, 0
        while got < n_per_tag and offset < 1000:
            page = get_json(f"{GAMMA}/markets", params={
                "closed": "true", "limit": 100, "offset": offset, "tag_id": tid,
                "order": "volumeNum", "ascending": "false"})
            if isinstance(page, dict) or not page:
                break
            for m in page:
                if m.get("id") in seen or m.get("gameStartTime") or m.get("sportsMarketType"):
                    continue
                vol = to_float(m.get("volumeNum") or m.get("volume")) or 0.0
                if vol < min_volume:
                    continue
                seen.add(m.get("id"))
                out.append({"cid": m.get("conditionId"), "q": (m.get("question") or "")[:60],
                            "vol": vol, "tag": slug})
                got += 1
                if got >= n_per_tag:
                    break
            offset += 100
            time.sleep(0.1)
    return out


def harvest(markets, args):
    counts, usd = defaultdict(int), defaultdict(float)
    for i, m in enumerate(markets, 1):
        if not m["cid"]:
            continue
        t = get_json(f"{DATA}/trades", params={"market": m["cid"], "limit": args.tape_limit})
        if isinstance(t, dict):
            t = t.get("data") or []
        for tr in (t if isinstance(t, list) else []):
            addr = (first(tr, "proxyWallet", "maker", "taker", "user", default="") or "").lower()
            if not addr.startswith("0x"):
                continue
            counts[addr] += 1
            usd[addr] += to_float(first(tr, "usdcSize", "size", default=0)) or 0.0
        if i % 20 == 0:
            print(f"  taped {i}/{len(markets)} markets, {len(counts)} wallets")
        time.sleep(0.08)
    cands = [(a, counts[a], usd[a]) for a in counts if counts[a] >= args.min_tape_hits]
    cands.sort(key=lambda x: -x[2])
    return cands[:args.max_wallets]


# ----------------------------------------------------------------- stage 2
def profile(addr, trades, args, mcache):
    if len(trades) < args.min_trades:
        return None, "too_few_trades"
    span_days = (trades[-1]["ts"] - trades[0]["ts"]) / 86400
    per = defaultdict(lambda: {"buys": defaultdict(lambda: {"sh": 0.0, "usd": 0.0, "first": None}), "n": 0})
    for t in trades:
        pm = per[t["cid"]]
        pm["n"] += 1
        if t["side"] == "BUY":
            k = t["outcome"] or str(t["oidx"])
            b = pm["buys"][k]
            b["sh"] += t["shares"]
            b["usd"] += t["usd"]
            b["first"] = t["ts"] if b["first"] is None else min(b["first"], t["ts"])
    n_mkts = len(per)
    directional = sum(1 for pm in per.values() if len(pm["buys"]) <= 1)
    pnls, holds, entries, impact, sports_n = [], [], [], [], 0
    for cid, pm in per.items():
        if len(pm["buys"]) != 1:
            continue
        info = market_info(cid, mcache)
        if not info:
            continue
        if info["sports"]:
            sports_n += 1
        if not info["closed"] or not info["final"]:
            continue
        k, b = next(iter(pm["buys"].items()))
        idx = outcome_index(info, k if k in (info.get("outcomes") or []) else None, k)
        if idx is None or idx >= len(info["final"]):
            continue
        fin = to_float(info["final"][idx])
        if fin is None or b["usd"] <= 0 or b["sh"] <= 0:
            continue
        won = 1 if fin >= 0.999 else 0
        pnls.append({"usd": b["usd"], "pnl": (b["sh"] - b["usd"]) if won else -b["usd"],
                     "won": won, "q": info["q"]})
        entries.append(b["usd"] / b["sh"])
        if info["end"]:
            holds.append((info["end"] - b["first"]) / 3600)
        if info["volume"] > 0:
            impact.append(b["usd"] / info["volume"])
    if len(pnls) < args.min_resolved:
        return None, "too_few_resolved_directional"
    cost = sum(p["usd"] for p in pnls)
    pnl = sum(p["pnl"] for p in pnls)
    best = max(p["pnl"] for p in pnls)
    row = {"address": addr, "n_trades": len(trades), "n_markets": n_mkts, "span_days": round(span_days, 1),
           "directional_share": round(directional / n_mkts, 3),
           "sports_share": round(sports_n / max(1, n_mkts), 3),
           "resolved_directional": len(pnls),
           "median_hold_h": round(stats.median(holds), 2) if holds else None,
           "median_order_usd": round(stats.median(p["usd"] for p in pnls), 2),
           "order_pct_of_volume": round(stats.median(impact) * 100, 3) if impact else None,
           "median_entry": round(stats.median(entries), 3),
           "win_rate": round(sum(p["won"] for p in pnls) / len(pnls), 3),
           "total_cost": round(cost, 2), "total_pnl": round(pnl, 2),
           "roi": round(pnl / cost, 4) if cost else 0.0,
           "pnl_concentration": round(best / pnl, 3) if pnl > 0 else None,
           "best_market": max(pnls, key=lambda p: p["pnl"])["q"]}
    return row, None


def gate(r, args):
    why = []
    if r["directional_share"] < args.min_directional:
        why.append(f"two_sided({r['directional_share']:.2f})")
    if r["median_hold_h"] is not None and r["median_hold_h"] < args.min_hold_h:
        why.append(f"too_fast({r['median_hold_h']:.1f}h)")
    if r["order_pct_of_volume"] is not None and r["order_pct_of_volume"] > args.max_impact_pct:
        why.append(f"moves_price({r['order_pct_of_volume']:.1f}%)")
    if r["pnl_concentration"] is not None and r["pnl_concentration"] > args.max_concentration:
        why.append(f"one_bet({r['pnl_concentration']:.2f})")
    if r["roi"] <= args.min_roi:
        why.append(f"roi({r['roi']:+.0%})")
    r["copyable"] = not why
    r["rejected_for"] = ",".join(why)
    r["profile_score"] = round(r["roi"] * (r["resolved_directional"] ** 0.5), 4) if r["copyable"] else -1
    return r


# ----------------------------------------------------------------- stage 3
def backtest(addr, trades, args, mcache, hcache):
    """Replay BUYs; copier fills at +delay minutes. Returns (rows, summary)."""
    # hedged markets: bought both sides within the window
    buys_by_mkt = defaultdict(lambda: defaultdict(list))
    for t in trades:
        if t["side"] == "BUY":
            buys_by_mkt[t["cid"]][t["outcome"] or str(t["oidx"])].append(t["ts"])
    hedged = set()
    for cid, outs in buys_by_mkt.items():
        ks = list(outs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if any(abs(a - b) <= args.hedge_window * 60 for a in outs[ks[i]] for b in outs[ks[j]]):
                    hedged.add(cid)
    rows, drop = [], defaultdict(int)
    maxd = max(args.delays)
    for t in [x for x in trades if x["side"] == "BUY"]:
        if t["cid"] in hedged:
            drop["hedged"] += 1
            continue
        info = market_info(t["cid"], mcache)
        if not info or not info["closed"] or not info["final"] or not info["tokens"]:
            drop["unresolved"] += 1
            continue
        idx = outcome_index(info, t["outcome"], t["oidx"])
        if idx is None or idx >= len(info["tokens"]) or idx >= len(info["final"]):
            drop["no_outcome_match"] += 1
            continue
        fin = to_float(info["final"][idx])
        if fin is None:
            drop["no_final"] += 1
            continue
        if info["volume"] and t["usd"] > info["volume"] / args.min_depth_mult:
            drop["order_too_big"] += 1
            continue
        pts = minute_history(info["tokens"][idx], t["ts"] - 120, t["ts"] + maxd * 60 + 600, hcache)
        if len(pts) < 2:
            drop["no_minute_history"] += 1
            continue
        won = 1 if fin >= 0.999 else 0
        row = {"address": addr, "ts": datetime.fromtimestamp(t["ts"], tz=timezone.utc).isoformat(),
               "title": (t["title"] or info["q"])[:70], "outcome": t["outcome"],
               "whale_price": t["price"], "usd": round(t["usd"], 2), "won": won,
               "whale_roi": round((1 / t["price"] - 1) if won else -1.0, 4)}
        ok = False
        for d in args.delays:
            p = price_at_or_after(pts, t["ts"] + d * 60, args.max_gap_min * 60)
            row[f"p{d}"] = p
            if p and 0 < p < 1:
                row[f"roi{d}"] = round((1 / p - 1) if won else -1.0, 4)
                row[f"slip{d}"] = round(p - t["price"], 4)
                ok = True
            else:
                row[f"roi{d}"], row[f"slip{d}"] = None, None
        if ok:
            rows.append(row)
        else:
            drop["no_price_after_delay"] += 1
    if not rows:
        return [], {"n": 0, "drop": dict(drop)}
    s = {"n": len(rows), "whale_roi": round(stats.mean(r["whale_roi"] for r in rows), 4),
         "win_rate": round(sum(r["won"] for r in rows) / len(rows), 3), "drop": dict(drop)}
    for d in args.delays:
        b = [r for r in rows if r.get(f"roi{d}") is not None]
        s[f"n{d}"] = len(b)
        s[f"copy_roi{d}"] = round(stats.mean(r[f"roi{d}"] for r in b), 4) if b else None
        s[f"slip{d}"] = round(stats.mean(r[f"slip{d}"] for r in b), 4) if b else None
        s[f"retained{d}"] = (round(s[f"copy_roi{d}"] / s["whale_roi"], 3)
                             if b and s["whale_roi"] else None)
    return rows, s


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--markets", type=int, default=12, help="resolved markets per tag to tape")
    ap.add_argument("--min-market-volume", type=float, default=50000.0)
    ap.add_argument("--tape-limit", type=int, default=500)
    ap.add_argument("--min-tape-hits", type=int, default=3)
    ap.add_argument("--max-wallets", type=int, default=150, help="profile this many")
    ap.add_argument("--backtest-top", type=int, default=10, help="backtest this many survivors")
    # profile gates
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--min-resolved", type=int, default=10)
    ap.add_argument("--min-directional", type=float, default=0.80)
    ap.add_argument("--min-hold-h", type=float, default=2.0)
    ap.add_argument("--max-impact-pct", type=float, default=2.0)
    ap.add_argument("--max-concentration", type=float, default=0.60)
    ap.add_argument("--min-roi", type=float, default=0.0)
    # backtest
    ap.add_argument("--delays", type=int, nargs="+", default=[1, 5, 15])
    ap.add_argument("--hedge-window", type=int, default=120)
    ap.add_argument("--min-depth-mult", type=float, default=20.0)
    ap.add_argument("--max-gap-min", type=int, default=10)
    ap.add_argument("--wallets-file")
    ap.add_argument("--resume", action="store_true", help="reuse caches (they are reused anyway)")
    args = ap.parse_args()

    mcache = Cache("markets.json")
    acache = Cache("activity.json", every=25)
    hcache = Cache("minutes.json", every=300)
    print(f"caches: {len(mcache)} markets, {len(acache)} wallets, {len(hcache)} minute windows")

    # ---- stage 1
    if args.wallets_file:
        with open(args.wallets_file) as f:
            cands = [(l.strip().lower(), 0, 0.0) for l in f if l.strip().startswith("0x")]
        print(f"STAGE 1: {len(cands)} wallets from file")
    else:
        tags = resolve_tags(args.tags)
        if not tags:
            sys.exit("no valid tags")
        print(f"STAGE 1: taping markets in {list(tags)}")
        markets = top_resolved_markets(tags, args.markets, args.min_market_volume)
        print(f"  {len(markets)} resolved non-sports markets >= ${args.min_market_volume:,.0f}")
        if not markets:
            sys.exit("no markets; lower --min-market-volume")
        cands = harvest(markets, args)
        print(f"  {len(cands)} candidate wallets (>= {args.min_tape_hits} appearances)")
    if not cands:
        sys.exit("no wallets")

    # ---- stage 2
    print(f"\nSTAGE 2: profiling {len(cands)} wallets")
    profiles, skipped, trades_by_addr = [], defaultdict(int), {}
    try:
        for i, (addr, hits, usd) in enumerate(cands, 1):
            try:
                act = wallet_activity(addr, acache)
                tr = trades_from_activity(act)
                trades_by_addr[addr] = tr
                row, why = profile(addr, tr, args, mcache)
            except Exception as e:
                row, why = None, f"exception_{type(e).__name__}"
            if row is None:
                skipped[why] += 1
            else:
                profiles.append(gate(row, args))
            if i % 10 == 0:
                print(f"  {i}/{len(cands)} profiled, {len(profiles)} usable, "
                      f"{sum(1 for p in profiles if p['copyable'])} copyable")
    finally:
        acache.flush()
        mcache.flush()

    if not profiles:
        sys.exit(f"no wallets profiled. skipped: {dict(skipped)}")
    profiles.sort(key=lambda r: -r["profile_score"])
    with open("wallet_profiles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(profiles[0].keys()))
        w.writeheader()
        w.writerows(profiles)
    good = [p for p in profiles if p["copyable"]]
    print(f"\n  profiled {len(profiles)}, copyable {len(good)}, skipped {dict(skipped)}")
    if not good:
        why = defaultdict(int)
        for p in profiles:
            for w_ in p["rejected_for"].split(","):
                if w_:
                    why[w_.split("(")[0]] += 1
        print("  nothing passed the gates. reasons:")
        for k, v in sorted(why.items(), key=lambda x: -x[1]):
            print(f"    {k:18s} {v}")
        print("  -> wrote wallet_profiles.csv anyway; loosen gates or widen --tags/--markets")
        sys.exit(0)

    # ---- stage 3
    todo = good[:args.backtest_top]
    print(f"\nSTAGE 3: backtesting the top {len(todo)} copyable wallets")
    all_rows, ranking = [], []
    try:
        for i, p in enumerate(todo, 1):
            addr = p["address"]
            tr = trades_by_addr.get(addr) or trades_from_activity(wallet_activity(addr, acache))
            rows, s = backtest(addr, tr, args, mcache, hcache)
            all_rows.extend(rows)
            rec = {"address": addr, "resolved_directional": p["resolved_directional"],
                   "median_hold_h": p["median_hold_h"], "median_entry": p["median_entry"],
                   "order_pct_of_volume": p["order_pct_of_volume"], "profile_roi": p["roi"],
                   "profile_pnl": p["total_pnl"], "tested_trades": s["n"],
                   "win_rate": s.get("win_rate"), "whale_roi": s.get("whale_roi")}
            for d in args.delays:
                rec[f"copy_roi_{d}m"] = s.get(f"copy_roi{d}")
                rec[f"slip_{d}m"] = s.get(f"slip{d}")
                rec[f"retained_{d}m"] = s.get(f"retained{d}")
            rec["dropped"] = json.dumps(s.get("drop", {}))
            ranking.append(rec)
            d0 = args.delays[0]
            print(f"  [{i}/{len(todo)}] {addr[:12]}  tested {s['n']:4d}  whale {(s.get('whale_roi') or 0):+.1%}  "
                  f"copy@{d0}m {(s.get(f'copy_roi{d0}') or 0):+.1%}  retained {(s.get(f'retained{d0}') or 0):.0%}")
    finally:
        hcache.flush()
        mcache.flush()

    # ---- stage 4
    d0 = args.delays[0]
    ranking = [r for r in ranking if r["tested_trades"] > 0]
    ranking.sort(key=lambda r: -((r.get(f"copy_roi_{d0}m") or -1) * (r["tested_trades"] ** 0.5)))
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open("copy_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k) for k in keys})
    if ranking:
        with open("copy_ranking.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
            w.writeheader()
            w.writerows(ranking)

    print(f"\n{'='*96}\nFINAL RANKING (copier enters +{d0}m after the wallet)\n{'='*96}")
    hdr = f"{'address':>14} {'n':>5} {'hold_h':>7} {'entry':>6} {'win':>5} {'whale_roi':>10}"
    for d in args.delays:
        hdr += f" {'copy@'+str(d)+'m':>10} {'keep':>5}"
    print(hdr)
    for r in ranking:
        line = (f"  {r['address'][:12]} {r['tested_trades']:5d} {(r['median_hold_h'] or 0):7.1f} "
                f"{r['median_entry']:6.3f} {(r['win_rate'] or 0):5.0%} {(r['whale_roi'] or 0):+10.1%}")
        for d in args.delays:
            line += f" {(r.get(f'copy_roi_{d}m') or 0):+10.1%} {(r.get(f'retained_{d}m') or 0):5.0%}"
        print(line)
    print("\nwrote wallet_profiles.csv, copy_results.csv, copy_ranking.csv")
    print("A wallet is worth following only if copy@1m ROI is positive on a real number of")
    print("trades. 'keep' is the share of the wallet's own ROI that survives the delay.")
    print("None of this is forward-looking: it is what copying WOULD have paid.")


if __name__ == "__main__":
    main()