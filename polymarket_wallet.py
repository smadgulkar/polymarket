#!/usr/bin/env python3
"""
polymarket_wallet.py  --  reverse-engineer a Polymarket account from public data

Pulls everything the public data API exposes for a wallet:
    /activity   every TRADE / REDEEM / SPLIT / MERGE with price, size, market
    /positions  current open positions
    /trades     (fallback) trade list
Saves the raw JSON, then rebuilds each (market, outcome) position from the
trades: shares bought/sold, average entry price, cost, proceeds, redemption,
first/last trade time; resolves the market's final outcome via Gamma; and
prints:
    - entry price distribution (where does this account buy?)
    - win rate, PnL and ROI by entry-price bucket
    - hold time distribution
    - PnL by category (event tags) and the biggest wins / losses with titles
    - concentration: what share of total PnL came from the top 5 markets

Usage:
    pip install curl_cffi
    python polymarket_wallet.py 0x2c335066fe58fe9237c3d3dc7b275c2a034a0563
    python polymarket_wallet.py <address> --no-gamma     (skip resolution lookup)

The API's field names have changed over time; the script prints the keys of
the first record of each endpoint and adapts to the common variants. If a
section comes out empty, the raw JSON is in wallet_<addr>_*.json.
"""

import argparse
import csv
import json
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
SESSION = requests.Session(impersonate="chrome")


def get_json(url, params=None, retries=4, backoff=1.5):
    last_err = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=40)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1) * 3)
                continue
            if r.status_code in (400, 404, 422):
                return {"_http_error": r.status_code, "_body": r.text[:200]}
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (attempt + 1) * 2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    return {"_http_error": "exhausted", "_body": str(last_err)[:200]}


def first(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
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
    try:
        return int(float(v)) if str(v).replace(".", "").isdigit() else int(datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


# ----------------------------------------------------------------- pull
def pull_paged(path, addr, page_size=500, max_pages=400, extra=None):
    out, offset = [], 0
    for _ in range(max_pages):
        params = {"user": addr, "limit": page_size, "offset": offset}
        if extra:
            params.update(extra)
        page = get_json(f"{DATA}{path}", params=params)
        if isinstance(page, dict):
            if "_http_error" in page:
                print(f"  {path}: error {page['_http_error']} at offset {offset}: {page.get('_body','')[:100]}")
                break
            page = page.get("data") or page.get("results") or []
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
        time.sleep(0.15)
    return out


def pull_all(addr):
    print(f"pulling /activity ...")
    activity = pull_paged("/activity", addr)
    print(f"  {len(activity)} activity records")
    if activity:
        print("  activity keys:", sorted(activity[0].keys()))
    print(f"pulling /positions ...")
    positions = get_json(f"{DATA}/positions", params={"user": addr, "limit": 500})
    positions = positions if isinstance(positions, list) else (positions.get("data") if isinstance(positions, dict) else []) or []
    print(f"  {len(positions)} open positions")
    if positions:
        print("  position keys:", sorted(positions[0].keys()))
    trades = []
    if not activity:
        print(f"pulling /trades (fallback) ...")
        trades = pull_paged("/trades", addr)
        print(f"  {len(trades)} trades")
        if trades:
            print("  trade keys:", sorted(trades[0].keys()))
    return activity, positions, trades


# ----------------------------------------------------------------- rebuild positions
def rebuild(activity, trades):
    """Aggregate per (conditionId, outcome). Handles the common field variants."""
    pos = {}
    recs = activity if activity else trades
    for a in recs:
        typ = str(first(a, "type", "activityType", default="TRADE")).upper()
        cid = first(a, "conditionId", "condition_id", "market")
        outcome = first(a, "outcome", "outcomeName")
        oidx = first(a, "outcomeIndex", "outcome_index")
        title = first(a, "title", "question", "marketTitle", default="")
        slug = first(a, "slug", "eventSlug", "marketSlug", default="")
        key = (cid, outcome if outcome is not None else oidx)
        p = pos.setdefault(key, {"cid": cid, "outcome": outcome, "oidx": oidx, "title": title, "slug": slug,
                                 "buy_sh": 0.0, "buy_usd": 0.0, "sell_sh": 0.0, "sell_usd": 0.0,
                                 "redeem_usd": 0.0, "redeem_sh": 0.0, "n_trades": 0,
                                 "first_ts": None, "last_ts": None})
        ts = to_ts(first(a, "timestamp", "time", "createdAt", "created_at"))
        if ts:
            p["first_ts"] = ts if p["first_ts"] is None else min(p["first_ts"], ts)
            p["last_ts"] = ts if p["last_ts"] is None else max(p["last_ts"], ts)
        price = to_float(first(a, "price"))
        shares = to_float(first(a, "size", "shares", "amount"))
        usd = to_float(first(a, "usdcSize", "usdc_size", "cash", "value"))
        if usd is None and price is not None and shares is not None:
            usd = price * shares
        side = str(first(a, "side", default="")).upper()
        if typ == "TRADE":
            p["n_trades"] += 1
            if side == "BUY":
                p["buy_sh"] += shares or 0.0
                p["buy_usd"] += usd or 0.0
            elif side == "SELL":
                p["sell_sh"] += shares or 0.0
                p["sell_usd"] += usd or 0.0
        elif typ == "REDEEM":
            p["redeem_sh"] += shares or 0.0
            p["redeem_usd"] += usd or (shares or 0.0)
        # SPLIT / MERGE / CONVERSION ignored for PnL (they are 1:1 collateral ops)
    return pos


def resolve_outcomes(pos, use_gamma):
    """Look up final outcome prices per conditionId via Gamma."""
    if not use_gamma:
        return {}
    cids = sorted({p["cid"] for p in pos.values() if p["cid"]})
    print(f"resolving {len(cids)} markets via Gamma ...")
    res = {}
    for i, cid in enumerate(cids, 1):
        m = get_json(f"{GAMMA}/markets", params={"condition_ids": cid, "limit": 1})
        if isinstance(m, list) and m:
            m = m[0]
            outs = m.get("outcomes")
            fps = m.get("outcomePrices")
            try:
                outs = json.loads(outs) if isinstance(outs, str) else outs
                fps = [float(x) for x in (json.loads(fps) if isinstance(fps, str) else fps)]
            except Exception:
                outs, fps = None, None
            tags = m.get("events", [{}])[0].get("tags") if m.get("events") else None
            res[cid] = {"closed": m.get("closed"), "outcomes": outs, "final": fps,
                        "question": m.get("question"), "tags": [t.get("label") for t in (tags or []) if isinstance(t, dict)],
                        "volume": m.get("volumeNum")}
        if i % 50 == 0:
            print(f"  {i}/{len(cids)}")
        time.sleep(0.12)
    return res


def finalize(pos, res):
    rows = []
    for key, p in pos.items():
        net_sh = p["buy_sh"] - p["sell_sh"]
        avg_entry = p["buy_usd"] / p["buy_sh"] if p["buy_sh"] > 0 else None
        r = res.get(p["cid"], {})
        final_for_outcome = None
        if r.get("outcomes") and r.get("final"):
            try:
                if p["outcome"] is not None and p["outcome"] in r["outcomes"]:
                    final_for_outcome = r["final"][r["outcomes"].index(p["outcome"])]
                elif p["oidx"] is not None:
                    final_for_outcome = r["final"][int(p["oidx"])]
            except Exception:
                pass
        closed = bool(r.get("closed"))
        # PnL: proceeds from sells + redemption (or terminal value of remaining shares if closed) - cost
        terminal = 0.0
        if p["redeem_usd"] > 0:
            terminal = p["redeem_usd"]
        elif closed and final_for_outcome is not None:
            terminal = max(net_sh, 0.0) * final_for_outcome
        pnl = p["sell_usd"] + terminal - p["buy_usd"]
        status = "open" if not closed and p["redeem_usd"] == 0 and net_sh > 1e-6 else ("closed" if closed or p["redeem_usd"] > 0 else "flat")
        won = None
        if status != "open" and final_for_outcome is not None and p["buy_sh"] > 0:
            won = 1 if final_for_outcome >= 0.999 else 0
        rows.append({
            "cid": p["cid"], "title": p["title"] or r.get("question", ""), "outcome": p["outcome"], "slug": p["slug"],
            "tags": "|".join(r.get("tags") or []), "market_volume": r.get("volume"),
            "n_trades": p["n_trades"], "buy_shares": round(p["buy_sh"], 2), "buy_usd": round(p["buy_usd"], 2),
            "sell_shares": round(p["sell_sh"], 2), "sell_usd": round(p["sell_usd"], 2),
            "redeem_usd": round(p["redeem_usd"], 2), "avg_entry": round(avg_entry, 4) if avg_entry else None,
            "final_price": final_for_outcome, "status": status, "won": won, "pnl": round(pnl, 2),
            "roi": round(pnl / p["buy_usd"], 4) if p["buy_usd"] > 0 else None,
            "first": datetime.fromtimestamp(p["first_ts"], tz=timezone.utc).date().isoformat() if p["first_ts"] else None,
            "last": datetime.fromtimestamp(p["last_ts"], tz=timezone.utc).date().isoformat() if p["last_ts"] else None,
            "hold_days": round((p["last_ts"] - p["first_ts"]) / 86400, 1) if p["first_ts"] and p["last_ts"] else None,
        })
    return rows


# ----------------------------------------------------------------- report
def report(rows):
    closed = [r for r in rows if r["status"] != "open" and r["avg_entry"] is not None and r["buy_usd"] > 0]
    openp = [r for r in rows if r["status"] == "open"]
    total_pnl = sum(r["pnl"] for r in closed)
    total_cost = sum(r["buy_usd"] for r in closed)
    print(f"\n=== positions: {len(rows)}  closed/flat: {len(closed)}  open: {len(openp)} ===")
    print(f"closed cost ${total_cost:,.0f}   closed PnL ${total_pnl:,.0f}   ROI {total_pnl/total_cost:+.1%}" if total_cost else "")
    scored = [r for r in closed if r["won"] is not None]
    if scored:
        print(f"win rate (resolved, bought side paid $1): {sum(r['won'] for r in scored)/len(scored):.1%} of {len(scored)}")

    print("\n-- by average ENTRY price --")
    print(f"  {'bucket':>12} {'n':>5} {'cost $':>12} {'pnl $':>12} {'roi':>8} {'win%':>6} {'med hold d':>10}")
    for lo, hi in [(0, .05), (.05, .10), (.10, .20), (.20, .35), (.35, .50), (.50, .65), (.65, .80), (.80, .90), (.90, .97), (.97, 1.01)]:
        b = [r for r in closed if lo <= r["avg_entry"] < hi]
        if not b:
            continue
        c = sum(r["buy_usd"] for r in b)
        pn = sum(r["pnl"] for r in b)
        sc = [r for r in b if r["won"] is not None]
        wr = f"{sum(r['won'] for r in sc)/len(sc):.0%}" if sc else "n/a"
        holds = sorted(r["hold_days"] for r in b if r["hold_days"] is not None)
        mh = holds[len(holds) // 2] if holds else float("nan")
        print(f"  [{lo:.2f},{hi:.2f}) {len(b):5d} {c:12,.0f} {pn:12,.0f} {pn/c if c else 0:8.1%} {wr:>6} {mh:10.1f}")

    print("\n-- by category (first tag) --")
    bt = defaultdict(list)
    for r in closed:
        bt[(r["tags"].split("|")[0] if r["tags"] else "?")].append(r)
    for t, b in sorted(bt.items(), key=lambda kv: -sum(r["pnl"] for r in kv[1]))[:15]:
        c = sum(r["buy_usd"] for r in b)
        pn = sum(r["pnl"] for r in b)
        print(f"  {t[:28]:28s} n={len(b):4d}  cost ${c:11,.0f}  pnl ${pn:11,.0f}  roi {pn/c if c else 0:+.1%}")

    srt = sorted(closed, key=lambda r: -r["pnl"])
    print("\n-- top 10 wins --")
    for r in srt[:10]:
        print(f"  ${r['pnl']:>10,.0f}  entry {r['avg_entry']:.3f}  cost ${r['buy_usd']:>9,.0f}  hold {r['hold_days']}d  {str(r['outcome'])[:8]:8s} {r['title'][:70]}")
    print("-- top 10 losses --")
    for r in srt[-10:][::-1]:
        print(f"  ${r['pnl']:>10,.0f}  entry {r['avg_entry']:.3f}  cost ${r['buy_usd']:>9,.0f}  hold {r['hold_days']}d  {str(r['outcome'])[:8]:8s} {r['title'][:70]}")
    if total_pnl > 0:
        top5 = sum(r["pnl"] for r in srt[:5])
        print(f"\nconcentration: top 5 markets = {top5/total_pnl:.0%} of total closed PnL; top 20 = {sum(r['pnl'] for r in srt[:20])/total_pnl:.0%}")
    if openp:
        print(f"\n-- open positions: {len(openp)}, cost ${sum(r['buy_usd'] for r in openp):,.0f} --")
        for r in sorted(openp, key=lambda r: -r["buy_usd"])[:10]:
            print(f"  cost ${r['buy_usd']:>9,.0f}  entry {r['avg_entry']}  {str(r['outcome'])[:8]:8s} {r['title'][:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("address")
    ap.add_argument("--no-gamma", action="store_true")
    args = ap.parse_args()
    addr = args.address.lower().strip()
    tag = addr[:10]

    activity, positions, trades = pull_all(addr)
    with open(f"wallet_{tag}_activity.json", "w") as f:
        json.dump(activity, f)
    with open(f"wallet_{tag}_positions.json", "w") as f:
        json.dump(positions, f)
    if trades:
        with open(f"wallet_{tag}_trades.json", "w") as f:
            json.dump(trades, f)
    if not activity and not trades:
        sys.exit("no trade data returned; check the address and the raw JSON files")

    types = defaultdict(int)
    for a in activity:
        types[str(first(a, "type", "activityType", default="?")).upper()] += 1
    print("activity by type:", dict(types))

    pos = rebuild(activity, trades)
    res = resolve_outcomes(pos, not args.no_gamma)
    rows = finalize(pos, res)
    rows.sort(key=lambda r: (r["first"] or ""))
    with open(f"wallet_{tag}_positions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report(rows)
    print(f"\nwrote wallet_{tag}_positions.csv and raw JSON")


if __name__ == "__main__":
    main()