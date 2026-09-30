#!/usr/bin/env python3
"""
polymarket_crypto_longshots.py  --  find crypto longshots the options market says are underpriced

Why this can work where blind longshot buying doesn't: a longshot only has
positive EV if you know its probability better than the crowd. For crypto
price markets ("Will Bitcoin reach $150k in October?", "BTC above $90k on
Oct 3?") there IS a better estimate: Deribit's options market, where
professionals price the same event with real money. When a Polymarket
contract (YES or NO) is offered far below what the options surface implies,
that is a longshot with measurable edge.

Method, every run:
  1. Deribit: build an implied-vol surface per asset (BTC, ETH inverse options;
     SOL, XRP, HYPE, AVAX from USDC-settled options): smile per expiry in
     log-moneyness, total variance interpolated in time.
  2. Polymarket: open crypto markets; parse asset, strike(s), and type from the
     question:  touch-up ("reach/hit"), touch-down ("dip/fall/drop to"),
     above/below at expiry, between K1 and K2 at expiry.
  3. Probability: terminal digitals from the smile (call-spread finite
     difference); touches via the reflection principle (2 x terminal digital).
     Computed twice: at market IV, and CONSERVATIVELY at IV x (1 - --vol-haircut)
     for tail outcomes -- options usually overprice crypto tails (variance
     risk premium), so the signal must survive the haircut.
  4. Signal: an outcome token (YES or NO) whose ask <= --max-price and whose
     conservative probability >= --min-ratio x ask and >= ask + --min-edge.
  5. Every signal is appended to crypto_longshots_log.csv; `--score` later
     marks resolved ones and reports realized hit rate vs price paid.

Setup:  pip install curl_cffi
Run:    python polymarket_crypto_longshots.py                 # scan + log signals
        python polymarket_crypto_longshots.py --all           # also print fair-priced contracts
        python polymarket_crypto_longshots.py --unparsed      # show questions the parser skipped
        python polymarket_crypto_longshots.py --score         # score logged signals that resolved

Caveats stated once: risk-neutral probabilities are not real-world ones (the
haircut is a rough correction); touch-probability via reflection ignores skew
between now and expiry; Polymarket resolves on a specific exchange/candle,
Deribit on its index -- a few-dollar basis matters only right at the strike.
Paper-score it before sizing it.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta

try:
    from curl_cffi import requests
except ImportError:
    sys.exit("curl_cffi not installed. Run:  pip install curl_cffi")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DERIBIT = "https://www.deribit.com/api/v2/public"
SESSION = requests.Session(impersonate="chrome")
LOG = "crypto_longshots_log.csv"

# asset -> (deribit currency, instrument prefix, index name)
ASSETS = {
    "BTC": ("BTC", "BTC", "btc_usd"),
    "ETH": ("ETH", "ETH", "eth_usd"),
    "SOL": ("USDC", "SOL_USDC", "sol_usdc"),
    "XRP": ("USDC", "XRP_USDC", "xrp_usdc"),
    "HYPE": ("USDC", "HYPE_USDC", "hype_usdc"),
    "AVAX": ("USDC", "AVAX_USDC", "avax_usdc"),
}
ALIASES = [(r"\bbitcoin\b|\bbtc\b", "BTC"), (r"\bethereum\b|\beth\b|\bether\b", "ETH"),
           (r"\bsolana\b|\bsol\b", "SOL"), (r"\bxrp\b|\bripple\b", "XRP"),
           (r"\bhyperliquid\b|\bhype\b", "HYPE"), (r"\bavalanche\b|\bavax\b", "AVAX")]
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                      "sep", "oct", "nov", "dec"], 1)}


# ----------------------------------------------------------------- helpers
def get_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=25)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            return r.json() if r.status_code < 400 else None
        except Exception:
            time.sleep(1 + attempt)
    return None


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
    try:
        return json.loads(v) if isinstance(v, str) else None
    except json.JSONDecodeError:
        return None


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def now_utc():
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------- deribit surface
def deribit_expiry(code):
    """'3OCT26' -> datetime 08:00 UTC."""
    m = re.fullmatch(r"(\d{1,2})([A-Z]{3})(\d{2})", code)
    return datetime(2000 + int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)), 8,
                    tzinfo=timezone.utc)


class Surface:
    """IV surface for one asset from Deribit mark IVs (OTM side of each strike)."""

    def __init__(self, asset):
        cur, prefix, index = ASSETS[asset]
        self.asset = asset
        res = (get_json(f"{DERIBIT}/get_book_summary_by_currency", {"currency": cur, "kind": "option"})
               or {}).get("result", [])
        idx = (get_json(f"{DERIBIT}/get_index_price", {"index_name": index}) or {}).get("result", {})
        self.spot = idx.get("index_price")
        by_exp = {}
        for o in res:
            parts = o["instrument_name"].split("-")
            if parts[0] != prefix or len(parts) != 4 or not o.get("mark_iv"):
                continue
            exp = deribit_expiry(parts[1])
            k = float(parts[2].replace("d", "."))
            fwd = o.get("underlying_price") or self.spot
            if not fwd:
                continue
            otm = (parts[3] == "C" and k >= fwd) or (parts[3] == "P" and k < fwd)
            e = by_exp.setdefault(exp, {"fwd": fwd, "pts": {}})
            if otm or k not in e["pts"]:
                e["pts"][k] = o["mark_iv"] / 100.0
        now = now_utc()
        self.expiries = []
        for exp, e in sorted(by_exp.items()):
            t = (exp - now).total_seconds() / (365 * 86400)
            if t <= 0 or len(e["pts"]) < 3:
                continue
            smile = sorted((math.log(k / e["fwd"]), v) for k, v in e["pts"].items())
            self.expiries.append((t, e["fwd"], smile))
        self.ok = bool(self.spot and self.expiries)

    @staticmethod
    def _smile_iv(smile, x):
        if x <= smile[0][0]:
            return smile[0][1]
        if x >= smile[-1][0]:
            return smile[-1][1]
        for (x0, v0), (x1, v1) in zip(smile, smile[1:]):
            if x0 <= x <= x1:
                return v0 + (v1 - v0) * (x - x0) / (x1 - x0)

    def fwd(self, t):
        ts = [e[0] for e in self.expiries]
        if t <= ts[0]:
            return self.spot + (self.expiries[0][1] - self.spot) * t / ts[0]
        for (t0, f0, _), (t1, f1, _) in zip(self.expiries, self.expiries[1:]):
            if t0 <= t <= t1:
                return f0 + (f1 - f0) * (t - t0) / (t1 - t0)
        return self.expiries[-1][1]

    def iv(self, k, t):
        """IV at strike k, time t (years): linear in total variance between expiries."""
        f = self.fwd(t)
        x = math.log(k / f)
        ex = self.expiries
        if t <= ex[0][0]:
            return self._smile_iv(ex[0][2], x)
        if t >= ex[-1][0]:
            return self._smile_iv(ex[-1][2], x)
        for (t0, _, s0), (t1, _, s1) in zip(ex, ex[1:]):
            if t0 <= t <= t1:
                w0, w1 = self._smile_iv(s0, x) ** 2 * t0, self._smile_iv(s1, x) ** 2 * t1
                return math.sqrt(max(w0 + (w1 - w0) * (t - t0) / (t1 - t0), 1e-12) / t)

    def p_above(self, k, t, vol_mult=1.0):
        """Risk-neutral P(S_T > k) from the smile: -dC/dK via Black-76 finite difference."""
        f = self.fwd(t)
        h = k * 0.005

        def call(kk):
            v = self.iv(kk, t) * vol_mult
            sd = v * math.sqrt(t)
            d1 = (math.log(f / kk) + 0.5 * sd * sd) / sd
            return f * ncdf(d1) - kk * ncdf(d1 - sd)
        return min(1.0, max(0.0, -(call(k + h) - call(k - h)) / (2 * h)))


# ----------------------------------------------------------------- polymarket parsing
PRICE = r"\$\s?([\d,]+(?:\.\d+)?)\s?([kKmMbB](?![a-zA-Z]))?"
# "Bitcoin above 85,600 on ..." -- no $ sign: take the number right after a price keyword
BARE_PRICE = (r"\b(?:above|below|over|under|reach|hit|touch|to|between|and|than)\s+"
              r"(\d[\d,]*(?:\.\d+)?)\s?([kKmM](?![a-zA-Z]))?")


def to_price(num, suf):
    v = float(num.replace(",", ""))
    return v * {"k": 1e3, "m": 1e6, "b": 1e9}.get((suf or "").lower(), 1)


def parse_question(q):
    """-> dict(asset, kind, k1, k2) or None. kind: touch_up, touch_down, above, below, between."""
    ql = q.lower()
    asset = next((a for pat, a in ALIASES if re.search(pat, ql)), None)
    if not asset or re.search(r"up or down|market cap|etf|dominance|flip|all[- ]time high|ath\b", ql):
        return None
    prices = [to_price(n, s) for n, s in re.findall(PRICE, q)]
    if not prices:
        prices = [to_price(n, s) for n, s in re.findall(BARE_PRICE, q, re.I)]
    if not prices:
        return None
    if re.search(r"\bbetween\b", ql) and len(prices) >= 2:
        return {"asset": asset, "kind": "between", "k1": min(prices[:2]), "k2": max(prices[:2])}
    k = prices[0]
    if re.search(r"\b(dip|fall|drop|crash|sink|decline)s?\b.*\bto\b|\bdip\b", ql):
        return {"asset": asset, "kind": "touch_down", "k1": k, "k2": None}
    if re.search(r"\b(reach|hit|touch|surpass|cross|break)(es)?\b", ql):
        return {"asset": asset, "kind": "touch_up", "k1": k, "k2": None}
    if re.search(r"\b(above|over|higher than|greater than|at least)\b", ql):
        return {"asset": asset, "kind": "above", "k1": k, "k2": None}
    if re.search(r"\b(below|under|lower than|less than)\b", ql):
        return {"asset": asset, "kind": "below", "k1": k, "k2": None}
    return None


def model_prob(surf, p, t, vol_mult):
    """P(YES) for a parsed market under the surface, vol scaled by vol_mult."""
    kind, k1, k2 = p["kind"], p["k1"], p["k2"]
    spot = surf.spot
    if kind == "touch_up":
        return 1.0 if spot >= k1 else min(1.0, 2 * surf.p_above(k1, t, vol_mult))
    if kind == "touch_down":
        return 1.0 if spot <= k1 else min(1.0, 2 * (1 - surf.p_above(k1, t, vol_mult)))
    if kind == "above":
        return surf.p_above(k1, t, vol_mult)
    if kind == "below":
        return 1 - surf.p_above(k1, t, vol_mult)
    return max(0.0, surf.p_above(k1, t, vol_mult) - surf.p_above(k2, t, vol_mult))


def conservative(surf, p, t, haircut):
    """Lower bound for EACH outcome: tails get less likely as vol shrinks, so take
    the min over {market vol, haircut vol} separately for YES and NO."""
    a, b = model_prob(surf, p, t, 1.0), model_prob(surf, p, t, 1 - haircut)
    return a, min(a, b), min(1 - a, 1 - b)


# ----------------------------------------------------------------- market data
def crypto_markets(max_days, page_size=100, offset_cap=1000, max_total=30000):
    """All open crypto markets ending within max_days. Gamma stops paging at an
    offset cap, and thousands of 5-minute up/down markets fill the first pages,
    so walk forward by restarting from the latest endDate seen."""
    tag = get_json(f"{GAMMA}/tags/slug/crypto")
    if not isinstance(tag, dict):
        sys.exit("could not reach Polymarket (gamma-api) -- network or geoblock")
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    end_min = now_utc().strftime(fmt)
    end_max = (now_utc() + timedelta(days=max_days)).strftime(fmt)
    out, seen = [], set()
    while len(out) < max_total:
        batch, offset = [], 0
        while offset < offset_cap:
            batch = get_json(f"{GAMMA}/markets", {"closed": "false", "active": "true", "limit": page_size,
                                                  "offset": offset, "tag_id": tag["id"],
                                                  "end_date_min": end_min, "end_date_max": end_max,
                                                  "order": "endDate", "ascending": "true"})
            if not isinstance(batch, list) or not batch:
                return out
            for m in batch:
                if m.get("id") not in seen:
                    seen.add(m.get("id"))
                    out.append(m)
            if len(batch) < page_size:
                return out
            offset += page_size
            time.sleep(0.1)
        ends = [parse_iso(m.get("endDate")) for m in batch]
        latest = max((e for e in ends if e), default=None)
        if latest is None:
            return out
        nxt = latest.strftime(fmt)
        if nxt <= end_min:  # >offset_cap markets share one endDate: step past it
            nxt = (latest + timedelta(seconds=1)).strftime(fmt)
        end_min = nxt
    return out


def best_ask(token):
    d = get_json(f"{CLOB}/book", {"token_id": token})
    if not isinstance(d, dict):
        return None, 0.0
    asks = sorted((float(a["price"]), float(a["size"])) for a in d.get("asks", []))
    if not asks:
        return None, 0.0
    ask = asks[0][0]
    return ask, round(sum(p * s for p, s in asks if p <= ask + 0.01), 2)


# ----------------------------------------------------------------- scan / score
def scan(args):
    surfaces = {}
    for a in ASSETS:
        s = Surface(a)
        if s.ok:
            surfaces[a] = s
    print("deribit: " + ", ".join(f"{a} {s.spot:,.2f} ({len(s.expiries)} expiries)" for a, s in surfaces.items()))
    mkts = crypto_markets(args.max_days)
    print(f"polymarket: {len(mkts)} open crypto markets ending within {args.max_days}d")
    rows, unparsed, near, kinds, updown = [], [], [], {}, 0
    for m in mkts:
        q = m.get("question") or ""
        p = parse_question(q)
        end = parse_iso(m.get("endDate"))
        outs, toks = parse_list(m.get("outcomes")), parse_list(m.get("clobTokenIds"))
        if not (p and end and outs and toks and len(toks) == 2) or p["asset"] not in surfaces:
            if not p:
                if re.search(r"up or down", q, re.I):
                    updown += 1
                else:
                    unparsed.append(q)
            continue
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        vol = float(m.get("volumeNum") or m.get("volume") or 0)
        if vol < args.min_volume:
            continue
        t = (end - now_utc()).total_seconds() / (365 * 86400)
        if t <= 1 / 365 / 24:
            continue
        surf = surfaces[p["asset"]]
        mid, cons_yes, cons_no = conservative(surf, p, t, args.vol_haircut)
        labels = [str(o).strip().lower() for o in outs]
        yi = labels.index("yes") if "yes" in labels else 0
        for side, tok, pc, pm in (("YES", toks[yi], cons_yes, mid), ("NO", toks[1 - yi], cons_no, 1 - mid)):
            ask, depth = best_ask(tok)
            if ask is None or ask > args.max_price or depth < args.min_depth:
                continue
            ratio = pc / ask
            sig = ratio >= args.min_ratio and pc - ask >= args.min_edge
            row = {"ts": now_utc().isoformat(timespec="seconds"), "market_id": m["id"],
                             "question": q, "side": side, "token": tok, "asset": p["asset"],
                             "kind": p["kind"], "k1": p["k1"], "k2": p["k2"] or "",
                             "spot": round(surf.spot, 4), "days": round(t * 365, 2), "ask": ask,
                             "depth_usd": depth, "p_model": round(pm, 4), "p_conservative": round(pc, 4),
                             "ratio": round(ratio, 2), "payout_x": round(1 / ask, 1), "signal": int(sig)}
            near.append(row)
            if sig or args.all:
                rows.append(row)
        time.sleep(0.05)
    print(f"parsed: {sum(kinds.values())} ({', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}); "
          f"skipped {updown} up/down, {len(unparsed)} other unparsed; "
          f"{len(near)} contracts priced <= {args.max_price} with depth")
    if not rows:  # show the closest misses so a quiet scan is still informative
        rows = sorted(near, key=lambda r: -r["ratio"])[:15]
        print("no signals -- closest 15 by ratio:")
    rows.sort(key=lambda r: -r["ratio"])
    print(f"\n{'side':<4} {'ask':>6} {'p_cons':>7} {'p_mid':>6} {'ratio':>6} {'pays':>6} {'depth$':>7} {'days':>6}  question")
    for r in rows:
        mark = "*" if r["signal"] else " "
        print(f"{r['side']:<4} {r['ask']:>6.3f} {r['p_conservative']:>7.3f} {r['p_model']:>6.3f} "
              f"{r['ratio']:>5.1f}x {r['payout_x']:>5.0f}x {r['depth_usd']:>7.0f} {r['days']:>6.1f} {mark} {r['question'][:70]}")
    sigs = [r for r in rows if r["signal"]]
    print(f"\n{len(sigs)} signals (*): ask <= {args.max_price}, conservative prob >= {args.min_ratio}x ask")
    if sigs:
        new = not os.path.exists(LOG)
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sigs[0].keys()))
            if new:
                w.writeheader()
            w.writerows(sigs)
        print(f"logged to {LOG}")
    if args.unparsed:
        print(f"\nunparsed ({len(unparsed)}):")
        for q in unparsed[:80]:
            print("  ", q[:110])


def score():
    if not os.path.exists(LOG):
        sys.exit(f"{LOG} not found -- run a scan first")
    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
    first = {}
    for r in rows:  # score each (market, side) once, at the first price we saw
        first.setdefault((r["market_id"], r["side"]), r)
    resolved = []
    for (mid, side), r in first.items():
        m = get_json(f"{GAMMA}/markets/{mid}")
        if not isinstance(m, dict) or not m.get("closed"):
            continue
        outs, fin = parse_list(m.get("outcomes")), parse_list(m.get("outcomePrices"))
        labels = [str(o).strip().lower() for o in outs]
        yes = float(fin[labels.index("yes") if "yes" in labels else 0])
        if 0.02 < yes < 0.98:
            continue
        won = (yes > 0.5) == (side == "YES")
        resolved.append((r, won))
        time.sleep(0.1)
    print(f"logged signals: {len(first)} unique, resolved: {len(resolved)}")
    if not resolved:
        return
    cost = sum(float(r["ask"]) for r, _ in resolved)
    ret = sum(1.0 for _, w in resolved if w)
    exp = sum(float(r["p_conservative"]) for r, _ in resolved)
    print(f"hits {int(ret)}/{len(resolved)}   expected (conservative) {exp:.1f}   "
          f"return on $1 per contract: {ret / cost - 1:+.0%}")
    for r, w in resolved:
        print(f"  {'WIN ' if w else 'lose'} {r['side']} @ {float(r['ask']):.3f} (p {float(r['p_conservative']):.3f})  {r['question'][:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.15, help="longshot: outcome ask at or below this")
    ap.add_argument("--min-ratio", type=float, default=1.5, help="conservative prob / ask")
    ap.add_argument("--min-edge", type=float, default=0.02, help="conservative prob minus ask")
    ap.add_argument("--vol-haircut", type=float, default=0.15, help="shrink IV by this for tail outcomes")
    ap.add_argument("--max-days", type=float, default=60)
    ap.add_argument("--min-volume", type=float, default=5000)
    ap.add_argument("--min-depth", type=float, default=10, help="$ available at the ask (+1c)")
    ap.add_argument("--all", action="store_true", help="print every cheap contract, not just signals")
    ap.add_argument("--unparsed", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score:
        score()
    else:
        scan(args)


if __name__ == "__main__":
    main()
