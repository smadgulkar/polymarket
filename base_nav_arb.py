#!/usr/bin/env python3
"""
base_nav_arb.py  --  are there flash-loanable NAV gaps on Base, and does anyone close them?

Trade being measured (atomic, no capital needed beyond gas):
  A  BUY-REDEEM  flash-borrow asset -> buy vault shares on a DEX below their
                 instant-redemption value -> redeem (ERC-4626 redeem) -> repay
  B  MINT-SELL   flash-borrow asset -> deposit into the vault (ERC-4626) -> sell
                 the shares on a DEX above mint cost -> repay
Generic MEV bots cover the big vaults; the long tail needs per-vault integration.
Whether that long tail is actually unharvested is what this measures.

Method:
  1. DISCOVER  vaults = contracts emitting ERC-4626 Deposit events (last 24h), then
               their DEX pools via GeckoTerminal (cached); keep pools pairing a
               vault share with its own asset()
  2. REPLAY    every --step blocks over the last --hours, read at that block:
               previewRedeem / previewDeposit / maxDeposit from the vault and the
               pool state (v3/Slipstream: slot0 + liquidity as virtual reserves;
               Uniswap-v2 style: getReserves; Aerodrome v2: getAmountOut ladder)
  3. SIZE      optimal trade size and profit per direction, minus gas and the
               flash-loan fee, in USD (DefiLlama prices)
  4. EPISODES  consecutive profitable samples -> refine exact open/close blocks by
               binary search -> pull the closing block's logs for that pool to see
               WHO closed it (a repeat address = someone already harvests it)

Setup:  pip install curl_cffi        (ALCHEMY_KEY=... in .env, or --rpc URL)
Run:    python base_nav_arb.py --now              # snapshot: every candidate pool, current gap
        python base_nav_arb.py --hours 24         # replay last 24h, 1 sample/min
        python base_nav_arb.py --hours 72 --step 150 --min-profit 1
Output: base_nav_episodes.csv (one row per profitable episode) + summary per pool.

Reading it: no episodes = nothing to harvest on these pools. Episodes that close
in 1-2 blocks with the same closer = already taken. Episodes that sit open for
minutes with $ that beats gas = the opening.

Caveats stated once: v3 sizing treats liquidity as constant up to the NAV price
(ignores tick crossings: rough at large sizes); previewRedeem ignores vault
liquidity limits (e.g. Morpho vaults capped by market liquidity) and pauses --
check the vault before building; only share/asset pools are covered (no
multi-hop routes, no Balancer/Uniswap v4 yet).
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

try:
    from curl_cffi import requests
except ImportError:
    sys.exit("curl_cffi not installed. Run:  pip install curl_cffi")

SESSION = requests.Session(impersonate="chrome")
GECKO = "https://api.geckoterminal.com/api/v2/networks/base"
LLAMA = "https://coins.llama.fi/prices/current"
POOL_CACHE = "base_pools_cache.json"
VAULT_CACHE = "base_vault_pools_cache.json"
OUT = "base_nav_episodes.csv"
BLOCK_SECONDS = 2.0
V2_FEE = {"uniswap-v2-base": 0.003, "sushiswap-v2-base": 0.003, "pancakeswap-v2-base": 0.0025}
PROBE = "0x000000000000000000000000000000000000dEaD"
DEPOSIT_TOPIC = "0xdcbc1c05240f31ff3ad067ef1ee35ce4997762752e3a095284754544f4c709d7"  # ERC-4626 Deposit

SEL = {"asset": "0x38d52e0f", "decimals": "0x313ce567", "symbol": "0x95d89b41",
       "previewRedeem": "0x4cdad506", "previewDeposit": "0xef8b30f7", "maxDeposit": "0x402d267d",
       "token0": "0x0dfe1681", "token1": "0xd21220a7", "slot0": "0x3850c7bd",
       "liquidity": "0x1a686502", "fee": "0xddca3f43", "getAmountOut": "0xf140a35a",
       "stable": "0x22be3de1", "getReserves": "0x0902f1ac"}


# ----------------------------------------------------------------- helpers
def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def u256(x):
    return format(int(x), "064x")


def addr_arg(a):
    return a.lower().replace("0x", "").rjust(64, "0")


def words(hexstr):
    h = (hexstr or "0x")[2:]
    return [int(h[i:i + 64], 16) for i in range(0, len(h) - len(h) % 64, 64)]


def decode_str(hexstr):
    try:
        h = bytes.fromhex(hexstr[2:])
        if len(h) >= 96:
            n = int.from_bytes(h[32:64], "big")
            return h[64:64 + n].decode("utf-8", "ignore")
        return h.rstrip(b"\0").decode("utf-8", "ignore")
    except Exception:
        return "?"


# Free-plan throughput and per-method costs, from each provider's published docs:
#   Infura  500 credits/s, 3M credits/day; eth_call/eth_blockNumber/getTx 80, eth_getLogs 255
#   Alchemy 300 CU/s, 30M CU/month;       eth_call 26, eth_getLogs 60
PROVIDERS = {
    "infura.io": {"per_s": 500, "default": 80, "cost": {"eth_getLogs": 255}, "budget": (3_000_000, "day")},
    "alchemy.com": {"per_s": 300, "default": 26, "cost": {"eth_getLogs": 60}, "budget": (30_000_000, "month")},
}


class RPC:
    def __init__(self, url, chunk=50, per_s=None):
        self.url, self.chunk, self.n, self.spent = url, chunk, 0, 0
        self.plan = next((p for host, p in PROVIDERS.items() if host in url), None)
        self.per_s = per_s or (self.plan["per_s"] if self.plan else None)

    def cost(self, method):
        return self.plan["cost"].get(method, self.plan["default"]) if self.plan else 1

    def pace(self, idx_reqs):
        """Trim a chunk to fit one second of throughput, and wait long enough after sending it."""
        if not self.per_s:
            return idx_reqs, 0.0
        budget = 0.9 * self.per_s
        out, spent = [], 0
        for k, m in idx_reqs:
            if out and spent + self.cost(m) > budget:
                break
            out.append((k, m))
            spent += self.cost(m)
        return out, spent / budget

    def batch(self, reqs):
        """reqs: list of (method, params). Returns list of result-or-None, same order.
        Per-item rate-limit errors are retried; reverts come back as None."""
        out = [None] * len(reqs)
        todo = list(range(len(reqs)))
        for attempt in range(8):
            retry = []
            i = 0
            while i < len(todo):
                fit, wait = self.pace([(k, reqs[k][0]) for k in todo[i:i + self.chunk]])
                idx = [k for k, _ in fit]
                i += len(idx)
                body = [{"jsonrpc": "2.0", "id": j, "method": reqs[k][0], "params": reqs[k][1]}
                        for j, k in enumerate(idx)]
                try:
                    r = SESSION.post(self.url, json=body, timeout=60)
                    res = r.json() if r.status_code not in (402, 429) else None
                    if r.status_code == 402:
                        sys.exit("RPC provider: plan credit limit reached (HTTP 402). Wait for the reset "
                                 "or use another provider via --rpc.")
                except Exception:  # network error or non-JSON body: retry
                    res = None
                time.sleep(wait)
                if not isinstance(res, list):
                    if isinstance(res, dict) and attempt == 0:
                        print(f"  rpc batch of {len(body)} rejected: {str(res.get('error'))[:100]}"
                              f" -- try a smaller --rpc-batch")
                    retry += idx
                    continue
                self.n += len(body)
                self.spent += sum(self.cost(reqs[k][0]) for k in idx)
                # match responses to requests by id (as text: some providers echo ids as strings),
                # falling back to order when every response lacks an id
                by_id = {str(x.get("id")): x for x in res if isinstance(x, dict) and x.get("id") is not None}
                if len(by_id) == len(idx) and all(str(j) in by_id for j in range(len(idx))):
                    matched = [(idx[j], by_id[str(j)]) for j in range(len(idx))]
                elif not by_id and len(res) == len(idx) and all(isinstance(x, dict) for x in res):
                    matched = list(zip(idx, res))
                else:
                    if self.chunk > 1:
                        self.chunk = max(1, self.chunk // 2)
                    print(f"  rpc batch error, raw response: {str(res)[:300]}; retrying with batches of {self.chunk}")
                    retry += idx
                    continue
                for k, x in matched:
                    err = x.get("error") or {}
                    if "result" in x:
                        out[k] = x["result"]
                    elif "rate" in str(err.get("message", "")).lower() or err.get("code") in (429, -32005, -32016):
                        retry.append(k)
                    elif reqs[k][0] != "eth_call" and not getattr(self, "_warned", False):
                        # reverting eth_calls are normal (non-vault tokens); anything else is worth seeing
                        print(f"  rpc {reqs[k][0]} error: {str(err)[:150]}")
                        self._warned = True
            if not retry:
                break
            todo = retry
            time.sleep(min(2 ** attempt, 20))
        return out

    def calls(self, triples):
        """triples: list of (to, data, block_or_'latest')."""
        return self.batch([("eth_call", [{"to": t, "data": d}, b if b == "latest" else hex(b)])
                           for t, d, b in triples])

    def block_number(self):
        return int(self.batch([("eth_blockNumber", [])])[0], 16)


def get_json(url, params=None):
    for attempt in range(6):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            return r.json() if r.status_code == 200 else None
        except Exception:
            time.sleep(2 + attempt)
    return None


# ----------------------------------------------------------------- discovery
def discover(rpc, args):
    """Vault-first: vault pools are the long tail, invisible in 'top pools' lists.
    1) every contract that emitted an ERC-4626 Deposit event in the last --discover-hours
    2) keep those whose asset() answers  3) ask GeckoTerminal for each vault's DEX pools."""
    if os.path.exists(POOL_CACHE) and not args.refresh:
        return json.load(open(POOL_CACHE, encoding="utf-8"))
    head = rpc.block_number()
    lo = head - int(args.discover_hours * 3600 / BLOCK_SECONDS)
    wins = [(b, min(b + args.log_window - 1, head)) for b in range(lo, head + 1, args.log_window)]
    logs = rpc.batch([("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(b), "topics": [DEPOSIT_TOPIC]}])
                      for a, b in wins])
    counts = {}
    for chunk in logs:
        for l in chunk or []:
            a = l["address"].lower()
            counts[a] = counts.get(a, 0) + 1
    if any(x is None for x in logs):
        print(f"  warning: {sum(x is None for x in logs)}/{len(wins)} log windows failed -- try smaller --log-window")
    emitters = sorted(counts, key=lambda a: -counts[a])
    res = rpc.calls([(a, SEL["asset"], "latest") for a in emitters])
    vaults = {a: "0x" + r[-40:] for a, r in zip(emitters, res) if r and len(r) == 66 and 0 < int(r, 16) < 2 ** 160}
    print(f"  {len(emitters)} Deposit emitters over {args.discover_hours}h, {len(vaults)} answer asset(); "
          f"looking up DEX pools for up to {args.max_vaults}")
    # per-vault lookups are slow (GeckoTerminal ~30 req/min): checkpoint them so an
    # interrupted run resumes instead of starting over
    seen = json.load(open(VAULT_CACHE, encoding="utf-8")) if os.path.exists(VAULT_CACHE) else {}
    todo = [v for v in list(vaults)[:args.max_vaults] if v not in seen]
    if seen:
        print(f"  resuming: {len(seen)} vaults already looked up, {len(todo)} to go")
    for i, v in enumerate(todo):
        d = get_json(f"{GECKO}/tokens/{v}/pools", {"page": 1})
        found = {}
        for p in (d or {}).get("data") or []:
            a, rel = p["attributes"], p["relationships"]
            found[a["address"].lower()] = {
                "dex": rel.get("dex", {}).get("data", {}).get("id", "?"), "name": a["name"],
                "tvl": float(a.get("reserve_in_usd") or 0),
                "ta": rel["base_token"]["data"]["id"].split("_", 1)[1].lower(),
                "tb": rel["quote_token"]["data"]["id"].split("_", 1)[1].lower()}
        if d is not None:  # don't checkpoint a failed lookup as "no pools"
            seen[v] = found
        time.sleep(2.2)  # GeckoTerminal free tier: ~30 req/min
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            json.dump(seen, open(VAULT_CACHE, "w", encoding="utf-8"))
            print(f"  {len(seen)} vaults checked, {sum(len(x) for x in seen.values())} pools found")
    pools = {a: p for found in seen.values() for a, p in found.items()}
    meta = classify(rpc, pools)
    paired = {c["share"] for c in meta}
    other = len(({p["ta"] for p in pools.values()} | {p["tb"] for p in pools.values()}) & set(vaults) - paired)
    print(f"  {other} vaults trade only against a token other than their asset (multi-hop; not covered yet)")
    json.dump(meta, open(POOL_CACHE, "w", encoding="utf-8"), indent=1)
    print(f"cached {len(meta)} share/asset pools in {POOL_CACHE} (--refresh to redo)")
    return meta


def classify(rpc, pools):
    """pools: {address: {dex,name,tvl,ta,tb}} -> enriched share/asset candidates."""
    toks = sorted({p["ta"] for p in pools.values()} | {p["tb"] for p in pools.values()})
    res = rpc.calls([(t, SEL["asset"], "latest") for t in toks])
    asset = {t: "0x" + r[-40:] for t, r in zip(toks, res)
             if r and len(r) == 66 and 0 < int(r, 16) < 2 ** 160}
    cands = []
    for addr, p in pools.items():
        if asset.get(p["ta"]) == p["tb"]:
            share, base = p["ta"], p["tb"]
        elif asset.get(p["tb"]) == p["ta"]:
            share, base = p["tb"], p["ta"]
        else:
            continue
        cands.append({"pool": addr, "dex": p["dex"], "name": p["name"], "tvl": p["tvl"],
                      "share": share, "asset": base})
    print(f"  {len(pools)} pools, {len(toks)} tokens, {len(asset)} ERC-4626 tokens, "
          f"{len(cands)} share/asset pools")
    return enrich(rpc, cands)


def enrich(rpc, cands):
    """Token decimals/symbols, pool type, token order, static fee."""
    out = []
    for c in cands:
        q = [(c["share"], SEL["decimals"]), (c["asset"], SEL["decimals"]), (c["share"], SEL["symbol"]),
             (c["asset"], SEL["symbol"]), (c["pool"], SEL["token0"]), (c["pool"], SEL["slot0"]),
             (c["pool"], SEL["liquidity"]), (c["pool"], SEL["fee"]), (c["pool"], SEL["stable"]),
             (c["pool"], SEL["getReserves"])]
        r = rpc.calls([(t, d, "latest") for t, d in q])
        if not (r[0] and r[1] and r[4]):
            continue
        c["share_dec"], c["asset_dec"] = words(r[0])[0], words(r[1])[0]
        c["share_sym"], c["asset_sym"] = decode_str(r[2]), decode_str(r[3])
        c["share_is_t0"] = ("0x" + r[4][-40:]).lower() == c["share"]
        if c["dex"] == "aerodrome-base" and r[8]:
            c["kind"], c["fee"] = "solidly", None  # fee comes out of getAmountOut
        elif r[5] and r[6] and len(words(r[5])) >= 2:
            c["kind"], c["fee"] = "v3", (words(r[7])[0] / 1e6 if r[7] else 0.003)
        elif r[9] and c["dex"] in V2_FEE:
            c["kind"], c["fee"] = "v2", V2_FEE[c["dex"]]
        else:
            continue
        out.append(c)
    return out


# ----------------------------------------------------------------- evaluation
def optimal(r_in, r_out, gamma, c):
    """Constant-product (reserves r_in/r_out, fee factor gamma). Buy out-tokens with in-tokens
    where each in-token costs c out-token-equivalents. Returns (size_in, out, profit_in_out_units)."""
    if r_in <= 0 or r_out <= 0 or c <= 0:
        return 0.0, 0.0, 0.0
    d = (math.sqrt(gamma * r_in * r_out / c) - r_in) / gamma
    if d <= 0:
        return 0.0, 0.0, 0.0
    out = r_out * gamma * d / (r_in + gamma * d)
    return d, out, out - c * d


def state_calls(c, block, ladder_sizes=None):
    s1, a1 = 10 ** c["share_dec"], 10 ** c["asset_dec"]
    q = [(c["share"], SEL["previewRedeem"] + u256(s1)), (c["share"], SEL["previewDeposit"] + u256(a1)),
         (c["share"], SEL["maxDeposit"] + addr_arg(PROBE))]
    if c["kind"] == "v3":
        q += [(c["pool"], SEL["slot0"]), (c["pool"], SEL["liquidity"])]
    elif c["kind"] == "v2":
        q += [(c["pool"], SEL["getReserves"])]
    else:  # solidly: ladder of getAmountOut in both directions
        for amt in ladder_sizes["asset"]:
            q.append((c["pool"], SEL["getAmountOut"] + u256(amt) + addr_arg(c["asset"])))
        for amt in ladder_sizes["share"]:
            q.append((c["pool"], SEL["getAmountOut"] + u256(amt) + addr_arg(c["share"])))
    return [(t, d, block) for t, d in q]


def ladder(c, px_asset):
    """Input sizes from ~$20 to ~$2M, in raw units."""
    usd = [20 * 3 ** i for i in range(12)]
    a = [int(u / px_asset * 10 ** c["asset_dec"]) for u in usd]
    return {"asset": a, "share": [int(x / 10 ** c["asset_dec"] * 10 ** c["share_dec"]) for x in a]}


def evaluate(c, res, px_asset, args, lad=None):
    """-> list of (direction, profit_usd_net, size_usd, nav, dex_px) from one block's call results."""
    if not res[0] or not res[1]:
        return None
    nav = words(res[0])[0] / 10 ** c["asset_dec"]                   # assets per share on redeem
    spa = words(res[1])[0] / 10 ** c["share_dec"]                   # shares per asset on deposit
    if nav <= 0 or spa <= 0:
        return None
    mint_cost = 1 / spa                                              # assets per share on mint
    cap_dep = words(res[2])[0] / 10 ** c["asset_dec"] if res[2] else float("inf")
    cost = args.gas_usd
    out = []
    if c["kind"] in ("v3", "v2"):
        if c["kind"] == "v3":
            if not (res[3] and res[4]):
                return None
            sqrtp = words(res[3])[0] / 2 ** 96
            liq = words(res[4])[0]
            if sqrtp <= 0 or liq <= 0:
                return None
            r0, r1 = liq / sqrtp, liq * sqrtp                        # virtual reserves, raw units
        else:
            if not res[3]:
                return None
            r0, r1 = words(res[3])[:2]
        d0, d1 = (c["share_dec"], c["asset_dec"]) if c["share_is_t0"] else (c["asset_dec"], c["share_dec"])
        x0, x1 = r0 / 10 ** d0, r1 / 10 ** d1
        rs, ra = (x0, x1) if c["share_is_t0"] else (x1, x0)
        gamma = 1 - c["fee"]
        dex_px = ra / rs
        # A: pay asset, get share, redeem at nav. In-token = asset, costs 1/nav share-equivalents
        d, got, prof = optimal(ra, rs, gamma, 1 / nav)
        if d > 0:
            pa = prof * nav * px_asset - d * px_asset * args.flash_fee
            out.append(("buy-redeem", pa - cost, d * px_asset, nav, dex_px))
        # B: mint at mint_cost, sell shares for asset. In-token = share, costs mint_cost assets
        d, got, prof = optimal(rs, ra, gamma, mint_cost)
        if d > 0:
            d = min(d, cap_dep / mint_cost)
            got = ra * gamma * d / (rs + gamma * d)
            pb = (got - d * mint_cost) * px_asset - d * mint_cost * px_asset * args.flash_fee
            out.append(("mint-sell", pb - cost, d * mint_cost * px_asset, nav, dex_px))
    else:
        na = len(lad["asset"])
        outs_a, outs_s = res[3:3 + na], res[3 + na:3 + 2 * na]
        best_a = best_b = None
        dex_px = None
        for amt, r in zip(lad["asset"], outs_a):
            if not r:
                continue
            ain, sout = amt / 10 ** c["asset_dec"], words(r)[0] / 10 ** c["share_dec"]
            dex_px = dex_px or (ain / sout if sout else None)
            p = (sout * nav - ain - ain * args.flash_fee) * px_asset - cost
            if best_a is None or p > best_a[0]:
                best_a = (p, ain * px_asset)
        for amt, r in zip(lad["share"], outs_s):
            if not r:
                continue
            sin, aout = amt / 10 ** c["share_dec"], words(r)[0] / 10 ** c["asset_dec"]
            if sin * mint_cost > cap_dep:
                continue
            p = (aout - sin * mint_cost * (1 + args.flash_fee)) * px_asset - cost
            if best_b is None or p > best_b[0]:
                best_b = (p, sin * mint_cost * px_asset)
        if best_a:
            out.append(("buy-redeem", best_a[0], best_a[1], nav, dex_px or 0))
        if best_b:
            out.append(("mint-sell", best_b[0], best_b[1], nav, dex_px or 0))
    return out


def prices(assets):
    q = ",".join(f"base:{a}" for a in sorted(set(assets)))
    d = get_json(f"{LLAMA}/{q}") or {}
    return {k.split(":", 1)[1].lower(): v["price"] for k, v in (d.get("coins") or {}).items()}


# ----------------------------------------------------------------- replay
def profit_at(rpc, c, block, px, args, lad):
    res = rpc.calls(state_calls(c, block, lad))
    ev = evaluate(c, res, px, args, lad)
    return {d: p for d, p, *_ in ev} if ev else {}


def refine_open(rpc, c, direction, lo_bad, hi_good, px, args, lad):
    """First profitable block in (lo_bad, hi_good]."""
    while hi_good - lo_bad > 1:
        mid = (lo_bad + hi_good) // 2
        if profit_at(rpc, c, mid, px, args, lad).get(direction, -1) > args.min_profit:
            hi_good = mid
        else:
            lo_bad = mid
    return hi_good


def refine_close(rpc, c, direction, last_good, first_bad, px, args, lad):
    """First unprofitable block in (last_good, first_bad]."""
    lo, hi = last_good, first_bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if profit_at(rpc, c, mid, px, args, lad).get(direction, -1) > args.min_profit:
            lo = mid
        else:
            hi = mid
    return hi


def closers(rpc, pool, block):
    logs = rpc.batch([("eth_getLogs", [{"address": pool, "fromBlock": hex(block), "toBlock": hex(block)}])])[0] or []
    txs = sorted({l["transactionHash"] for l in logs})
    if not txs:
        return []
    res = rpc.batch([("eth_getTransactionByHash", [h]) for h in txs])
    return [(r or {}).get("from", "?") for r in res]


def replay(rpc, cands, px, args):
    head = rpc.block_number()
    blocks = list(range(head - int(args.hours * 3600 / BLOCK_SECONDS), head + 1, args.step))
    lads = {c["pool"]: ladder(c, px[c["asset"]]) for c in cands if c["kind"] == "solidly"}
    print(f"replaying {len(blocks)} samples x {len(cands)} pools, blocks {blocks[0]}..{blocks[-1]} "
          f"({args.hours}h, every {args.step} blocks)")
    series = {}  # (pool, dir) -> list of (block, profit, size, nav, dex_px)
    t0 = time.time()
    for bi, b in enumerate(blocks):
        triples, spans = [], []
        for c in cands:
            q = state_calls(c, b, lads.get(c["pool"]))
            spans.append((c, len(triples), len(q)))
            triples += q
        res = rpc.calls(triples)
        for c, start, n in spans:
            ev = evaluate(c, res[start:start + n], px[c["asset"]], args, lads.get(c["pool"]))
            for d, p, size, nav, dpx in ev or []:
                series.setdefault((c["pool"], d), []).append((b, p, size, nav, dpx))
        if bi % 60 == 0:
            print(f"  {bi}/{len(blocks)} samples  ({rpc.n} rpc calls, {time.time() - t0:.0f}s)")
    return blocks, series, lads


def budget_ok(rpc, cands, px, args):
    """Estimate the replay's cost against the provider's free-plan budget before spending it."""
    samples = int(args.hours * 3600 / BLOCK_SECONDS / args.step) + 1
    per_sample = sum(len(state_calls(c, 0, ladder(c, px[c["asset"]]) if c["kind"] == "solidly" else None))
                     for c in cands)
    calls = samples * per_sample
    credits = calls * rpc.cost("eth_call")
    secs = credits / (0.9 * rpc.per_s) if rpc.per_s else 0
    print(f"replay estimate: {calls:,} calls, {credits:,} credits, ~{secs / 60:.0f} min at the paced rate "
          f"(+ episode refinement)")
    if rpc.plan:
        cap, period = rpc.plan["budget"]
        if credits > 0.8 * cap and not args.force:
            print(f"  that is {credits / cap:.0%} of the free plan's {cap:,}/{period}. Shrink it with fewer "
                  f"--hours or a larger --step (or pass --force).")
            return False
    return True


def episodes(rpc, cands, blocks, series, px, lads, args):
    by_pool = {c["pool"]: c for c in cands}
    eps = []
    for (pool, d), pts in series.items():
        run = []
        for b, p, size, nav, dpx in pts + [(None, -1, 0, 0, 0)]:
            if b is not None and p > args.min_profit:
                run.append((b, p, size))
                continue
            if run:
                c = by_pool[pool]
                first, last = run[0][0], run[-1][0]
                opened = first
                if args.refine and first - args.step >= blocks[0]:
                    opened = refine_open(rpc, c, d, first - args.step, first, px[c["asset"]], args, lads.get(pool))
                close = None
                if args.refine and b is not None:
                    close = refine_close(rpc, c, d, last, b, px[c["asset"]], args, lads.get(pool))
                who = closers(rpc, pool, close) if close else []
                eps.append({"pool": pool, "pair": f"{c['share_sym']}/{c['asset_sym']}", "dex": c["dex"],
                            "direction": d, "open_block": opened, "close_block": close or "",
                            "blocks_open": (close - opened) if close else "",
                            "seconds_open": round(((close or last) - opened) * BLOCK_SECONDS),
                            "max_profit_usd": round(max(x[1] for x in run), 2),
                            "size_usd_at_max": round(max(run, key=lambda x: x[1])[2], 2),
                            "closed_by": ";".join(sorted(set(who)))})
                run = []
    return eps


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpc", help="Base RPC URL (default: Alchemy from ALCHEMY_KEY in .env)")
    ap.add_argument("--rpc-batch", type=int, default=50, help="JSON-RPC calls per HTTP batch")
    ap.add_argument("--rpc-per-s", type=float,
                    help="credits/CU per second to pace to (default: provider's free-plan limit)")
    ap.add_argument("--force", action="store_true", help="run even if the estimate exceeds the plan budget")
    ap.add_argument("--now", action="store_true", help="snapshot of every candidate pool at the head block")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--step", type=int, default=30, help="blocks between samples (30 = 1 min on Base)")
    ap.add_argument("--min-profit", type=float, default=0.5, help="net USD to count as an opportunity")
    ap.add_argument("--gas-usd", type=float, default=0.05, help="cost of one flash-loan arb tx on Base")
    ap.add_argument("--flash-fee", type=float, default=0.0,
                    help="flash loan fee (0: Morpho/Balancer; 0.0005: Aave v3)")
    ap.add_argument("--min-tvl", type=float, default=5000, help="skip pools with less USD liquidity")
    ap.add_argument("--discover-hours", type=float, default=24, help="look back this far for vault Deposit events")
    ap.add_argument("--log-window", type=int, default=2000, help="blocks per eth_getLogs call")
    ap.add_argument("--max-vaults", type=int, default=300, help="vaults to look up on GeckoTerminal")
    ap.add_argument("--refresh", action="store_true", help="re-run pool discovery")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    help="skip exact open/close block search and closer lookup")
    args = ap.parse_args()

    load_env()
    url = args.rpc or os.environ.get("BASE_RPC") or (
        f"https://base-mainnet.g.alchemy.com/v2/{os.environ['ALCHEMY_KEY']}" if os.environ.get("ALCHEMY_KEY") else None)
    if not url:
        sys.exit("set ALCHEMY_KEY in .env or pass --rpc")
    rpc = RPC(url, args.rpc_batch, args.rpc_per_s)
    if rpc.plan:
        print(f"provider pacing: {rpc.per_s:.0f} credits/s ({url.split('/')[2]})")
    cands = [c for c in discover(rpc, args) if c["tvl"] >= args.min_tvl]
    px = prices([c["asset"] for c in cands])
    cands = [c for c in cands if c["asset"] in px]
    print(f"{len(cands)} share/asset pools with TVL >= ${args.min_tvl:,.0f} and a USD price")
    for c in sorted(cands, key=lambda c: -c["tvl"]):
        print(f"  {c['share_sym']:>14}/{c['asset_sym']:<8} {c['dex']:<22} {c['kind']:<7} tvl ${c['tvl']:>12,.0f}  {c['pool']}")
    if not cands:
        return

    if args.now:
        head = rpc.block_number()
        print(f"\nsnapshot at block {head}: net USD at optimal size (negative = no gap)")
        for c in cands:
            lad = ladder(c, px[c["asset"]]) if c["kind"] == "solidly" else None
            ev = evaluate(c, rpc.calls(state_calls(c, head, lad)), px[c["asset"]], args, lad)
            if not ev:
                print(f"  {c['share_sym']:>14}/{c['asset_sym']:<8} no data (vault call reverted?)")
                continue
            nav, dpx = ev[0][3], ev[0][4]
            gaps = "  ".join(f"{d}: ${p:+,.2f} @ ${s:,.0f}" for d, p, s, *_ in ev)
            print(f"  {c['share_sym']:>14}/{c['asset_sym']:<8} nav {nav:.6f}  dex {dpx:.6f} "
                  f"({(dpx / nav - 1) * 100:+.3f}%)  {gaps}")
        return

    if not budget_ok(rpc, cands, px, args):
        return
    blocks, series, lads = replay(rpc, cands, px, args)
    eps = episodes(rpc, cands, blocks, series, px, lads, args)
    if eps:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eps[0].keys()))
            w.writeheader()
            w.writerows(eps)
    print(f"\n=== {len(eps)} profitable episodes (net > ${args.min_profit}) over {args.hours}h ===")
    print(f"{'pair':<24}{'direction':<12}{'samples+':>9}{'episodes':>9}{'max $':>9}{'median s open':>15}  top closer")
    for c in cands:
        for d in ("buy-redeem", "mint-sell"):
            pts = series.get((c["pool"], d), [])
            mine = [e for e in eps if e["pool"] == c["pool"] and e["direction"] == d]
            pos = sum(1 for x in pts if x[1] > args.min_profit)
            if not pts:
                continue
            who = [w for e in mine for w in e["closed_by"].split(";") if w]
            top = max(set(who), key=who.count) if who else ""
            print(f"{c['share_sym'] + '/' + c['asset_sym']:<24}{d:<12}{pos:>4}/{len(pts):<4}{len(mine):>9}"
                  f"{max([e['max_profit_usd'] for e in mine], default=0):>9.2f}"
                  f"{statistics.median([e['seconds_open'] for e in mine]) if mine else 0:>15.0f}"
                  f"  {top}{f' (x{who.count(top)})' if top else ''}")
    print(f"\n{rpc.n} rpc calls. Episodes in {OUT}." if eps else f"\n{rpc.n} rpc calls.")


if __name__ == "__main__":
    main()
