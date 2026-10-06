#!/usr/bin/env python3
"""
base_atomic_census.py  --  follow the money: who earns atomic profits on Base, how much, and how crowded?

Instead of guessing a flash-loan strategy, measure the ones already being run:

  LIQUIDATIONS  every liquidation on Base lending protocols, found by event signature
                (so forks are included without hard-coding addresses):
                  Morpho Blue Liquidate, Aave-v3-style LiquidationCall (Aave, Seamless, ...),
                  Compound-v2-style LiquidateBorrow (Moonwell, ...), Euler v2 Liquidate
                -> bonus captured in USD (collateral seized minus debt repaid), gas paid,
                   liquidator (tx sender) and executor contract, per protocol and market
  FLASH LOANS   every flash loan: Morpho FlashLoan, Aave-v3-style FlashLoan, Balancer v2
                FlashLoan, Uniswap-v3-style Flash (Uniswap, Aerodrome Slipstream, Pancake ...)
                -> what the transaction did (liquidation / multi-swap arbitrage / single
                   swap / other), what the sender and executor contract netted in token
                   transfers (USD), minus gas
  NICHES        markets/strategies ranked by monthly $ with how many independent players
                take them and how concentrated: '$X/month split between 1-2 bots' is the
                shape worth building for; '$X/month, 40 bots' is not

Event signatures are copied from each protocol's source (morpho-blue EventsLib.sol,
aave-v3-core IPool.sol, compound-protocol CTokenInterfaces.sol, euler-vault-kit
Events.sol, balancer-v2 IVault.sol, uniswap v3-core IUniswapV3PoolEvents.sol).

Setup:  pip install curl_cffi        (BASE_RPC or ALCHEMY_KEY in .env, or --rpc URL)
Run:    python base_atomic_census.py --days 7
        python base_atomic_census.py --days 30 --max-flash-receipts 5000
Output: census_liquidations.csv, census_flashloans.csv, and the summary tables.

Caveats stated once: USD uses CURRENT DefiLlama prices (fine for ranking niches, rough
for old events); liquidation bonus ignores the swap cost of selling seized collateral
and protocol seize fees; flash-loan profit counts ERC-20 transfers only (not native ETH
or value routed to builders via coinbase transfers); flash receipts are sampled evenly
above --max-flash-receipts and scaled back up in the totals.
"""

import argparse
import csv
import os
import statistics
import sys
import time

from base_nav_arb import RPC, get_json, load_env, words, u256, addr_arg, BLOCK_SECONDS

LLAMA = "https://coins.llama.fi/prices/current"
WETH = "0x4200000000000000000000000000000000000006"

T = {  # topic0 = keccak(signature)
    "morpho_liq": "0xa4946ede45d0c6f06a0f5ce92c9ad3b4751452d2fe0e25010783bcab57a67e41",
    "aave_liq": "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286",
    "comp_liq": "0x298637f684da70674f26509b10f07ec2fbc77a335ab1e7d6215a4b2484d8bb52",
    "euler_liq": "0x8246cc71ab01533b5bebc672a636df812f10637ad720797319d5741d5ebb3962",
    "morpho_flash": "0xc76f1b4fe4396ac07a9fa55a415d4ca430e72651d37d3401f3bed7cb13fc4f12",
    "aave_flash": "0xefefaba5e921573100900a3ad9cf29f222d995fb3b6045797eaea7521bd8d6f0",
    "bal_flash": "0x0d7d75e01ab95780d3cd1c8ec0dd6c2ce19e3a20427eec8bf53283b6fb8e95f0",
    "univ3_flash": "0xbdbdb71d7860376ba52b25a5028beea23581364a40522f6bcfb86bb1f2dca633",
}
LIQ = {"morpho_liq": "Morpho Blue", "aave_liq": "Aave-style", "comp_liq": "Compound-style", "euler_liq": "Euler v2"}
FLASH = {"morpho_flash": "Morpho", "aave_flash": "Aave-style", "bal_flash": "Balancer v2", "univ3_flash": "UniV3-style"}
TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
SWAPS = {"0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",   # Uniswap v2 style
         "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67",   # Uniswap v3 / Slipstream
         "0xb3e2773606abfd36b5bd91394b3a54d1398336c65005baf7bf7a05efeffaf75b",   # Aerodrome v2
         "0x40e9cecb9f5f1f1c5b9c97dec2917b7ee92e57ba5563708daca94dd84ad7112f"}   # Uniswap v4
SEL = {"decimals": "0x313ce567", "symbol": "0x95d89b41", "idToMarketParams": "0x2c3c9157",
       "underlying": "0x6f307dc3", "comptroller": "0x5fe3b567", "exchangeRateStored": "0x182df0f5",
       "asset": "0x38d52e0f", "convertToAssets": "0x07a2d13a"}


def word(r, i=0, default=None):
    """i-th 32-byte word of an eth_call result; default when the call reverted or returned
    nothing (e.g. '0x' from a non-contract)."""
    w = words(r) if r else []
    return w[i] if len(w) > i else default


def addr(word_hex):
    return "0x" + word_hex[-40:].lower()


def data_words(log):
    return words(log["data"])


# ----------------------------------------------------------------- logs
def get_logs(rpc, topics, lo, hi, window):
    """All logs with topic0 in `topics` between blocks lo..hi; failed windows are split."""
    wins = [(a, min(a + window - 1, hi)) for a in range(lo, hi + 1, window)]
    out = []
    while wins:
        res = rpc.batch([("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(b), "topics": [topics]}])
                         for a, b in wins])
        retry = []
        for (a, b), r in zip(wins, res):
            if r is not None:
                out += r
            elif b - a >= 20:
                mid = (a + b) // 2
                retry += [(a, mid), (mid + 1, b)]
            else:
                print(f"  giving up on blocks {a}-{b}")
        if retry:
            print(f"  {len(retry) // 2} log windows too large or failed; splitting")
        wins = retry
    return out


def receipts(rpc, hashes):
    res = rpc.batch([("eth_getTransactionReceipt", [h]) for h in hashes])
    return {h: r for h, r in zip(hashes, res) if r}


def gas_eth(rc):
    used = int(rc.get("gasUsed", "0x0"), 16)
    price = int(rc.get("effectiveGasPrice", "0x0"), 16)
    l1 = int(rc.get("l1Fee", "0x0") or "0x0", 16)  # OP-stack L1 data fee
    return (used * price + l1) / 1e18


# ----------------------------------------------------------------- token metadata / prices
class Tokens:
    def __init__(self, rpc):
        self.rpc, self.dec, self.sym, self.px = rpc, {}, {}, {}

    def load(self, toks):
        toks = sorted({t for t in toks if t and t not in self.dec})
        if not toks:
            return
        res = self.rpc.calls([(t, SEL["decimals"], "latest") for t in toks] +
                             [(t, SEL["symbol"], "latest") for t in toks])
        n = len(toks)
        for i, t in enumerate(toks):
            self.dec[t] = word(res[i], default=18)
            self.sym[t] = _str(res[n + i]) if res[n + i] else t[:8]
        for i in range(0, n, 80):
            q = ",".join(f"base:{t}" for t in toks[i:i + 80])
            d = get_json(f"{LLAMA}/{q}") or {}
            for k, v in (d.get("coins") or {}).items():
                self.px[k.split(":", 1)[1].lower()] = v.get("price")

    def usd(self, tok, raw):
        p = self.px.get(tok)
        return None if p is None else raw / 10 ** self.dec.get(tok, 18) * p


def _str(h):
    try:
        b = bytes.fromhex(h[2:])
        if len(b) >= 96:
            return b[64:64 + int.from_bytes(b[32:64], "big")].decode("utf-8", "ignore")
        return b.rstrip(b"\0").decode("utf-8", "ignore")
    except Exception:
        return "?"


# ----------------------------------------------------------------- liquidations
def decode_liquidations(rpc, logs, tok):
    """-> list of dicts with protocol, market, debt/collateral token+raw amounts."""
    out, morpho_ids, comp_mtokens, euler_vaults = [], set(), set(), set()
    for l in logs:
        t0 = l["topics"][0]
        if t0 == T["morpho_liq"]:
            morpho_ids.add((l["address"].lower(), l["topics"][1]))
        elif t0 == T["comp_liq"]:
            w = data_words(l)
            comp_mtokens |= {l["address"].lower(), "0x%040x" % w[3]}
        elif t0 == T["euler_liq"]:
            euler_vaults |= {l["address"].lower(), "0x%040x" % data_words(l)[0]}
    # Morpho market params
    ids = sorted(morpho_ids)
    res = rpc.calls([(m, SEL["idToMarketParams"] + i[2:], "latest") for m, i in ids])
    mp = {i: ("0x%040x" % word(r, 0), "0x%040x" % word(r, 1)) for (m, i), r in zip(ids, res)
          if word(r, 1) is not None}
    # Compound-style mTokens: underlying (native-ETH markets have none -> WETH), exchange rate, decimals
    mts = sorted(comp_mtokens)
    res = rpc.calls([(m, SEL["underlying"], "latest") for m in mts] + [(m, SEL["exchangeRateStored"], "latest") for m in mts])
    under = {m: (addr(res[i]) if word(res[i]) else WETH) for i, m in enumerate(mts)}
    xrate = {m: word(res[len(mts) + i]) for i, m in enumerate(mts)}
    # Euler vaults: underlying asset
    evs = sorted(euler_vaults)
    res = rpc.calls([(v, SEL["asset"], "latest") for v in evs])
    easset = {v: addr(r) for v, r in zip(evs, res) if word(r)}
    euler_conv = []
    for l in logs:
        t0, w, base = l["topics"][0], data_words(l), {"tx": l["transactionHash"], "block": int(l["blockNumber"], 16)}
        if t0 == T["morpho_liq"] and l["topics"][1] in mp:
            loan, coll = mp[l["topics"][1]]
            out.append({**base, "protocol": "Morpho Blue", "market": l["topics"][1][:18], "debt_tok": loan,
                        "debt_raw": w[0], "coll_tok": coll, "coll_raw": w[2], "bad_debt": w[3] > 0})
        elif t0 == T["aave_liq"]:
            out.append({**base, "protocol": "Aave-style", "market": l["address"].lower()[:12],
                        "debt_tok": addr(l["topics"][2]), "debt_raw": w[0],
                        "coll_tok": addr(l["topics"][1]), "coll_raw": w[1], "bad_debt": False})
        elif t0 == T["comp_liq"]:
            m, cm = l["address"].lower(), "0x%040x" % w[3]
            coll_raw = w[4] * xrate[cm] // 10 ** 18 if xrate.get(cm) else None
            out.append({**base, "protocol": "Compound-style", "market": m[:12], "debt_tok": under[m],
                        "debt_raw": w[2], "coll_tok": under.get(cm), "coll_raw": coll_raw, "bad_debt": False})
        elif t0 == T["euler_liq"]:
            v, cv = l["address"].lower(), "0x%040x" % w[0]
            row = {**base, "protocol": "Euler v2", "market": v[:12], "debt_tok": easset.get(v), "debt_raw": w[1],
                   "coll_tok": easset.get(cv), "coll_raw": None, "bad_debt": False}
            euler_conv.append((row, cv, w[2]))
            out.append(row)
    if euler_conv:  # Euler pays collateral in vault shares: convert to assets
        res = rpc.calls([(cv, SEL["convertToAssets"] + u256(sh), "latest") for _, cv, sh in euler_conv])
        for (row, _, _), r in zip(euler_conv, res):
            row["coll_raw"] = word(r)
    tok.load([r["debt_tok"] for r in out] + [r["coll_tok"] for r in out])
    return out


# ----------------------------------------------------------------- flash loans
def flash_profit(rc, tok, eth_px):
    """Net ERC-20 flow (USD) into the tx sender + executor contract, minus gas; and the strategy."""
    me = {(rc.get("from") or "").lower(), (rc.get("to") or "").lower()}
    net, swaps, liq = {}, 0, False
    for l in rc.get("logs", []):
        t0 = l["topics"][0] if l["topics"] else ""
        if t0 in SWAPS:
            swaps += 1
        elif t0 in (T["morpho_liq"], T["aave_liq"], T["comp_liq"], T["euler_liq"]):
            liq = True
        elif t0 == TRANSFER and len(l["topics"]) == 3 and l["data"] not in ("0x", ""):
            src, dst, amt = addr(l["topics"][1]), addr(l["topics"][2]), int(l["data"][:66], 16)
            t = l["address"].lower()
            if dst in me:
                net[t] = net.get(t, 0) + amt
            if src in me:
                net[t] = net.get(t, 0) - amt
    usd, unpriced = 0.0, 0
    for t, raw in net.items():
        if raw == 0:
            continue
        v = tok.usd(t, raw)
        if v is None:
            unpriced += 1
        else:
            usd += v
    strategy = "liquidation" if liq else "arbitrage (2+ swaps)" if swaps >= 2 else \
        "single swap / leverage" if swaps == 1 else "other (refinance, collateral swap)"
    return usd, gas_eth(rc) * eth_px, strategy, unpriced


# ----------------------------------------------------------------- reporting
def concentration(players):
    """(# unique, share of the top player)."""
    if not players:
        return 0, 0.0
    counts = {}
    for p in players:
        counts[p] = counts.get(p, 0) + 1
    return len(counts), max(counts.values()) / len(players)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpc", help="Base RPC URL (default: BASE_RPC or ALCHEMY_KEY from .env)")
    ap.add_argument("--rpc-batch", type=int, default=20)
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--log-window", type=int, default=2000, help="blocks per eth_getLogs (auto-splits on failure)")
    ap.add_argument("--max-flash-receipts", type=int, default=3000,
                    help="sample at most this many flash-loan txs (receipts cost credits)")
    ap.add_argument("--niche-min-usd", type=float, default=300, help="min $/month to list a niche")
    ap.add_argument("--skip-liquidations", action="store_true")
    ap.add_argument("--skip-flash", action="store_true")
    args = ap.parse_args()

    load_env()
    url = args.rpc or os.environ.get("BASE_RPC") or (
        f"https://base-mainnet.g.alchemy.com/v2/{os.environ['ALCHEMY_KEY']}" if os.environ.get("ALCHEMY_KEY") else None)
    if not url:
        sys.exit("set BASE_RPC or ALCHEMY_KEY in .env, or pass --rpc")
    rpc = RPC(url, args.rpc_batch)
    head = rpc.block_number()
    lo = head - int(args.days * 86400 / BLOCK_SECONDS)
    month = 30 / args.days
    print(f"Base blocks {lo}..{head} ({args.days} days); monthly figures = observed x {month:.1f}")
    tok = Tokens(rpc)
    tok.load([WETH])
    eth_px = tok.px.get(WETH) or 0

    rows = []
    if not args.skip_liquidations:
        rows = liquidation_census(rpc, tok, eth_px, lo, head, month, args)
    if not args.skip_flash:
        flash_census(rpc, tok, eth_px, lo, head, month, args)
    niches_report(rows, args)
    print(f"\n{rpc.n} rpc calls ({rpc.spent:,} credits if paced). Rows in census_liquidations.csv / census_flashloans.csv")


def liquidation_census(rpc, tok, eth_px, lo, head, month, args):
    print("\n[1/2] liquidations: scanning logs ...", flush=True)
    llogs = get_logs(rpc, [T[k] for k in LIQ], lo, head, args.log_window)
    liqs = decode_liquidations(rpc, llogs, tok)
    rcs = receipts(rpc, sorted({r["tx"] for r in liqs}))
    for r in liqs:
        rc = rcs.get(r["tx"]) or {}
        d, c = tok.usd(r["debt_tok"], r["debt_raw"]), tok.usd(r["coll_tok"], r["coll_raw"] or 0) if r["coll_tok"] else None
        r.update(debt_sym=tok.sym.get(r["debt_tok"], "?"), coll_sym=tok.sym.get(r["coll_tok"], "?"),
                 repaid_usd=round(d, 2) if d is not None else None,
                 bonus_usd=round(c - d, 2) if (c is not None and d is not None) else None,
                 gas_usd=round(gas_eth(rc) * eth_px, 4) if rc else None,
                 liquidator=(rc.get("from") or "").lower(), executor=(rc.get("to") or "").lower())
    write_csv("census_liquidations.csv", liqs)
    print(f"  {len(liqs)} liquidations in {len({r['tx'] for r in liqs})} txs")
    groups = {}
    for r in liqs:
        groups.setdefault((r["protocol"], f"{r['coll_sym']}/{r['debt_sym']}", r["market"]), []).append(r)
    rows = []
    for (proto, pair, mkt), rs in groups.items():
        bonus = [r["bonus_usd"] for r in rs if r["bonus_usd"] is not None]
        n_players, top = concentration([r["liquidator"] for r in rs])
        rows.append({"protocol": proto, "pair": pair, "market": mkt, "n": len(rs),
                     "bonus_month": sum(bonus) * month, "median_bonus": statistics.median(bonus) if bonus else 0,
                     "median_gas": statistics.median([r["gas_usd"] for r in rs if r["gas_usd"] is not None] or [0]),
                     "players": n_players, "top_share": top})
    rows.sort(key=lambda x: -x["bonus_month"])
    print(f"\n  {'protocol':<15}{'collateral/debt':<22}{'n':>5}{'bonus $/mo':>12}{'median $':>10}"
          f"{'gas $':>8}{'players':>9}{'top %':>7}  market")
    for x in rows[:30]:
        print(f"  {x['protocol']:<15}{x['pair'][:21]:<22}{x['n']:>5}{x['bonus_month']:>12,.0f}"
              f"{x['median_bonus']:>10,.2f}{x['median_gas']:>8.3f}{x['players']:>9}{x['top_share']:>7.0%}  {x['market']}")
    by_proto = {}
    for r in liqs:
        b = by_proto.setdefault(r["protocol"], [0, 0.0, set()])
        b[0] += 1
        b[1] += r["bonus_usd"] or 0
        b[2].add(r["liquidator"])
    print("\n  by protocol:", "; ".join(f"{p}: {n} liqs, ${v * month:,.0f}/mo, {len(s)} liquidators"
                                       for p, (n, v, s) in sorted(by_proto.items(), key=lambda kv: -kv[1][1])))
    return rows


def flash_census(rpc, tok, eth_px, lo, head, month, args):
    print("\n[2/2] flash loans: scanning logs ...", flush=True)
    flogs = get_logs(rpc, [T[k] for k in FLASH], lo, head, args.log_window)
    provider = {}
    for l in flogs:
        provider.setdefault(l["transactionHash"], FLASH[next(k for k in FLASH if T[k] == l["topics"][0])])
    txs = sorted(provider)
    step = max(1, len(txs) // args.max_flash_receipts)
    sample = txs[::step]
    scale = len(txs) / max(len(sample), 1)
    print(f"  {len(flogs)} flash loans in {len(txs)} txs; reading {len(sample)} receipts (x{scale:.1f} to scale)")
    frcs = receipts(rpc, sample)
    touched = {l["address"].lower() for rc in frcs.values() for l in rc.get("logs", [])
               if l["topics"] and l["topics"][0] == TRANSFER}
    tok.load(touched)
    frows = []
    for h, rc in frcs.items():
        profit, gas, strat, unpriced = flash_profit(rc, tok, eth_px)
        frows.append({"tx": h, "block": int(rc["blockNumber"], 16), "provider": provider[h], "strategy": strat,
                      "profit_usd": round(profit, 2), "gas_usd": round(gas, 4), "net_usd": round(profit - gas, 2),
                      "unpriced_tokens": unpriced, "sender": (rc.get("from") or "").lower(),
                      "executor": (rc.get("to") or "").lower(), "status": int(rc.get("status", "0x1"), 16)})
    write_csv("census_flashloans.csv", frows)
    fg = {}
    for r in frows:
        fg.setdefault(r["strategy"], []).append(r)
    # Only searcher strategies have a meaningful 'profit'. Leverage loops, refinancing and
    # collateral swaps are users moving their own positions through shared routers: their
    # token flows are deposits/withdrawals, not P&L, so they're counted but not valued.
    searcher = ("arbitrage (2+ swaps)", "liquidation")
    print(f"\n  {'strategy':<36}{'txs/mo':>9}{'net $/mo':>12}{'median net':>12}{'executors':>10}{'top %':>7}")
    for s, rs in sorted(fg.items(), key=lambda kv: (kv[0] not in searcher, -len(kv[1]))):
        n_ex, top = concentration([r["executor"] for r in rs])
        if s in searcher:
            nets = [r["net_usd"] for r in rs]
            money = f"{sum(nets) * scale * month:>12,.0f}{statistics.median(nets):>12,.2f}"
        else:
            money = f"{'(users)':>12}{'':>12}"
        print(f"  {s:<36}{len(rs) * scale * month:>9,.0f}{money}{n_ex:>10}{top:>7.0%}")
    ex = {}
    for r in frows:
        if r["strategy"] in searcher:
            e = ex.setdefault(r["executor"], [0, 0.0, r["strategy"]])
            e[0] += 1
            e[1] += r["net_usd"]
    print("\n  flash-loan searchers (arbitrage/liquidation) by net $/month:")
    for e, (n, v, s) in sorted(ex.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"    {e}  {n * scale * month:>7,.0f} txs/mo  ${v * scale * month:>10,.0f}/mo  ({s})")
    if not ex:
        print("    none in the sample")
    print("  note: arbitrage profit routed out as native ETH or to the block builder isn't counted")


def niches_report(rows, args):
    print(f"\n=== NICHES: >= ${args.niche_min_usd:,.0f}/month taken by <= 3 liquidators ===")
    niches = [x for x in rows if x["bonus_month"] >= args.niche_min_usd and x["players"] <= 3]
    for x in niches:
        print(f"  {x['protocol']:<15}{x['pair'][:21]:<22}${x['bonus_month']:>9,.0f}/mo  "
              f"{x['players']} liquidator(s), top {x['top_share']:.0%}, median bonus ${x['median_bonus']:,.2f}  {x['market']}")
    if not niches:
        print("  none: no market with real money is held by 3 or fewer liquidators"
              + ("" if rows else " (no liquidations in this window)"))


if __name__ == "__main__":
    main()
