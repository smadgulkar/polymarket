#!/usr/bin/env python3
"""
polymarket_jev_tags.py  --  Measurement #2: which longshots, if any, are underpriced?

Question: the blind 2-20c longshot bucket in longshot_hv_samples.csv realizes
BELOW its price (favourite-longshot bias). Is there a SUBSET, defined by what
the market's question + rules text says, that realizes ABOVE its price?

Method: for every market with a priced sample, pull the question + rules
(Gamma `description`), send it to Jev (typesafe-ai/jev via Vercel AI Gateway)
with a fixed set of typed questions, cache the answers, then tabulate realized
YES vs mean price per tag value, per horizon, with Wilson 95% CIs.

Setup:
    pip install curl_cffi
    put AI_GATEWAY_API_KEY=... in .env (or TYPESAFE_API_KEY=... to call
    api.typesafe.ai directly)
Run:
    python polymarket_jev_tags.py                 # fetch text, tag, analyze
    python polymarket_jev_tags.py --dry-run       # print one Jev payload, no calls
    python polymarket_jev_tags.py --analyze-only  # re-analyze cached tags
    python polymarket_jev_tags.py --samples other_samples.csv --max-price 0.10

Output: jev_market_text.json (Gamma cache), jev_tags.json (Jev cache, resumable),
        jev_tags.csv (one row per market), EV table per tag on stdout.

Decision rule (same bar as #1): a tag value is a candidate edge only if, at a
given horizon, n >= 150, realized YES > mean price, and the Wilson CI LOWER
bound is above mean price. Anything short of that is noise -- with ~200 tagged
markets you are hunting for a hypothesis to test on the deep scan, not a
strategy. Tags were chosen BEFORE looking at results; don't add tags until one
"works" -- that's how you overfit a 20:1 lottery.

Caveats: prices are last-trade, not the ask; horizons of the same market are
correlated (report per horizon, never pool them); Jev reads today's rules
text, which Polymarket sometimes edits after creation.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

try:
    from curl_cffi import requests
except ImportError:
    sys.exit("curl_cffi not installed. Run:  pip install curl_cffi")

GAMMA = "https://gamma-api.polymarket.com"
GATEWAY_URL = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
SESSION = requests.Session(impersonate="chrome")

TEXT_CACHE = "jev_market_text.json"
TAG_CACHE = "jev_tags.json"
TAG_CSV = "jev_tags.csv"
MAX_STATE_CHARS = 6000  # ~1.5k tokens; well inside Jev's 32K window

# Fixed BEFORE looking at outcomes. Each is a hypothesis about why a longshot
# might be mispriced; see the docstring's overfitting warning before adding more.
QUESTIONS = {
    "needs_action": {
        "type": "noul",
        "instructions": "Does YES require some new event or action to happen, as opposed to "
                        "resolving YES if the current situation simply continues?",
        "criteria": {"true": "YES needs something to happen or change",
                     "false": "YES if things stay as they are"},
    },
    "deadline": {
        "type": "noul",
        "instructions": "Is this an 'X happens by <date>' market, where YES can occur at any "
                        "time before a deadline?",
    },
    "timing": {
        "type": "choice",
        "instructions": "How is the deciding moment scheduled?",
        "criteria": {
            "scheduled": "Decided at a known, scheduled event: election, vote, court date, "
                         "data release, earnings, award ceremony",
            "threshold": "A measurable quantity (price, index, count, poll) crossing a level",
            "unscheduled": "Any time an unscheduled real-world event occurs",
        },
    },
    "single_actor": {
        "type": "noul",
        "instructions": "Could a single person or organization make this resolve YES by "
                        "their own choice (resign, announce, sign, post, launch)?",
    },
    "loose_rules": {
        "type": "noul",
        "instructions": "Do the resolution rules allow YES through paths broader than the "
                        "title suggests (e.g. an announcement, report, or partial event counts)?",
    },
    "ambiguous_source": {
        "type": "noul",
        "instructions": "Is the resolution source or criterion vague, subjective, or open to "
                        "dispute?",
    },
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this market about?",
        "criteria": {
            "conflict": "War, military action, geopolitics, sanctions, diplomacy",
            "elections": "Elections, polls, party politics, appointments",
            "policy": "Laws, regulation, court rulings, government action",
            "markets": "Crypto, stock, commodity prices, macro data, central banks",
            "corporate": "Companies, tech products, business deals, executives",
            "culture": "Entertainment, celebrities, awards, social media, science, other",
        },
    },
}
NOUL_THRESHOLD = 0.5


# ----------------------------------------------------------------- helpers
def load_env(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - hw, c + hw)


def get_json(url, retries=4, backoff=1.5):
    last_err = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (attempt + 1) * 2)
                continue
            if r.status_code in (400, 404, 422):
                return {"_http_error": r.status_code}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    return {"_http_error": "exhausted", "_body": str(last_err)[:120]}


# ----------------------------------------------------------------- data
def load_samples(path, max_price):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("price") or r.get("outcome_yes") in (None, ""):
                continue
            r["price"] = float(r["price"])
            r["outcome_yes"] = int(r["outcome_yes"])
            rows.append(r)
    markets = {r["market_id"]: r["question"] for r in rows if r["price"] < max_price}
    return rows, markets


def fetch_texts(markets, cache):
    todo = [m for m in markets if m not in cache]
    print(f"rules text: {len(markets) - len(todo)} cached, fetching {len(todo)}")
    for i, mid in enumerate(todo, 1):
        m = get_json(f"{GAMMA}/markets/{mid}")
        if "_http_error" in m:
            print(f"  {mid}: {m['_http_error']} -- tagging on question only")
            cache[mid] = {"question": markets[mid], "description": "", "error": str(m["_http_error"])}
        else:
            cache[mid] = {"question": m.get("question") or markets[mid],
                          "description": m.get("description") or ""}
        if i % 25 == 0:
            save_json(TEXT_CACHE, cache)
            print(f"  {i}/{len(todo)}")
        time.sleep(0.15)
    save_json(TEXT_CACHE, cache)


def build_state(text):
    s = f"Prediction market question: {text['question']}\n\nResolution rules:\n{text['description']}"
    return s[:MAX_STATE_CHARS]


# ----------------------------------------------------------------- jev
def jev_endpoint():
    if os.environ.get("AI_GATEWAY_API_KEY"):
        return GATEWAY_URL, os.environ["AI_GATEWAY_API_KEY"], "typesafe-ai/jev"
    if os.environ.get("TYPESAFE_API_KEY"):
        return TYPESAFE_URL, os.environ["TYPESAFE_API_KEY"], "jev-latest"
    sys.exit("No AI_GATEWAY_API_KEY or TYPESAFE_API_KEY in environment or --env-file.")


def jev_call(url, key, model, state, retries=5):
    body = {"model": model, "state": state, "questions": QUESTIONS}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            r = SESSION.post(url, headers=headers, json=body, timeout=60)
        except Exception as e:
            err = str(e)[:120]
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 529) or r.status_code >= 500:
            err = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            sys.exit(f"Jev HTTP {r.status_code}: {r.text[:300]}")  # 401/422: fix, don't retry
        return r.json(), None
    return None, err


def flatten(answers):
    """Jev answers -> {tag: value}. noul -> (bool, p); choice -> (label, confidence)."""
    out = {}
    for q, a in answers.items():
        if a.get("type") == "noul":
            out[q] = a["noul"] >= NOUL_THRESHOLD
            out[q + "_p"] = round(a["noul"], 3)
        elif a.get("type") == "choice":
            out[q] = a["choice"]
            out[q + "_conf"] = round(a.get("confidence", float("nan")), 3)
    return out


def tag_markets(markets, texts, tags, limit):
    url, key, model = jev_endpoint()
    todo = [m for m in markets if m not in tags][:limit]
    print(f"jev: {len(markets) - len(todo)} cached, tagging {len(todo)} via {url}")
    cost, tokens = 0.0, 0
    for i, mid in enumerate(todo, 1):
        res, err = jev_call(url, key, model, build_state(texts[mid]))
        if err:
            print(f"  {mid}: {err} -- skipped, rerun to retry")
            continue
        tags[mid] = flatten(res["answers"])
        tokens += res.get("usage", {}).get("input_tokens", 0)
        cost += float(res.get("provider_metadata", {}).get("gateway", {}).get("cost", 0) or 0)
        if i % 25 == 0:
            save_json(TAG_CACHE, tags)
            print(f"  {i}/{len(todo)}  tokens={tokens}  cost=${cost:.4f}")
    save_json(TAG_CACHE, tags)
    print(f"jev done: tokens={tokens} cost=${cost:.4f}")


def write_tag_csv(markets, texts, tags):
    cols = sorted({k for t in tags.values() for k in t})
    with open(TAG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["market_id", "question"] + cols)
        for mid in markets:
            if mid in tags:
                w.writerow([mid, texts.get(mid, {}).get("question", markets[mid])]
                           + [tags[mid].get(c, "") for c in cols])


# ----------------------------------------------------------------- analysis
def ev_row(label, s):
    n = len(s)
    if n == 0:
        return None
    k = sum(r["outcome_yes"] for r in s)
    price = sum(r["price"] for r in s) / n
    lo, hi = wilson(k, n)
    verdict = ""
    if k / n > price and lo > price and n >= 150:
        verdict = "EDGE?"
    elif k / n > price:
        verdict = "above price, CI too wide" if n >= 30 else "above price, n tiny"
    return f"  {label:<28}{n:>5}{len({r['market_id'] for r in s}):>6}{price:>8.3f}{k / n:>8.3f}" \
           f"   [{lo:.3f},{hi:.3f}]{(k / n / price - 1) * 100:>+8.0f}%  {verdict}"


def analyze(rows, tags, max_price, min_price):
    s_all = [r for r in rows if min_price <= r["price"] < max_price and r["market_id"] in tags]
    tag_names = [q for q in QUESTIONS] + ["neg_risk"]
    header = f"  {'tag = value':<28}{'n':>5}{'mkts':>6}{'price':>8}{'real.':>8}   {'95% CI':<15}{'EV/$':>7}"
    for h in sorted({r["horizon_days"] for r in s_all}, key=int):
        s_h = [r for r in s_all if r["horizon_days"] == h]
        print(f"\n=== horizon {h}d, price [{min_price:.2f},{max_price:.2f}), tagged samples {len(s_h)} ===")
        print(header)
        print(ev_row("ALL (baseline)", s_h))
        for t in tag_names:
            val = (lambda r: r["neg_risk"]) if t == "neg_risk" else (lambda r, t=t: tags[r["market_id"]].get(t))
            for v in sorted({str(val(r)) for r in s_h} - {"None"}):
                line = ev_row(f"{t} = {v}", [r for r in s_h if str(val(r)) == v])
                if line:
                    print(line)
    untagged = len({r["market_id"] for r in rows if min_price <= r["price"] < max_price} - set(tags))
    if untagged:
        print(f"\n({untagged} markets in range not yet tagged -- rerun without --analyze-only)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="longshot_hv_samples.csv")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--max-price", type=float, default=0.20, help="tag/analyze samples priced below this")
    ap.add_argument("--min-price", type=float, default=0.02, help="analyze samples priced at/above this")
    ap.add_argument("--limit", type=int, default=10 ** 9, help="max new Jev calls this run")
    ap.add_argument("--dry-run", action="store_true", help="print one Jev payload and exit")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    load_env(args.env_file)
    rows, markets = load_samples(args.samples, args.max_price)
    print(f"{len(rows)} priced samples, {len(markets)} markets priced < {args.max_price}")
    texts, tags = load_json(TEXT_CACHE), load_json(TAG_CACHE)

    if not args.analyze_only:
        fetch_texts(markets, texts)
        if args.dry_run:
            mid = next(iter(markets))
            print(json.dumps({"model": "typesafe-ai/jev", "state": build_state(texts[mid]),
                              "questions": QUESTIONS}, indent=1))
            return
        tag_markets(markets, texts, tags, args.limit)
        write_tag_csv(markets, texts, tags)
    analyze(rows, tags, args.max_price, args.min_price)


if __name__ == "__main__":
    main()
