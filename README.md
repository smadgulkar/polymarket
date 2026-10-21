# Polymarket microstructure research

A set of empirical studies on [Polymarket](https://polymarket.com) prediction markets, built to answer one question honestly: **are there pricing inefficiencies that survive real-world frictions, or does the edge disappear once you account for stale prices, spreads and depth?**

Each script is a self-contained study with a stated hypothesis, a measurement method, and a decision rule fixed *before* looking at the results. Nothing here places orders. Everything works from Polymarket's public APIs (Gamma for market metadata, CLOB for price history and order books, the data API for wallet activity).

## Studies

| Script | Question | Method |
|---|---|---|
| `polymarket_longshot_hv.py` | Do contracts priced 2–10¢ resolve YES less often than their price implies (favourite–longshot bias)? | Samples last-changed prices at 7/14/21/30 days before scheduled end across closed, high-volume markets; reports calibration by price bucket with 95% CIs. Pre-registered rule: pooled 2–10¢ at 14d and 21d with n ≥ 150, realized ≤ 1.5% and CI upper bound below mean price ⇒ edge is real. |
| `polymarket_residual.py` | If named longshots are overpriced, is the "Other / none of the above" contract underpriced by the same amount? | Uses the identity P(Other) = 1 − Σ P(named) on resolved multi-outcome events; calibration and EV per $1 on Other, with candidates added after the sample time detected and dropped. |
| `polymarket_term_structure.py` | Does lottery demand overprice the *nearest* deadline when the same event is listed at several dates? | Treats nested deadlines as a survival curve, derives implied "between" probabilities and hazards, and tests calendar spreads (sell near / buy far) against realized outcomes. Drops families whose outcomes are non-monotone. |
| `polymarket_term_live.py` | Do those term-structure mispricings exist on live order books, at a size you can actually trade? | Scans open nested-deadline pairs, reads real bids/asks and depth, flags riskless violations (near bid > far ask) and hazard-shape inversions; appends to a timestamped log so repeated runs form a paper track record. |
| `polymarket_propogation.py` | When one leg of a linked family jumps, how fast do the other legs reprice? | Finds large one-hour price shocks (threshold configurable) in negRisk events and cumulative-deadline curves, pulls minute-level history around each, and measures the family's mispricing at +1, +5, +15, +60 and +180 minutes. |
| `polymarket_shock_monitor.py` | Is that post-shock residual *sellable* at the bid, or only visible in last-trade prices? | Live monitor: detects shocks on mid prices and snapshots full order books of every leg at +0/1/5/15/60 min. Decision rule: sibling lag at the bid near zero at minute 1 ⇒ dead; ≥ ~5¢ ⇒ worth building execution. |
| `polymarket_tail_cluster.py` | Are cheap geopolitical contracts priced off a calm base rate that lags real escalation? | Groups cheap, near-dated contracts into clusters by shared entities and alerts when a cluster reprices together. The code documents that measured EV on this bucket was roughly zero, so it is treated as a fair lottery, not an edge. |
| `polymarket_core_pipeline.py` | Which profitable wallets can actually be copied, and how much of their edge survives entering late? | Harvests traders from high-volume markets, filters out market makers, ultra-short holds and one-lucky-bet profiles, then replays each wallet's buys with 1/5/15-minute fill delays and ranks by copier ROI. |
| `polymarket_wallet.py` | What does a single account's trading actually look like? | Rebuilds every position from public trade history: entry-price distribution, win rate and ROI by price bucket, hold times, PnL concentration. |

## Design principles

- **Pre-registered decision rules.** Each study states what result would confirm or kill the idea before it runs.
- **Frictions first.** Last-trade prices, bid/ask spreads, depth and timing are modelled explicitly; where a backtest uses last trades, a live script checks the same effect on real books.
- **Confidence intervals, not point estimates.** Calibration tables report sample sizes and 95% intervals.
- **Caching and resumability.** Long API walks cache to disk so a crash or re-run costs minutes, not hours.

## Data files

The CSV/JSON files in the repository are outputs from runs of these scripts (samples, shock logs, live-scan logs, a public wallet's activity). They contain only public market data.

## Running

```bash
pip install curl_cffi
python polymarket_longshot_hv.py            # any script; each has --help
```

Python 3.10+. Each script's docstring explains its options and how to read the output.

## Author

Shubhankar Madgulkar — [GitHub](https://github.com/smadgulkar) · [LinkedIn](https://www.linkedin.com/in/shubhankar-madgulkar-7a300814/)
