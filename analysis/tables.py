"""Recompute every table in the README from the run logs in results/runs.

    python analysis/tables.py
"""
import glob
import json
import os
import statistics as st

RUNS = os.path.join(os.path.dirname(__file__), "..", "results", "runs")


def load(path):
    rows = [json.loads(line) for line in open(path)]
    rewards = {r["t"]: r["reward"] for r in rows if r.get("kind") == "reward"}
    kl = {r["t"]: r["kl"] for r in rows if r.get("kind") == "kl0"}
    heldout = [r["acc"] for r in rows if r.get("kind") == "heldout"]
    return {"reward": [rewards[t] for t in sorted(rewards)], "kl": kl, "heldout": heldout[-1] if heldout else None}


def runs(group, arm):
    paths = sorted(glob.glob(os.path.join(RUNS, group, f"{arm}_s*.jsonl")))
    return {int(p.rsplit("_s", 1)[1][:-6]): load(p) for p in paths}


def jump(run):
    """Largest KL(pi_0 || pi_t) over steps 1-3, minus its value at step 0."""
    return max(run["kl"].get(t, 0.0) for t in (1, 2, 3)) - run["kl"].get(0, 0.0)


def row(name, rs, cols):
    seeds = sorted(rs)
    cells = []
    for c in cols:
        vals = [c[1](rs[s]) for s in seeds]
        cells.append(f"{100 * st.mean(vals):.1f}%" if c[0] == "held-out" else c[2].format(st.mean(vals)))
    print(f"| {name} | {len(seeds)} | " + " | ".join(cells) + " |")


ALL = ("mean reward, all steps", lambda r: st.mean(r["reward"]), "{:.3f}")
HELD = ("held-out", lambda r: r["heldout"], "")
JUMP = ("first-step jump (KL)", jump, "{:.3f}")

print("\n## FP8 primary weights, K2.6-like checkpoint (held-out at 512 tokens)\n")
print("| FP32 master, rounding | seeds | " + " | ".join(c[0] for c in (ALL, HELD, JUMP)) + " |")
print("|---|---|---|---|---|")
for name, arm in [("bf16 reference", "bf16"), ("grid, round to nearest", "grid_rne"), ("grid, stochastic", "grid_sr"),
                  ("grid, fixed threshold", "grid_fixed"), ("checkpoint values, round to nearest", "ckpt_rne"),
                  ("checkpoint values, round to nearest, FP32 amax scales", "ckpt_rne_amax"),
                  ("checkpoint values, stochastic", "ckpt_sr"), ("checkpoint values, fixed threshold", "ckpt_fixed")]:
    row(name, runs("fp8", arm), (ALL, HELD, JUMP))

print("\n## FP8 primary weights with weight gradients on 20% of input columns (grid start)\n")
print("| rounding | seeds | mean reward, all steps | held-out |\n|---|---|---|---|")
for name, arm in [("round to nearest", "grid_rne_top20"), ("fixed threshold", "grid_fixed_top20")]:
    row(name, runs("fp8_columns", arm), (ALL, HELD))

# Column subsets, second passes and second-order directions (plain Qwen2.5-0.5B, held-out at 256 tokens).
groups = {"dense": runs("columns", "dense"), "top20": runs("columns", "top20"),
          "random20": runs("columns", "random20"), "probe_only": runs("columns", "probe_only"),
          "dense_2pass": runs("passes", "dense_2pass"), "top20_2pass": runs("passes", "top20_2pass"),
          "soap": runs("second_order", "soap"), "kfac": runs("second_order", "kfac")}
pool = [r for k in ("dense", "top20", "kfac", "soap", "top20_2pass", "dense_2pass") for r in groups[k].values()]
r0 = st.mean(st.mean(r["reward"][:10]) for r in pool)
threshold = r0 + 0.9 * (st.mean(st.mean(r["reward"][-50:]) for r in groups["dense"].values()) - r0)


def s90(run):
    """First rollout batch at which the trailing 25-batch mean reward reaches 90% of dense AdamW's gain."""
    r = run["reward"]
    for t in range(25, len(r) + 1):
        if st.mean(r[t - 25:t]) >= threshold:
            return t
    return len(r)


S90 = ("rollout batches to 90% of dense gain", s90, "{:.0f}")
LAST = ("reward, last 50", lambda r: st.mean(r["reward"][-50:]), "{:.3f}")
print("\n## Weight gradients on a subset of input columns (held-out at 256 tokens)\n")
print(f"Threshold: reward {threshold:.3f}.\n")
print("| update | seeds | " + " | ".join(c[0] for c in (S90, LAST, HELD)) + " |\n|---|---|---|---|---|")
for name, k in [("dense AdamW", "dense"), ("top 20% of columns", "top20"), ("random 20% of columns", "random20"),
                ("probe steps only", "probe_only"), ("dense AdamW, two passes", "dense_2pass"),
                ("top 20% of columns, two passes", "top20_2pass"), ("top 20%, SOAP direction", "soap"),
                ("top 20%, K-FAC direction", "kfac")]:
    row(name, groups[k], (S90, LAST, HELD))

bench = json.load(open(os.path.join(RUNS, "..", "benchmarks", "dw_columns_a100.json")))
for r in bench["rows"]:
    if r.get("summary") and r["model"] == "7B" and r["frac"] == 0.2:
        print(f"\nWeight-gradient GEMMs at Qwen2.5-7B shapes, 20% of columns, {bench['tokens']} tokens, {bench['gpu']}: "
              f"{r['ms_dw_total']:.1f} ms vs {r['ms_dw_dense']:.1f} ms dense ({r['dw_speedup']:.1f}x); "
              f"{100 * r['backward_matmul_time_saved']:.0f}% of weight-plus-input-gradient matmul time saved.")
