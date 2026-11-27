# rl-fp8-rounding

When RL fine-tuning keeps its weights in FP8, the rounding from the FP32 master to FP8 decides what the model learns.
An RL step moves a weight by about 1e-6, while the E4M3 cell around a typical weight is about 2e-3 wide. This repository
measures what that does to GRPO, and tests a fix: a fixed rounding threshold for each weight. It is the code and data
behind a proposed Megatron-LM flag, `--fp8-param-rounding fixed_threshold`, whose patch is in [megatron/](megatron/).

Everything here runs on one GPU. The runs use Qwen2.5-0.5B-Instruct, GRPO on GSM8K (TRL), 250 rollout batches of 32
completions and lr 1e-6. FP8 primary weights are simulated the way Megatron's blockwise recipe applies them. The FP32
master lives outside the model, and after each optimizer step every weight matrix is set to the E4M3 cast of its
master, with 128x128 block scales.

## Quick start

```
pip install -r requirements.txt
python train.py --tiny --fp8 fixed --start grid --out runs/smoke      # CPU smoke test, a minute
python train.py --fp8 rne   --start grid --int4-mlp --bf16 --out runs/grid_rne_s15     # one A100, about 35 min
python train.py --fp8 fixed --start grid --int4-mlp --bf16 --out runs/grid_fixed_s15
python analysis/tables.py                                            # every table below, from results/runs
```

`python train.py --help` lists every flag. Each table row below names the flags that produce it.

## Two failures, one for each way of building the master

Megatron builds the FP32 master in one of two ways. A fine-tune load without optimizer state builds it from the
dequantized FP8 parameter, so every weight starts exactly on the E4M3 grid. Round to nearest then discards each update
until the weight has drifted half a cell, and learning stalls. With `--load-main-params-from-ckpt` it is built from the
bf16 checkpoint values instead. Under power-of-two block scales, TE's default for blockwise FP8, 1/16 of bf16 values
sit exactly halfway between two E4M3 values. The first update breaks every one of those ties toward its own sign, and
the FP8 model jumps.

The runs below start from a checkpoint whose MLP weights are quantized to INT4 with group size 32, the format Kimi K2.6
ships for its experts (`--int4-mlp`). Seeds are 15, 16 and 17. The jump column is the largest KL(π₀‖π_t) over the
first three steps, minus its value before training.

| FP32 master | rounding | flags | seeds | mean reward, all steps | held-out GSM8K | first-step jump |
|---|---|---|---|---|---|---|
| (bf16 reference) | | `--int4-mlp` | 3 | 0.342 | 37.4% | 0.016 |
| FP8 grid | round to nearest | `--fp8 rne --start grid` | 3 | **0.194** | 31.4% | 0.004 |
| FP8 grid | stochastic | `--fp8 sr --start grid` | 2 | 0.334 | 38.5% | 0.213 |
| FP8 grid | fixed threshold | `--fp8 fixed --start grid` | 3 | **0.341** | 38.8% | 0.160 |
| checkpoint values | round to nearest | `--fp8 rne --start ckpt` | 3 | 0.130 | 22.6% | **1.049** |
| checkpoint values | round to nearest, FP32 amax scales | `... --scales amax` | 2 | 0.289 | 38.7% | 0.137 |
| checkpoint values | stochastic | `--fp8 sr --start ckpt` | 2 | 0.302 | 38.0% | 0.093 |
| checkpoint values | fixed threshold | `--fp8 fixed --start ckpt` | 3 | 0.277 | 38.6% | **0.045** |

Held-out accuracy is greedy, on 300 test questions with a 512-token limit; one standard error is about 2.8 points.

From the grid, round to nearest keeps a bit more than half of bf16's reward (0.194 against 0.342), and the fixed
threshold recovers all of it. From checkpoint values, round to nearest jumps by about 1 nat of KL per token at the first
step, and one of the three runs never recovered. With the fixed threshold the jump falls to 0.045. From the grid,
the fixed threshold and stochastic rounding move the policy more over the first steps than bf16 does (0.16 and 0.21
against 0.016). They let updates through, and each FP8 value that changes moves by a whole cell. In a separate
one-step test on plain Qwen2.5-0.5B, the jump under round to nearest was 1.87 nats per token at every learning rate
from 1e-8 to 1e-5, so warmup does not prevent it.

FP32 amax scales, which TE uses with `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`, leave 0.5% of weights on ties instead of
6.25%. They cut the jump about sevenfold, and on the checkpoint path they do as well as the fixed threshold. They
do nothing for the stall, since a master on the grid stays on the grid under any scale. Stochastic rounding fixes both
failures, but it draws fresh noise at every cast. From checkpoint values it changed 33.7% of FP8 values per step,
against 2.7% for the fixed threshold, and it makes the cast nondeterministic.

## The fix

Each weight gets a threshold 1/2 + u, with u drawn once from [-1/2, 1/2) and kept for the run. A weight's E4M3 value
rounds up from its lower grid neighbour once its position in the cell passes that threshold
([fp8rl/casts.py](fp8rl/casts.py)). Averaged over u, the cast is unbiased. Weights on the grid stay put, nothing sits
on a tie, and a weight that drifts by a fraction p of a cell changes its FP8 value with probability p. Round to nearest
from the grid changes nothing until p reaches 1/2.

[te_patch/fp8_param_rounding.py](te_patch/fp8_param_rounding.py) applies this rule inside Transformer Engine's
blockwise cast. It hands TE's `fp8_block_scaling_partial_cast` kernel a temporary copy of each master shard, snapped to
the grid value the fixed threshold picks. The kernel's round to nearest then returns that value unchanged. Thresholds
come from a hash of each weight's index, so nothing is stored and every data-parallel shard agrees. On an H100 with TE
2.8 and 2.7 ([check_te.py](te_patch/check_te.py),
[results/benchmarks](results/benchmarks/)), TE's kernel produced exactly the fixed-threshold bytes on every shape and
shard offset tested, both alone and inside `cast_master_weights_to_fp8`. The snap took 0.22 ms per 7.3M-element shard,
against 0.056 ms for TE's own cast. The runs above drew u with `torch.rand` and stored it. The patch uses the same rule
with hashed u, so its results match the table in distribution, not bit for bit.

`fp8_master_census` in the same file reports how many master weights sit on the grid and how many on ties. Run once
after load, it tells you which of the two failures a checkpoint is in.

## Weight gradients for a fifth of the input columns

RL updates concentrate in a few input columns of each weight matrix. In one seed, 5% of weights, taken as whole columns,
carried two thirds of the update's importance (Adam's second moment times the squared change), and the same columns
carried it over the next 25 steps. A linear layer's weight gradient splits by input column, dW[:, J] = dYᵀ X[:, J], so the rest can be
skipped. `--columns top` runs one full backward every 25 steps, keeps weight gradients only for the top 20% of columns
by gradient energy until the next one, and lets the other columns move on Adam's momentum. The input gradient stays
dense. These runs use plain Qwen2.5-0.5B, seeds 13 to 15, with held-out accuracy at a 256-token limit
(`--heldout-tokens 256`).

| update | flags | seeds | rollout batches to 90% of dense gain | reward, last 50 | held-out |
|---|---|---|---|---|---|
| dense AdamW | | 3 | 108 | 0.548 | 48.3% |
| top 20% of columns | `--columns top` | 3 | 123 | 0.521 | 44.1% |
| random 20% of columns | `--columns random` | 2 | 230 | 0.519 | 42.3% |
| probe steps only | `--columns top --frac 0` | 2 | 250 | 0.423 | 37.7% |
| top 20%, SOAP direction | `--columns top --direction soap` | 3 | 122 | 0.545 | 46.7% |
| top 20%, K-FAC direction | `--columns top --direction kfac` | 1 | 222 | 0.499 | 42.7% |
| dense AdamW, two passes | `--passes 2` | 3 | 70 | 0.572 | 44.2% |
| top 20% of columns, two passes | `--columns top --passes 2` | 3 | 78 | 0.538 | 44.3% |

A rollout batch reaches 90% of dense gain when the trailing 25-batch mean reward reaches 0.509. A run that never gets
there scores 250.

Computing 20% of the weight gradients costs some reward (0.521 against 0.548) and about 4 points of held-out accuracy.
On Qwen2.5-7B layer shapes with 16k tokens, those gradients took 25.6 ms against 66.7 ms dense on an A100
([benchmarks/dw_columns.py](benchmarks/dw_columns.py)). That is 2.6 times faster, and it saves 31% of the
weight-plus-input-gradient matmul time. At 0.5B shapes the saving is within noise.

The freed compute could go to a better update on the columns that do move. A second-order direction does not help.
SOAP-style Adam in the columns' input eigenbasis matches plain Adam, and a K-FAC input preconditioner is much worse.
The input statistics both need cost 10 to 12% of the freed FLOPs. A second optimizer pass over each rollout batch does help.
With 20% of columns it reaches 90% of the dense gain in 78 rollout batches instead of 108. Dense two-pass training
takes 70, and the 20% version uses 78% of its trainer FLOPs. Both two-pass variants lose about 4 points of held-out accuracy.

With FP8 primary weights the two results interact. Skipped columns move only by small momentum steps, which round to
nearest discards from a grid start:

| FP8 grid start, 20% of columns | flags | seeds | mean reward, all steps | held-out GSM8K |
|---|---|---|---|---|
| round to nearest | `--fp8 rne --start grid --int4-mlp --columns top` | 2 | 0.166 | 31.2% |
| fixed threshold | `--fp8 fixed --start grid --int4-mlp --columns top` | 3 | **0.344** | 39.1% |

FP8 weights with the fixed threshold and 20% of the weight gradients match dense bf16 (0.342).

## What is here

```
train.py                       every run: GRPO with flags for FP8 rounding, column subsets, passes, direction
fp8rl/casts.py                 E4M3 block scales; round to nearest, stochastic and fixed-threshold casts; INT4
fp8rl/fp8_primary.py           FP8 primary weights: FP32 master outside the model, cast after every step
fp8rl/lazy_dw.py               weight gradients for a subset of input columns
fp8rl/second_order.py          K-FAC and SOAP directions on those columns
fp8rl/common.py                data, reward, trainer config, held-out accuracy
te_patch/                      the fixed-threshold cast for TE's blockwise path, its tests, the H100 check
megatron/                      the Megatron-LM patch adding --fp8-param-rounding
benchmarks/dw_columns.py       timing of column-subset weight-gradient GEMMs
analysis/tables.py             recomputes every table in this README from results/
results/runs/                  logs of every run above (JSON lines: reward per step, KL, FP8 code changes, held-out)
results/benchmarks/            GEMM timings (A100) and the TE check (H100)
```

## Limits

Every training result is from one 0.5B dense model with 2 or 3 seeds. The 7B numbers are kernel timings, not
training runs. FP8 is simulated in PyTorch with TE's arithmetic, not run through Megatron. The TE patch is checked
against TE's kernels, but it has not yet trained a model inside Megatron.
