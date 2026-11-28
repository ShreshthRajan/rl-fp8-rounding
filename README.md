# rl-fp8-rounding

When RL fine-tuning keeps a model's weights in FP8, the cast from the FP32 master back to FP8 after each optimizer
step decides what the model learns. An RL step moves a weight by about 1e-6, and the E4M3 cell around a typical weight
is about 2e-3 wide, so with round to nearest most updates never reach the FP8 copy. This repository shows what that
does to GRPO, and tests a fix in which each weight keeps one random rounding threshold for the whole run. It is the
code and data behind
[NVIDIA/Megatron-LM#9845](https://github.com/NVIDIA/Megatron-LM/pull/9845), which adds the fix to Megatron as
`--fp8-param-rounding fixed_threshold`.

Every experiment runs on one GPU. The model is Qwen2.5-0.5B-Instruct, trained with GRPO on GSM8K for 250 rollout
batches of 32 completions at learning rate 1e-6. FP8 primary weights are simulated in PyTorch the way Megatron's
blockwise recipe applies them, with the FP32 master kept outside the model and every weight matrix reset to the E4M3
cast of its master after each step.

## Quick start

```
pip install -r requirements.txt
python train.py --tiny --fp8 fixed --out runs/smoke                 # CPU smoke test, about a minute
python train.py --fp8 rne   --int4-mlp --bf16 --out runs/rne        # one A100, about 35 minutes
python train.py --fp8 fixed --int4-mlp --bf16 --out runs/fixed
python analysis/tables.py                                           # every number below, from results/runs
```

`train.py` is the only entry point, and `python train.py --help` lists its flags.

## The stall

Megatron fine-tunes from an FP8 checkpoint by building the FP32 master from the dequantized FP8 parameter, so every
weight starts exactly on the E4M3 grid. Round to nearest then throws away each update until a weight has moved half a
cell, which at this learning rate takes about a thousand steps of consistent movement. We ran this from a checkpoint whose MLP weights are
quantized to INT4 with group size 32, the format Kimi K2.6 ships for its experts. Over three seeds the mean reward was
0.194, against 0.342 for the same model trained in bf16.

## The jump

Megatron can instead build the master from the bf16 checkpoint values (`--load-main-params-from-ckpt`). Under the
power-of-two block scales that TE uses by default, one bf16 value in sixteen sits exactly halfway between two E4M3
values. The first optimizer step breaks every one of these ties toward the sign of its update, so the FP8 model takes a
full-cell step on about 3% of its weights at once. The policy moved by about 1 nat of KL per token in the first step,
and one of three runs never recovered. In a one-step test the jump was the same at every learning rate from 1e-8 to
1e-5, so warmup does not prevent it.

## The fix

With a fixed threshold, each weight rounds up from its lower grid neighbour once its position in the cell passes
1/2 + u, where u is drawn once from [-1/2, 1/2) and kept for the run
([fp8rl/casts.py](fp8rl/casts.py)). Averaged over u the cast is unbiased, weights on the grid stay where they are, and
no weight sits on a tie. A weight that moves a tenth of a cell changes its FP8 value with probability about one in ten,
where round to nearest would leave it unchanged.

| FP32 master | rounding | mean reward, all steps | held-out GSM8K | first-step KL jump |
|---|---|---|---|---|
| bf16 reference | none | 0.342 | 37.4% | 0.016 |
| FP8 grid | round to nearest | 0.194 | 31.4% | 0.004 |
| FP8 grid | fixed threshold | 0.341 | 38.8% | 0.160 |
| checkpoint values | round to nearest | 0.130 | 22.6% | 1.049 |
| checkpoint values | fixed threshold | 0.277 | 38.6% | 0.045 |

Each row is the mean of three seeds. Held-out accuracy is greedy on 300 test questions with a 512-token limit, and one
standard error is about 2.8 points. From the grid, the fixed threshold moves the policy more over the first steps than
bf16 does, because every FP8 value that changes moves by a whole cell. On the checkpoint path, FP32 amax block scales (`--scales amax`) did slightly
better than the fixed threshold, at 0.289 over two seeds. They fix only the jump, though, because a master on the grid stays on the
grid under any scale. Stochastic rounding recovers both cases too, but it draws fresh noise at every cast and changed
a third of all FP8 values per step from the checkpoint start, against 3% with the fixed threshold.

[te_patch/fp8_param_rounding.py](te_patch/fp8_param_rounding.py) applies the fix inside Transformer Engine. Before
TE's blockwise cast kernel runs, it replaces each master shard with a copy moved to the grid value the fixed threshold
picks, so the kernel's round to nearest returns that value unchanged. On an H100 with TE 2.7 and 2.8 the kernel
produced exactly the fixed-threshold bytes on every shape and shard offset we tried, and the extra pass took 0.22 ms
per 7.3M-element shard ([te_patch/check_te.py](te_patch/check_te.py)). The thresholds come from a hash of each
weight's index, so nothing extra is stored and every data-parallel rank agrees. The runs above stored their
thresholds instead, so the patch matches them in distribution rather than weight by weight. The same file has
`fp8_master_census`, which tells you after loading whether a checkpoint is in the stall case or the jump case.

## Weight gradients for a fifth of the columns

RL updates concentrate in a small set of input columns of each weight matrix. In the run we measured, 5% of the
weights, taken as whole columns, carried two thirds of the update, and the same columns carried it in the next 25
steps. A linear layer's weight gradient splits by input column, so the GEMM for the other
columns can be skipped. With `--columns top`, one full backward every 25 steps picks the top 20% of columns by
gradient energy, the steps in between compute weight gradients only for those columns, and the rest move on Adam's
momentum. On plain Qwen2.5-0.5B this kept most of the reward (0.521 over the last 50 batches, against 0.548 dense)
and cost about 4 points of held-out accuracy. At Qwen2.5-7B layer shapes the weight-gradient GEMMs ran 2.6 times
faster on an A100 ([benchmarks/dw_columns.py](benchmarks/dw_columns.py)).

We tried two ways of spending the freed compute. A second-order direction on the active columns did not help, since
SOAP-style Adam in their input eigenbasis matched plain Adam and a K-FAC preconditioner was worse. A second optimizer
pass over each rollout batch did help. Dense training with two passes reached 90% of the one-pass gain in 70 rollout
batches instead of 108, and with 20% of the columns it took 78. Both two-pass versions lost about 4 points of held-out
accuracy, which fits the baseline repulsion that [Sheng et al.](https://arxiv.org/abs/2611.33102) describe for RL updates
on reused rollouts.

The two results interact in FP8. The skipped columns move only by small momentum steps, and round to nearest from the
grid discards them, so reward fell to 0.166. With the fixed threshold it was 0.344, level with bf16 at full gradients.
We proposed column-subset weight gradients to Megatron in
[NVIDIA/Megatron-LM#9846](https://github.com/NVIDIA/Megatron-LM/issues/9846).

## Upstream

The fix is proposed in [NVIDIA/Megatron-LM#9844](https://github.com/NVIDIA/Megatron-LM/issues/9844) and
[#9845](https://github.com/NVIDIA/Megatron-LM/pull/9845), backported to Miles' Megatron fork in
[radixark/Megatron-LM#143](https://github.com/radixark/Megatron-LM/pull/143), and documented for Miles in
[radixark/miles#5081](https://github.com/radixark/miles/pull/5081).

## Layout

```
train.py              every run, selected by flags
fp8rl/                casts, FP8 primary weights, column-subset gradients, second-order directions
te_patch/             the TE patch, its tests and the H100 check
megatron/             the Megatron-LM patch
benchmarks/           weight-gradient GEMM timing
analysis/tables.py    recomputes every number in this README
results/              logs of all 46 runs, GEMM timings and the H100 check
```

## Limits

All training runs use one 0.5B dense model with two or three seeds, and FP8 is simulated rather than run through
Megatron. The 7B numbers are kernel timings, not training runs.
