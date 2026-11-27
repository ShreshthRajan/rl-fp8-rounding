"""One GRPO run on GSM8K. Every result in the README is one call of this script with different flags.

  python train.py --out runs/x                                     dense bf16 (reference)
  python train.py --fp8 rne --start grid --int4-mlp --out runs/x   FP8 primary weights, Megatron's default load
  python train.py --fp8 fixed --start grid --int4-mlp --out runs/x   ... with fixed-threshold rounding
  python train.py --columns top --out runs/x                       weight gradients on the top 20% of input columns
  python train.py --columns top --passes 2 --out runs/x            ... and a second optimizer pass per rollout batch
  python train.py --columns top --direction soap --out runs/x      ... with a second-order direction on those columns

Add --tiny for a CPU smoke test with a tiny model.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer

from fp8rl.casts import set_block_scales
from fp8rl.common import (PROMPT_SUFFIX, TINY_MODEL, RunLog, exact_answer_reward, grpo_config, heldout_accuracy,
                          parity_reward, train_dataset)
from fp8rl.fp8_primary import FP8PrimaryWeights
from fp8rl.lazy_dw import ColumnSubsetGradients
from fp8rl.second_order import SecondOrderOnColumns
from transformers import TrainerCallback


class RewardLog(TrainerCallback):
    def __init__(self, log):
        self.log = log

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "reward" in logs:
            self.log({"kind": "reward", "t": state.global_step, "reward": logs["reward"], "entropy": logs.get("entropy")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=250, help="rollout batches")
    ap.add_argument("--fp8", choices=("off", "rne", "sr", "fixed"), default="off",
                    help="FP8 primary weights and their rounding; off trains in bf16")
    ap.add_argument("--start", choices=("grid", "ckpt"), default="grid",
                    help="FP32 master from the dequantized FP8 param (grid) or from checkpoint values (ckpt)")
    ap.add_argument("--scales", choices=("pow2", "amax"), default="pow2", help="E4M3 block scales")
    ap.add_argument("--int4-mlp", action="store_true", help="quantize MLP weights to INT4 group 32 first (K2.6-like)")
    ap.add_argument("--columns", choices=("all", "top", "random"), default="all",
                    help="input columns that get weight gradients between full probe steps")
    ap.add_argument("--frac", type=float, default=0.2, help="fraction of weights kept by --columns top/random")
    ap.add_argument("--direction", choices=("adam", "kfac", "soap"), default="adam",
                    help="update direction on the active columns")
    ap.add_argument("--gamma", type=float, default=0.1, help="K-FAC damping, relative to mean(diag A)")
    ap.add_argument("--passes", type=int, default=1, help="optimizer passes per rollout batch (GRPO mu)")
    ap.add_argument("--heldout-tokens", type=int, default=512)
    ap.add_argument("--n-heldout", type=int, default=300)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (use on GPU)")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (same math, less memory)")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    set_block_scales(args.scales)
    name = TINY_MODEL if args.tiny else args.model
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full, ds = train_dataset(args.tiny)
    log = RunLog(args.out)
    log({"kind": "args", **vars(args)})

    callbacks = [RewardLog(log)]
    if args.fp8 != "off" or args.int4_mlp:
        prompts = [full[5000 + i]["question"] + PROMPT_SUFFIX for i in range(4 if args.tiny else 24)]
        callbacks.append(FP8PrimaryWeights(log, tok, prompts, args.start, "bf16" if args.fp8 == "off" else args.fp8,
                                           args.int4_mlp, n_new=8 if args.tiny else 48, chunk=2 if args.tiny else 4))
    refresh = 3 if args.tiny else 25
    if args.direction != "adam":
        callbacks.append(SecondOrderOnColumns(log, args.frac, args.direction, gamma=args.gamma, refresh=refresh,
                                              seed=args.seed))
    elif args.columns != "all":
        callbacks.append(ColumnSubsetGradients(log, args.columns, args.frac, refresh=refresh, seed=args.seed))

    trainer = GRPOTrainer(model=model, processing_class=tok, train_dataset=ds, args=grpo_config(args, args.out),
                          reward_funcs=[parity_reward if args.tiny else exact_answer_reward], callbacks=callbacks)
    trainer.train()
    heldout_accuracy(trainer.model, tok, 8 if args.tiny else args.n_heldout, 16 if args.tiny else args.heldout_tokens,
                     log)


if __name__ == "__main__":
    main()
