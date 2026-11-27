"""GRPO on GSM8K with TRL: data, reward, trainer config, and held-out accuracy. Shared by every run."""
import json
import os
import re

import torch
from datasets import load_dataset
from trl import GRPOConfig

PROMPT_SUFFIX = "\nThink step by step. End with: #### <answer>."
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text):
    nums = _NUM.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def exact_answer_reward(completions, answer, **kwargs):
    """1 if the last number in the completion equals the reference answer, else 0."""
    out = []
    for comp, gold in zip(completions, answer):
        text = comp if isinstance(comp, str) else comp[-1]["content"]
        pred, ref = last_number(text), last_number(gold.split("####")[-1])
        out.append(float(pred is not None and pred == ref))
    return out


def parity_reward(completions, **kwargs):
    """Smoke-test reward that varies within a group, so the optimizer takes real steps."""
    return [float(len(c if isinstance(c, str) else c[-1]["content"]) % 2) for c in completions]


def train_dataset(tiny):
    full = load_dataset("openai/gsm8k", "main", split="train")
    ds = full.select(range(32 if tiny else 2000)).map(lambda ex: {"prompt": ex["question"] + PROMPT_SUFFIX})
    return full, ds


def grpo_config(args, out):
    """The configuration of every run: 32 completions per step (4 prompts x 8), 256 new tokens, constant lr, no KL."""
    return GRPOConfig(
        output_dir=out, seed=args.seed, learning_rate=args.lr, adam_beta1=0.9, adam_beta2=0.999,
        max_steps=8 if args.tiny else args.steps * args.passes, per_device_train_batch_size=4 if args.tiny else 32,
        num_generations=2 if args.tiny else 8, max_completion_length=16 if args.tiny else 256, temperature=1.0,
        bf16=args.bf16, beta=0.0, use_vllm=False, num_iterations=args.passes,
        gradient_checkpointing=args.grad_ckpt, logging_steps=1, save_strategy="no", report_to=[],
        lr_scheduler_type="constant", warmup_steps=0)


class RunLog:
    """Append-only JSON-lines log shared by the callbacks of one run."""

    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.f = open(os.path.join(out, "log.jsonl"), "a")

    def __call__(self, record):
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()


@torch.no_grad()
def heldout_accuracy(model, tok, n, max_new_tokens, log):
    """Greedy accuracy on the first n GSM8K test questions, scored with the training reward's rule."""
    test = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    model.eval()
    tok.padding_side = "left"
    correct = 0
    for i in range(0, n, 16):
        batch = test.select(range(i, min(n, i + 16)))
        enc = tok([q + PROMPT_SUFFIX for q in batch["question"]], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for txt, ans in zip(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True),
                            batch["answer"]):
            pred, ref = last_number(txt), last_number(ans.split("####")[-1])
            correct += int(pred is not None and pred == ref)
    log({"kind": "heldout", "n": n, "max_new_tokens": max_new_tokens, "acc": correct / n})
