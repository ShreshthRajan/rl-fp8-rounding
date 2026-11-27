"""FP8 primary weights, simulated in a TRL trainer.

The FP32 master lives outside the model. Before training and after every optimizer step each weight matrix is set to
cast(master), so rollouts, the forward pass and the gradients all use the FP8 copy, and the optimizer updates the
master. Embeddings and the LM head stay in bf16.

start  grid  master = the dequantized FP8 parameter, which is what Megatron's fine-tune load without optimizer state
             builds. Every weight starts on the E4M3 grid.
       ckpt  master = the checkpoint values, as with --load-main-params-from-ckpt.
cast   rne   round to nearest (TE today); sr: stochastic rounding, fresh noise per cast; fixed: fixed-threshold
             rounding with one u per weight; bf16: no FP8 (reference).
"""
import torch
from transformers import TrainerCallback

from .casts import fp8_fixed_threshold, fp8_rne, fp8_sr, int4_g32, tie_fraction

MLP_KEYS = ("gate_proj", "up_proj", "down_proj")
KL_STEPS = (1, 2, 3, 5, 10, 25, 50, 100, 150, 200, 250)


def weight_matrices(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and p.dim() == 2 and "embed" not in n and "lm_head" not in n]


class FP8PrimaryWeights(TrainerCallback):
    def __init__(self, log, tok, eval_prompts, start, cast, int4_mlp, n_new=48, chunk=4):
        self.log, self.tok, self.eval_prompts = log, tok, eval_prompts
        self.start, self.cast, self.int4_mlp = start, cast, int4_mlp
        self.n_new, self.chunk = n_new, chunk
        self.gen = torch.Generator().manual_seed(0)
        self.master, self.u, self.prev, self.zero0 = {}, {}, {}, {}

    def _cast(self, w, n):
        if self.cast == "sr":
            return fp8_sr(w, self.gen)
        if self.cast == "fixed":
            return fp8_fixed_threshold(w, self.u[n].float())
        return fp8_rne(w)

    @torch.no_grad()
    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
        self.model = model
        self.named = weight_matrices(model)
        if self.int4_mlp:  # a Kimi-K2.6-like checkpoint: INT4 group-32 MLP weights, bf16 attention
            for n, p in self.named:
                if any(k in n for k in MLP_KEYS) and p.shape[1] % 32 == 0:
                    p.data.copy_(int4_g32(p.data))
        self._make_evalset()
        self.base0 = self._logprobs()
        if self.cast == "bf16":
            self.log({"kind": "meta", "start": self.start, "cast": self.cast})
            return
        ties = tot = 0
        for n, p in self.named:
            m = p.data.float().clone()
            ties += tie_fraction(m) * m.numel(); tot += m.numel()
            if self.start == "grid":
                m = fp8_rne(m)
            self.master[n] = m
            self.zero0[n] = m == 0
            if self.cast == "fixed":
                self.u[n] = (torch.rand(m.shape, generator=self.gen) - 0.5).to(m.device, torch.float16)
            q = self._cast(m, n)
            p.data.copy_(q)
            self.prev[n] = q.to(torch.bfloat16)
        inner = getattr(optimizer, "optimizer", optimizer)
        inner.register_step_pre_hook(self._pre)
        inner.register_step_post_hook(self._post)
        self.log({"kind": "meta", "start": self.start, "cast": self.cast, "tie_frac_ckpt": ties / tot})
        self.log({"kind": "kl0", "t": 0, "kl": self._kl_now()})

    @torch.no_grad()
    def _pre(self, opt, *a, **k):
        for n, p in self.named:
            p.data.copy_(self.master[n])

    @torch.no_grad()
    def _post(self, opt, *a, **k):
        changed = tot = ch_zero = n_zero = 0
        for n, p in self.named:
            self.master[n].copy_(p.data)
            q = self._cast(self.master[n], n)
            qb = q.to(torch.bfloat16)
            d = qb.view(torch.int16) != self.prev[n].view(torch.int16)
            z = self.zero0[n]
            changed += int(d.sum()); tot += qb.numel()
            ch_zero += int((d & z).sum()); n_zero += int(z.sum())
            self.prev[n] = qb
            p.data.copy_(q)
        self.log({"kind": "codes", "changed_frac": changed / tot,
                  "changed_frac_nonzero": (changed - ch_zero) / max(tot - n_zero, 1),
                  "changed_frac_zero": ch_zero / max(n_zero, 1), "zero_frac": n_zero / tot})

    @torch.no_grad()
    def _make_evalset(self):
        """Fixed completions sampled once from the starting policy, for KL(pi_0 || pi_t)."""
        m, dev = self.model, next(self.model.parameters()).device
        self.tok.padding_side = "left"
        enc = self.tok(self.eval_prompts, return_tensors="pt", padding=True).to(dev)
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(1234)
            out = m.generate(**enc, max_new_tokens=self.n_new, do_sample=True, temperature=1.0,
                             pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        comp = torch.ones_like(gen, dtype=torch.bool)
        for i in range(gen.shape[0]):
            hits = (gen[i] == self.tok.eos_token_id).nonzero()
            if len(hits):
                comp[i, hits[0, 0] + 1:] = False
        am = torch.cat([enc["attention_mask"], comp.long()], 1)
        cm = torch.cat([torch.zeros_like(enc["attention_mask"], dtype=torch.bool), comp], 1)
        self.evalset = (out, am, cm)
        self.tok.padding_side = "right"

    @torch.no_grad()
    def _logprobs(self):
        ids, am, cm = self.evalset
        res = []
        for s in range(0, ids.shape[0], self.chunk):
            sl = slice(s, s + self.chunk)
            with torch.autocast("cuda" if ids.is_cuda else "cpu", dtype=torch.bfloat16):
                logits = self.model(input_ids=ids[sl], attention_mask=am[sl]).logits[:, :-1].float()
            res.append(torch.log_softmax(logits[cm[sl][:, 1:]], -1))
            del logits
        return res

    def _kl_now(self):
        tot = cnt = 0.0
        for b, c in zip(self.base0, self._logprobs()):
            tot += float((b.exp() * (b - c)).sum()); cnt += b.shape[0]
        return tot / max(cnt, 1)

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kw):
        if state.global_step in KL_STEPS:
            self.log({"kind": "kl0", "t": state.global_step, "kl": self._kl_now()})
