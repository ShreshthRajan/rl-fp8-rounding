"""Weight gradients for a subset of input columns.

Every `refresh` steps the step is a full probe: every weight gradient is kept, and each input column of each weight
matrix is scored by its gradient energy (sum over rows of g^2). For the next refresh - 1 steps only the top `frac` of
elements, taken as whole columns, receive weight gradients. The other columns get none, as if their part of the
dW = dY^T X GEMM were skipped, and move only on Adam's momentum. The input gradient dX is unchanged.
"""

import torch
from transformers import TrainerCallback

from .fp8_primary import weight_matrices


class ColumnSubsetGradients(TrainerCallback):
    def __init__(self, log, select, frac, refresh=25, seed=0):
        assert select in ("all", "top", "random")
        self.log, self.select, self.frac, self.refresh = log, select, frac, refresh
        self.gen = torch.Generator().manual_seed(seed)
        self.masks, self.energy, self.probe, self.hooks = {}, {}, True, []

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
        self.model = model
        self.named = weight_matrices(model)
        self.N = sum(p.numel() for _, p in self.named)
        for n, p in self.named:
            self.masks[n] = torch.ones(1, p.shape[1], device=p.device, dtype=p.dtype)
            self.hooks.append(p.register_hook(self._hook(n)))
        self.log({"kind": "meta_columns", "select": self.select, "frac": self.frac, "refresh": self.refresh})

    def _hook(self, n):
        def h(g):
            if self.probe:
                self.energy[n] = self.energy.get(n, 0) + (g.float() ** 2).sum(0)
                return g
            return g * self.masks[n]
        return h

    def on_step_begin(self, args, state, control, **kw):
        self.probe = self.select == "all" or state.global_step % self.refresh == 0

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kw):
        t = state.global_step
        if self.select == "all" or (t - 1) % self.refresh:
            self.energy = {}
            return
        if self.select == "top" and sum(float(e.sum()) for e in self.energy.values()) == 0:
            self.log({"kind": "select", "t": t, "skipped": "zero gradient on the probe step; masks kept"})
            return
        mass, size, keys = [], [], []
        for n, p in self.named:
            e = self.energy.get(n, torch.zeros(p.shape[1], device=p.device))
            if self.select == "random":
                e = torch.rand(p.shape[1], generator=self.gen).to(p.device)
            mass.append(e.double()); size.append(torch.full_like(e, p.shape[0]).double()); keys.append((n, p.shape[1]))
        mass, size = torch.cat(mass), torch.cat(size)
        order = torch.argsort(mass / size, descending=True)
        cs = torch.cumsum(size[order], 0)
        k = int(torch.searchsorted(cs, torch.tensor(self.frac * self.N, dtype=cs.dtype, device=cs.device)))
        chosen = torch.zeros_like(mass, dtype=torch.bool)
        chosen[order[:k + 1]] = True
        off = kept = 0
        for n, cnt in keys:
            m = chosen[off:off + cnt]; off += cnt
            self.masks[n] = m.to(self.masks[n].dtype).view(1, -1)
            kept += int(m.sum()) * dict(self.named)[n].shape[0]
        tot = float(mass.sum())
        cap = float(mass[chosen].sum() / tot) if tot > 0 and self.select != "random" else None
        self.log({"kind": "select", "t": t, "kept_frac": kept / self.N, "energy_captured": cap})
        self.energy = {}
