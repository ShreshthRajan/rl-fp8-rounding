"""A second-order direction on the active input columns, spending part of the compute that skipped columns free.

On the columns J that receive weight gradients, Adam's update direction is replaced and its per-matrix step size kept:

  kfac  D_J = m_J (A_J + lam I)^-1, with A_J an EMA of X_J^T X_J / T (the K-FAC input factor) and
        lam = gamma * mean(diag A_J)
  soap  Adam run in the eigenbasis of A_J (one-sided SOAP)

Each matrix's update on J is rescaled to the Frobenius norm of the AdamW update it replaces, so only the direction
differs. Probe steps are plain AdamW.
"""
import time

import torch

from .lazy_dw import ColumnSubsetGradients


class SecondOrderOnColumns(ColumnSubsetGradients):
    def __init__(self, log, frac, mode, gamma=0.1, beta=0.95, refresh=25, seed=0, basis_every=5):
        super().__init__(log, "top", frac, refresh=refresh, seed=seed)
        assert mode in ("kfac", "soap")
        self.mode, self.basis_every = mode, basis_every
        self.Q, self.v2, self.nstep = {}, {}, {}
        self.gamma, self.beta = gamma, beta
        self.A, self.acc, self.cnt, self.cols, self.old, self.mask_id = {}, {}, {}, {}, {}, {}

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
        super().on_train_begin(args, state, control, model=model, optimizer=optimizer, **kw)
        mods = dict(model.named_modules())
        self.rows = None
        model.register_forward_pre_hook(self._mask_hook, with_kwargs=True)
        for n, _ in self.named:
            mods[n.rsplit(".", 1)[0]].register_forward_hook(self._fwd(n))
        self.opt = getattr(optimizer, "optimizer", optimizer)
        self.opt.register_step_pre_hook(self._pre)
        self.opt.register_step_post_hook(self._post)
        self._refresh_cols()
        self.log({"kind": "meta_second_order", "gamma": self.gamma, "beta": self.beta, "mode": self.mode,
                   "basis_every": self.basis_every})

    def _refresh_cols(self):
        for n, _ in self.named:
            m = self.masks[n]
            if self.mask_id.get(n) is not m:
                self.mask_id[n] = m
                self.cols[n] = torch.nonzero(m.view(-1) > 0).view(-1)
                self.A.pop(n, None)  # new columns: restart the factor
                self.Q.pop(n, None); self.v2.pop(n, None); self.nstep.pop(n, None)

    def _mask_hook(self, mod, args, kwargs):
        am = kwargs.get("attention_mask")
        self.rows = None if am is None or am.dim() != 2 else am.reshape(-1) > 0

    def _fwd(self, n):
        # A_J accumulates in the forward pass over non-padding tokens, so no activation is held until backward.
        def h(mod, inp, out):
            if self.probe or not torch.is_grad_enabled() or not out.requires_grad:
                return
            with torch.no_grad():
                x = inp[0].reshape(-1, inp[0].shape[-1])
                xs = x.index_select(1, self.cols[n])
                rows = self.rows if self.rows is not None and self.rows.numel() == xs.shape[0] else None
                if rows is not None:
                    xs = xs[rows]
                self.acc[n] = self.acc.get(n, 0) + (xs.T @ xs).float()
                self.cnt[n] = self.cnt.get(n, 0) + xs.shape[0]
                del xs
        return h

    @torch.no_grad()
    def _pre(self, opt, *a, **k):
        self.old = {}
        if self.probe:
            self.acc, self.cnt = {}, {}
            return
        for n, p in self.named:
            self.old[n] = p.data[:, self.cols[n]].clone()

    @torch.no_grad()
    def _post(self, opt, *a, **k):
        if self.probe or not self.old:
            return
        t0 = time.time(); coss = []
        for n, p in self.named:
            if n not in self.acc or self.cnt.get(n, 0) == 0:
                continue
            J = self.cols[n]
            a = self.acc.pop(n) / self.cnt.pop(n)
            A = a if n not in self.A else self.beta * self.A[n] + (1 - self.beta) * a
            self.A[n] = A
            m = self.opt.state[p]["exp_avg"][:, J].float()
            if self.mode == "soap":
                D = self._soap(n, p, J, A, m)
                if D is None:
                    continue
            else:
                lam = self.gamma * A.diagonal().mean().clamp_min(1e-12)
                eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                L, info = torch.linalg.cholesky_ex(A + lam * eye)
                if int(info) != 0:
                    continue
                D = torch.cholesky_solve(m.T.contiguous(), L).T
            old = self.old[n]
            dadam = p.data[:, J].float() - old.float()
            na, nd = dadam.norm(), D.norm()
            if nd == 0 or na == 0:
                continue
            upd = -D * (na / nd)
            coss.append(float((upd * dadam).sum() / (na * na)))
            p.data[:, J] = (old.float() + upd).to(p.dtype)
        self.acc, self.cnt, self.old = {}, {}, {}
        if coss:
            self.log({"kind": "second_order", "cos_vs_adam_mean": sum(coss) / len(coss), "cos_vs_adam_min": min(coss),
                       "n_mats": len(coss), "post_ms": 1000 * (time.time() - t0)})

    def _soap(self, n, p, J, A, m):
        """One-sided SOAP: Adam run in the eigenbasis Q of A_J. Adam's own first moment is rotated; the second moment
        is an EMA of the rotated gradient squared. With Q = I this is Adam on J, so the only change from plain
        column-subset Adam is the rotation."""
        k = self.nstep.get(n, 0); self.nstep[n] = k + 1
        if n not in self.Q or k % self.basis_every == 0:
            try:
                # eigh on CPU in float64 (a GPU eigh was followed by a cuBLAS error in one run)
                self.Q[n] = torch.linalg.eigh(A.double().cpu())[1].float().to(A.device)
            except Exception:
                return None
        Q = self.Q[n]
        g = p.grad[:, J].float() @ Q  # p.grad is the masked gradient; Adam's state already includes this step
        b2 = self.opt.param_groups[0]["betas"][1]
        # start from Adam's own second moment on J (diagonal in the old basis, as SOAP does at a basis change), so
        # with Q = I the direction is exactly Adam's
        v0 = self.v2[n] if n in self.v2 else self.opt.state[p]["exp_avg_sq"][:, J].float()
        v = b2 * v0 + (1 - b2) * g * g
        self.v2[n] = v
        return ((m @ Q) / (v.sqrt() + 1e-12)) @ Q.T

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kw):
        super().on_step_end(args, state, control, **kw)
        self._refresh_cols()
