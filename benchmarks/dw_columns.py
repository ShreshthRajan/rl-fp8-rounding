"""Time of the weight-gradient GEMM restricted to a subset of input columns.

    python benchmarks/dw_columns.py [tokens]    # default 16384; writes results/benchmarks/dw_columns.json

dW = dY^T X (out x tokens)(tokens x in). The column-restricted version computes dW[:, J] = dY^T X[:, J] for a set J
of input columns: gather X[:, J] (index_select), then one cuBLAS GEMM into a compact (out x |J|) buffer that the
optimizer scatters. Shapes are the per-layer matrices of Qwen2.5-0.5B and Qwen2.5-7B. Times are medians of 50 runs,
bf16 inputs with fp32 accumulation (torch default), CUDA events, including the gather.
Also times dX = dY W (needed in any case), so the share of the backward pass saved is reported.
"""
import json, os, sys, torch
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
SHAPES = {
    "0.5B": {"qkv": (896 + 2 * 128, 896), "o": (896, 896), "gate_up": (2 * 4864, 896), "down": (896, 4864)},
    "7B": {"qkv": (3584 + 2 * 512, 3584), "o": (3584, 3584), "gate_up": (2 * 18944, 3584), "down": (3584, 18944)},
}
TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
FRACS = (1.0, 0.5, 0.2, 0.05)


def timeit(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


out = {"tokens": TOKENS, "gpu": torch.cuda.get_device_name(0), "rows": []}
for model, shapes in SHAPES.items():
    tot = {f: 0.0 for f in FRACS}; tot_dx = 0.0
    for name, (o, i) in shapes.items():
        X = torch.randn(TOKENS, i, device=dev, dtype=torch.bfloat16)
        dY = torch.randn(TOKENS, o, device=dev, dtype=torch.bfloat16)
        W = torch.randn(o, i, device=dev, dtype=torch.bfloat16)
        tdx = timeit(lambda: dY @ W)
        tot_dx += tdx
        for f in FRACS:
            if f == 1.0:
                t = timeit(lambda: dY.t() @ X)
            else:
                J = torch.randperm(i, device=dev)[: max(1, int(f * i))].sort().values
                t = timeit(lambda: dY.t() @ X.index_select(1, J))
            tot[f] += t
            out["rows"].append({"model": model, "matrix": name, "shape": [o, i], "frac": f, "ms_dw": t, "ms_dx": tdx})
        del X, dY, W; torch.cuda.empty_cache()
    for f in FRACS:
        saved_bwd = (tot[1.0] - tot[f]) / (tot[1.0] + tot_dx)
        out["rows"].append({"model": model, "summary": True, "frac": f, "ms_dw_total": tot[f], "ms_dw_dense": tot[1.0],
                            "ms_dx_total": tot_dx, "dw_speedup": tot[1.0] / tot[f], "backward_matmul_time_saved": saved_bwd})
        print(model, f, f"dW {tot[f]:.2f} ms vs dense {tot[1.0]:.2f} ms ({tot[1.0]/tot[f]:.2f}x); dX {tot_dx:.2f} ms;"
              f" backward matmul time saved {100*saved_bwd:.1f}%", flush=True)
os.makedirs("results/benchmarks", exist_ok=True)
json.dump(out, open("results/benchmarks/dw_columns.json", "w"), indent=1)
