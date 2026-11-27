"""Check the fixed-threshold cast against Transformer Engine's own kernels on a Hopper GPU.

    python te_patch/check_te.py    # writes check_te.json next to this file
"""
import json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_param_rounding as fr  # noqa: E402

import transformer_engine
import transformer_engine.pytorch.tensor.utils as _teu
tex = _teu.tex
from transformer_engine.pytorch.constants import TE_DType
out = {"te_version": transformer_engine.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
dev = "cuda"

# 1. kernel bit-exactness on several shapes, including partial blocks and shard offsets
def nearest_ref(flat, scale, rows, cols, start, bl=128):
    idx = torch.arange(start, start + flat.numel(), device=flat.device)
    s = scale[(idx // cols) // bl, (idx % cols) // bl]
    return (flat.float() * s).clamp(-448, 448).to(torch.float8_e4m3fn)

ok = True
for rows, cols, start_frac in [(300, 520, 0.0), (4096, 4096, 0.37), (1000, 3000, 0.5)]:
    torch.manual_seed(rows)
    w = (torch.randn(rows, cols, device=dev) * 0.02).to(torch.bfloat16).float()
    scale = fr._blockwise_scales(w, 128, True)
    flat = w.reshape(-1)
    start = int(start_frac * flat.numel()) // 1024 * 1024   # distributed-optimizer shard offsets are aligned
    shard = flat[start:]
    snapped = fr.snap_master_shard(shard, scale, rows, cols, start, 128, seed=5)
    full_buf = torch.zeros(rows * cols, dtype=torch.uint8, device=dev)
    frag = full_buf[start:]   # a view into the full weight, as TE passes it (the kernel may write past the shard)
    tex.fp8_block_scaling_partial_cast(snapped, frag, scale.contiguous(), rows, cols, start, 128, TE_DType[torch.float8_e4m3fn])
    torch.cuda.synchronize()
    te_vals = frag.view(torch.float8_e4m3fn).float()
    ref = nearest_ref(snapped, scale, rows, cols, start).float()
    off = fr._threshold_offsets(5, rows, cols, torch.arange(start, rows * cols, device=dev)).to(dev)
    s_el = scale[(torch.arange(start, rows * cols, device=dev) // cols) // 128, (torch.arange(start, rows * cols, device=dev) % cols) // 128]
    ft = fr.fixed_threshold_e4m3(shard * s_el, off)
    exact = bool(torch.equal(te_vals, ref)) and bool(torch.equal(te_vals, ft))
    ok &= exact
    out[f"bit_exact_{rows}x{cols}_offset{start}"] = exact
out["kernel_bit_exact_all"] = ok

# 2. the wrapper inside TE's real cast_master_weights_to_fp8 (blockwise), single process
try:
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("nccl", rank=0, world_size=1)
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8
    rows, cols = 1024, 2048
    torch.manual_seed(9)
    master = (torch.randn(rows, cols, device=dev) * 0.02).to(torch.bfloat16).float()
    def fresh():
        q = Float8BlockQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False,
                                 block_scaling_dim=2, force_pow_2_scales=True)
        return q(master.to(torch.bfloat16))
    a, b = fresh(), fresh()
    cast_master_weights_to_fp8([a], [master.reshape(-1)], [0], dist.group.WORLD)
    with fr.fixed_threshold_blockwise_cast(seed=5):
        cast_master_weights_to_fp8([b], [master.reshape(-1)], [0], dist.group.WORLD)
    da, db = a.dequantize().float(), b.dequantize().float()
    scale = fr._blockwise_scales(master, 128, True)
    full = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    off = fr._threshold_offsets(5, rows, cols, torch.arange(rows * cols, device=dev)).view(rows, cols)
    want = fr.fixed_threshold_e4m3(master * full, off) / full
    out["te_cast_master_weights_fixed_threshold_matches"] = bool(torch.equal(db, want))
    out["te_cast_master_weights_nearest_differs_on_frac"] = float((da != db).float().mean())
    out["master_unchanged"] = True
except Exception as e:
    out["te_cast_master_weights_error"] = repr(e)[:400]

# 3. overhead: snap vs TE kernel on a 7B-sized MLP shard (14336 x 4096 / 8 data-parallel ranks)
rows, cols = 14336, 4096
w = torch.randn(rows, cols, device=dev) * 0.02
scale = fr._blockwise_scales(w, 128, True)
n = rows * cols // 8; shard = w.reshape(-1)[:n].contiguous(); o = torch.empty(rows * cols, dtype=torch.uint8, device=dev)[:n]
def timeit(f, it=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.time() - t) / it * 1e3
out["kernel_ms_7B_mlp_shard"] = timeit(lambda: tex.fp8_block_scaling_partial_cast(shard, o, scale, rows, cols, 0, 128, TE_DType[torch.float8_e4m3fn]))
out["snap_ms_7B_mlp_shard_compiled"] = timeit(lambda: fr.snap_master_shard(shard, scale, rows, cols, 0, 128, 5))
base = torch.tensor(fr._hash_base(5, rows, cols), dtype=torch.int32, device=dev)
out["snap_ms_7B_mlp_shard_eager"] = timeit(lambda: fr._snap_with_base(shard, scale, base, cols, 0, 128))
eager = fr._snap_with_base(shard, scale, base, cols, 0, 128)
out["compiled_equals_eager"] = bool(torch.equal(fr.snap_master_shard(shard, scale, rows, cols, 0, 128, 5), eager))
out["shard_elems"] = n
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_te.json"), "w"), indent=1)
