# export_mla_kbT_gguf.py -- sidecar GGUF with the transposed key projection per layer for the compressed MLA path:
#   blk.L.attn_k_bT.weight, Q8_0, ggml ne = (128 nope, 512 latent, 128 heads), built from the shards' attn_kv_b.
#   The runtime adds this file to its shard list under --mla and streams the tensor like any always-used weight,
#   which replaces the per-layer on-GPU dequantisation (~0.18 s per pass).
import os, sys, glob, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, os.environ.get("GGUF_PY", "gguf-py")); sys.dont_write_bytecode = True
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize
from gguf.constants import GGMLQuantizationType
out = sys.argv[1] if len(sys.argv) > 1 else "ptx_trees/mla_kbT.gguf"
H, NOPE, LAT = 128, 128, 512
tensors = {}
for fp in sorted(glob.glob(os.path.join(os.environ.get("GGUF_DIR", "models"), "*.gguf"))):
    R = GGUFReader(fp)
    for t in R.tensors:
        if t.name.endswith("attn_kv_b.weight"): tensors[t.name] = t
w = GGUFWriter(out, "deepseek2-mla-sidecar"); t0 = time.time()
for L in range(61):
    t = tensors[f"blk.{L}.attn_kv_b.weight"]
    W = np.asarray(dequantize(t.data, t.tensor_type), np.float32).reshape(H * 256, LAT)   # rows = outputs (per head 128 k_nope then 128 v), 512 inputs
    kbT = np.ascontiguousarray(np.transpose(W.reshape(H, 256, LAT)[:, :NOPE, :], (0, 2, 1)))   # [head, latent j, nope i] -> ggml ne0 = 128 (i), ne1 = 512 (j), ne2 = 128
    q = quantize(kbT.reshape(H * LAT, NOPE), GGMLQuantizationType.Q8_0)
    w.add_tensor(f"blk.{L}.attn_k_bT.weight", q.reshape(H, LAT, -1), raw_dtype=GGMLQuantizationType.Q8_0)   # byte shape (rows x 136 B); gguf derives ne = (128, 512, 128)
    if L % 15 == 0: print(f"layer {L} ({time.time() - t0:.0f}s)")
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {out}: {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f}s")
R = GGUFReader(out); t = [x for x in R.tensors if x.name == "blk.3.attn_k_bT.weight"][0]; print("check:", t.name, list(t.shape), t.tensor_type.name)
