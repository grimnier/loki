# export_trim_gguf.py -- sidecar GGUF with the trimmed always-used tensors for the fused engine (server_core_fused --fused 1):
#   blk.L.attn_kv_b_v.weight   Q6_K, ne = (512, 16384): the v half of attn_kv_b (per head rows 128..255 of its 256-row slab), the only
#                              part the sidecar MLA path reads (the k_nope half lives in attn_k_bT already): 13.8 -> 6.9 MB per layer
#   blk.L.ffn_gate_inp_f16.weight  F16, ne = (7168, 256): the router weights (F32 in the shards) in half precision: 7.3 -> 3.7 MB (layers 3-60)
#   The engine adds this file to its shard list under --fused and swaps the two names in each layer's always-used list.
import os, sys, glob, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, os.environ.get("GGUF_PY", "gguf-py")); sys.dont_write_bytecode = True
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGMLQuantizationType
out = sys.argv[1] if len(sys.argv) > 1 else "ptx_trees/mla_trim.gguf"
H, LAT, ROWB = 128, 512, 420   # Q6_K row of 512 = 2 blocks x 210 B
tensors = {}
for fp in sorted(glob.glob(os.path.join(os.environ.get("GGUF_DIR", "models"), "*.gguf"))):
    R = GGUFReader(fp)
    for t in R.tensors:
        if t.name.endswith("attn_kv_b.weight") or t.name.endswith("ffn_gate_inp.weight"): tensors[t.name] = t
w = GGUFWriter(out, "deepseek2-fused-trim"); t0 = time.time(); total = 0
for L in range(61):
    t = tensors[f"blk.{L}.attn_kv_b.weight"]; assert t.tensor_type == GGMLQuantizationType.Q6_K, t.tensor_type
    raw = np.asarray(t.data).reshape(-1); assert raw.size == H * 256 * ROWB, raw.size
    v = np.ascontiguousarray(raw.reshape(H, 256, ROWB)[:, 128:, :]).reshape(H * 128, ROWB)   # rows 128..255 of each head
    w.add_tensor(f"blk.{L}.attn_kv_b_v.weight", v, raw_dtype=GGMLQuantizationType.Q6_K); total += v.nbytes
    if L >= 3:
        g = tensors[f"blk.{L}.ffn_gate_inp.weight"]; assert g.tensor_type == GGMLQuantizationType.F32, g.tensor_type
        gf = np.asarray(g.data, dtype=np.float32).reshape(256, 7168).astype(np.float16)
        w.add_tensor(f"blk.{L}.ffn_gate_inp_f16.weight", gf); total += gf.nbytes
    if L % 15 == 0: print(f"layer {L} ({time.time() - t0:.0f}s)")
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {out}: {os.path.getsize(out) / 1e6:.0f} MB ({total / 1e6:.0f} MB of tensors) in {time.time() - t0:.0f}s")
R = GGUFReader(out)
for nm in ("blk.3.attn_kv_b_v.weight", "blk.3.ffn_gate_inp_f16.weight", "blk.0.attn_kv_b_v.weight"):
    t = [x for x in R.tensors if x.name == nm][0]; print("check:", t.name, list(t.shape), t.tensor_type.name, t.data.nbytes)
# value check on layer 3: the v rows of head 5 must equal the original rows 5*256+128.. of attn_kv_b, and the f16 router weights round-trip
t = tensors["blk.3.attn_kv_b.weight"]; raw = np.asarray(t.data).reshape(H * 256, ROWB); tv = [x for x in R.tensors if x.name == "blk.3.attn_kv_b_v.weight"][0]
vv = np.asarray(tv.data).reshape(H * 128, ROWB); print("v rows identical:", bool(np.array_equal(vv[5 * 128:(5 + 1) * 128], raw[5 * 256 + 128:5 * 256 + 256])))
g = np.asarray(tensors["blk.3.ffn_gate_inp.weight"].data, dtype=np.float32).reshape(-1); tg = [x for x in R.tensors if x.name == "blk.3.ffn_gate_inp_f16.weight"][0]
gg = np.asarray(tg.data).reshape(-1).astype(np.float32); print("gate_inp f16 max |diff| %.3e of max |w| %.3e" % (np.abs(gg - g).max(), np.abs(g).max()))
