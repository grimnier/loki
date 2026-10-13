# export_mla_kb.py -- one-time sidecar for the compressed (absorbed) MLA attention path:
#   per layer, the transposed key projection k_b^T per head, Q8_0, ggml layout ne = (128 nope, 512 latent, 128 heads),
#   built from the shards' combined attn_kv_b (512 -> 128 heads x [128 k_nope | 128 v]).
#   file: magic "MLAK", int32 count, then per tensor: int32 name_len, name bytes, int32 type, int64 ne[3], int64 offset, int64 nbytes; data follows the table
import os, sys, glob, struct, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, os.environ.get("GGUF_PY", "gguf-py")); sys.dont_write_bytecode = True
from gguf import GGUFReader
from gguf.quants import dequantize, quantize
from gguf.constants import GGMLQuantizationType
out = sys.argv[1] if len(sys.argv) > 1 else "ptx_trees/mla_kbT.bin"
H, NOPE, LAT = 128, 128, 512
tensors = {}
for fp in sorted(glob.glob(os.path.join(os.environ.get("GGUF_DIR", "models"), "*.gguf"))):
    R = GGUFReader(fp)
    for t in R.tensors:
        if t.name.endswith("attn_kv_b.weight"): tensors[t.name] = t
print(len(tensors), "attn_kv_b tensors")
table, blobs, off, t0 = [], [], 0, time.time()
for L in range(61):
    t = tensors[f"blk.{L}.attn_kv_b.weight"]
    W = np.asarray(dequantize(t.data, t.tensor_type), np.float32).reshape(H * 256, LAT)   # rows = outputs (per head: 128 k_nope rows then 128 v rows), 512 inputs
    Wk = W.reshape(H, 256, LAT)[:, :NOPE, :]                                               # [head, i (nope), j (latent)]
    kbT = np.ascontiguousarray(np.transpose(Wk, (0, 2, 1)))                                # [head, j, i]: ggml ne0 = 128 (i), ne1 = 512 (j), ne2 = 128 heads
    q = quantize(kbT.reshape(H * LAT, NOPE), GGMLQuantizationType.Q8_0)                   # row-wise Q8_0 over the 128-wide rows
    b = q.tobytes(); table.append((f"blk.{L}.attn_k_bT.weight", int(GGMLQuantizationType.Q8_0), (NOPE, LAT, H), off, len(b))); blobs.append(b); off += len(b)
    if L % 10 == 0: print(f"layer {L}: kbT {kbT.shape} -> {len(b)/1e6:.1f} MB Q8_0 ({time.time()-t0:.0f}s)")
hdr = bytearray(b"MLAK" + struct.pack("<i", len(table)))
for name, ty, ne, o, nb in table:
    nm = name.encode(); hdr += struct.pack("<i", len(nm)) + nm + struct.pack("<i", ty) + struct.pack("<3q", *ne) + struct.pack("<q", o) + struct.pack("<q", nb)
base = len(hdr)
with open(out, "wb") as f:
    f.write(hdr)
    for b in blobs: f.write(b)
print(f"wrote {out}: {len(table)} tensors, data base {base} B, total {(base + off) / 1e9:.2f} GB in {time.time() - t0:.0f}s")
