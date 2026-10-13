# loki

DeepSeek-R1 671B running whole on a 6 GB laptop GPU, with a terminal client to talk to it.

Every layer of the model runs as hand-written CUDA inside one process: tensor-core attention, a GEMM-shaped
quantised matvec, the 256 routed experts per layer streamed from the drive into a VRAM arena, and a geometry
memory that recognises states it has walked before and serves them instead of re-reading the weights. On an
RTX 4050 laptop (6 GB) with 16 GB of RAM and an NVMe drive it decodes at about 1 token/s on new ground and
3 to 12 on ground the memory knows, and prefills a 600-token prompt in about 40 s. `GEOMETRY.md` is the
account of what the memory does.

## Requirements

- Windows, an NVIDIA GPU (built for sm_89), CUDA Toolkit 13.x, Visual Studio 2022 Build Tools.
- A CUDA build of llama.cpp's ggml: its include directory, its `ggml.lib` / `ggml-base.lib` / `ggml-cuda.lib`,
  and its runtime DLLs. Point `GGML_INC`, `GGML_LIB` and `GGML_BIN` at them (see `build.bat`, `run.bat`).
- The model: the `DeepSeek-R1-UD-IQ2_XXS` GGUF shards in `models/` (or `--shards <dir>`), plus the two side
  files made from them by `tools/export_mla_kbT_gguf.py` and `tools/export_trim_gguf.py` into `ptx_trees/`
  (`GGUF_DIR` = the shards, `GGUF_PY` = llama.cpp's `gguf-py`).
- Python 3.12 for the client and the tools (standard library only for the client).

## Build and run

```bat
build.bat
run.bat --prompt "What color is the sky?" --tokens 40 --experts 2 --mla --fused 1 --prefill-chunk 512 --pinned-gb 6
loki
```

`loki` (`bin\loki.cmd`) starts the terminal client on the engine; `chat.py` is the line client.
`chat_settings.json` (created on first run) holds the context size, the think and answer budgets, the memory
mode and the sampling dials (`temp`, `top_k`, `top_p`, `seed`, `stop`).

## Layout

| path | what |
|---|---|
| `src/` | the engine: `server_core_fused.cpp` (streaming, memory, chat protocol) and `fused_layer.cu` (every layer as CUDA) |
| `loki/`, `chat.py`, `bin/` | the clients |
| `tools/` | the exporters for the model's side files |
| `include/`, `third_party/` | the tokenizer header, nlohmann/json |
| `GEOMETRY.md` | the geometry memory, written out |

Weights, side files, memory files and results stay outside the repository.
