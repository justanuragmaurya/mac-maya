---
license: apache-2.0
base_model: maya-research/maya2-native
pipeline_tag: text-to-speech
language: [as, bn, en, gu, hi, kn, ml, mr, or, pa, ta, te]
tags: [text-to-speech, tts, apple-silicon, mlx, metal, mps, macos, indic, voice-cloning]
---

# maya-mac: Maya TTS on Apple Silicon

Run [maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native) on a Mac. It covers 12 Indic languages
and English, has 21 built-in voices, and supports zero-shot voice cloning. The official `maya3` package needs an
NVIDIA GPU because it uses Triton kernels and CUDA graphs. This package is a rewrite of that inference path for Apple
Silicon: an [MLX](https://github.com/ml-explore/mlx) engine with custom Metal kernels (the default; ~3x faster than real
time on an M5 Pro, ~4x with 8-bit weights), and the original PyTorch MPS engine as a fallback.

This repo contains **code only**. The weights download from the original model repos the first time you run it
(~5.5 GB, cached by Hugging Face).

## Requirements

- Mac with Apple Silicon (M1 or newer), macOS 14+
- Python 3.10–3.12
- ~6 GB free disk
- RAM: tested on 24 GB. The MLX engine peaks at ~8.5 GB (fp16) or ~7 GB (`--bits 8`), so 16 GB Macs should work.

## Install

```bash
# pip (installs the prebuilt wheel from this repo):
pip install "maya-mac @ https://huggingface.co/justanurag/maya-mac/resolve/main/dist/maya_mac-0.2.0-py3-none-any.whl"

# or uv, straight from git (puts the `maya` command on your PATH):
uv tool install --python 3.12 "maya-mac @ git+https://huggingface.co/justanurag/maya-mac"
```

`pip install git+https://huggingface.co/...` does **not** work: the Hugging Face git server rejects the partial
clone that recent pip uses. Use the wheel URL above, or uv.

## Use

```bash
maya "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।" -l hi -v Ahana --normalize -o speech.wav
maya --list-voices
maya "Hello there." -l en --ref-audio me.wav --ref-text "exact transcript of me.wav"     # clone a voice
maya "Hello there." -l en --bits 8                     # 8-bit weights: ~1.4x faster again, see Speed
```

```python
import soundfile as sf
from maya_mac import MayaTTS

tts = MayaTTS.from_pretrained()          # first run downloads the weights; loads in ~10-20 s
wav = tts.generate("नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।", lang="hi", voice="Ahana", normalize=True)
sf.write("speech.wav", wav, 24000)       # float32 mono, 24 kHz

for chunk in tts.stream("A long paragraph ...", lang="en", voice="Vaidehi"):   # 240 ms chunks as they are made
    ...

tts.register_voice("me", ref_audio="me.wav", ref_text="exact transcript", lang="en")
wav = tts.generate("धन्यवाद!", lang="hi", voice="me")       # the reference and output languages can differ
```

- **Languages:** `as bn en gu hi kn ml mr or pa ta te`
- **Sampling options:** `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`
- **`normalize=True`:** expands numbers, currency, dates, URLs and similar written forms into spoken text.
- **Engine settings:** see `maya_mac.engine.EngineConfig`, and pass them as `from_pretrained(**settings)`. The MLX
  engine is used whenever MLX is installed on Apple Silicon; `backend="torch"` (or `device="mps"`) selects the
  PyTorch engine, and `backbone_bits=8, depth_bits=8` the 8-bit weights.
- **Local weights:** set `MAYA_MODEL_DIR` to a folder holding `maya2-native/`, `mimi/` and `ecapa/` to use those
  instead of downloading.

Only clone a voice with the speaker's permission. See the
[original model card](https://huggingface.co/maya-research/maya2-native) for intended use and limitations.

## Server

`maya-server` runs a server for other apps to use. It has an OpenAI-compatible speech endpoint and WebSocket
streaming, and uses the same protocol as the upstream `maya3` server, so existing maya3 clients and OpenAI-style TTS
clients work unchanged.

```bash
maya-server --port 8080            # first run downloads the weights; GET /health turns 200 when ready
maya-server --port 8080 --bits 8   # 8-bit weights

curl -s localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input": "नमस्ते, आप कैसे हैं?", "voice": "Ahana", "response_format": "mp3"}' -o hello.mp3
```

| Endpoint | What it does |
| --- | --- |
| `POST /v1/audio/speech` | OpenAI-compatible; `response_format` wav / pcm / mp3 / opus (streamed) or flac; `stream_format: "sse"`; `language` (guessed from the script if absent), `voice`, `normalize`, `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_frames` |
| `WS /v1/tts/stream` | Low-latency PCM streaming: `speak` / `cancel` / `ping` messages, several contexts per socket |
| `GET/POST /v1/voices`, `DELETE /v1/voices/{id}` | List voices, clone one from a wav (multipart `ref_audio` + `ref_text` + `lang`), remove it |
| `GET /health`, `GET /v1/models` | Readiness and stats; model list |

```python
from maya_mac.client import Maya3Client
Maya3Client("http://127.0.0.1:8080").speech("Hello there.", "out.wav", language="en", voice="Kayal")
```

**Requests run one at a time** in arrival order. The upstream server batches dozens of requests on one NVIDIA GPU,
but this one doesn't batch. If a request is still waiting for its first audio after `--queue-deadline-ms` (default
30 s), or more than `--max-pending` requests (default 4) are already waiting, the server refuses it with 503 /
`overloaded`. A client that disconnects or cancels frees the engine within about a quarter of a second. The server
listens on `127.0.0.1` by default; pass `--host 0.0.0.0` to expose it on your network.

## Speed (M5 Pro, 24 GB)

Measured back to back on the same machine, on battery in High Power mode. Real-time factor = generation time / audio
duration, so lower is faster.

| | torch MPS | MLX fp16 (default) | MLX `--bits 8` |
| --- | --- | --- | --- |
| Real-time factor, 21 s paragraph | 0.77 | 0.32 | 0.23 |
| Real-time factor, short sentences | 0.77–0.80 | 0.33 | 0.23 |
| Time per 80 ms frame | ~63 ms | 26.4 ms | 18.8 ms |
| First audio | 0.31–0.39 s | 0.16–0.18 s | 0.12–0.14 s |
| Load time, including warm-up (weights cached) | ~12 s | ~1 s | ~1 s |
| Whisper-small CER en / hi (18 utterances each) | 6.4 / 16.3 % | 6.4 / 16.3 % | 7.0 / 13.3 % |

In Low Power Mode the GPU drops to about half its clock after a second of sustained load, and the paragraph's
real-time factor becomes 1.41 / 0.63 / 0.44.

The fp16 engine computes the same function as the torch engine: against the model in fp32 its hidden states, cb0 logits
and depth logits are within 0.1–0.25 % (torch MPS fp16: ~0.4 %). 8-bit weights move the logits by ~2–3 %, about as much
as bf16 does; transcription error is unchanged within noise. `tests/test_mlx_parity.py` checks both.

## How it works

Model, prompt format, sampler, stop rule and guards are the same as upstream. This is the inference code only.

Per 80 ms frame the engine runs one Qwen3-1.7B decode step and then a 31-level depth decoder, i.e. 124 tiny transformer
layer passes, each depending on the one before. At one stream that is bound by memory bandwidth (~6 GB of weights read
per frame in fp16) and by the latency of dependent GPU kernels, not by FLOPs, so the engine is built to read fewer bytes
and launch fewer kernels:

- **MLX instead of PyTorch MPS.** MLX's fused Metal kernels (`rms_norm`, `rope`, `scaled_dot_product_attention`,
  quantized matmul) and lazy graphs, with no per-op Python dispatch for the depth decoder: the cb0 sampler, all 31 depth
  levels and the feedback embedding are one `mx.compile`d graph.
- **Custom Metal kernels for the depth decoder** (`mlx_kernels.py`). A layer pass goes from ~12 dependent kernels to 5:
  LayerNorm folded into the matvec (one pass, no barrier), GELU / residual / speaker-constant epilogues, KV append +
  causal attention in one kernel, and top-k (exact radix select) + Gumbel-max draw + next-level embedding in one kernel.
  The fp16 depth step drops from ~27 ms to ~16-21 ms.
- **Pipelined frames.** Frame t+1 is queued on the GPU before the host reads frame t (stop flag + 32 codes, one read),
  so guards and the hand-off to Mimi overlap GPU work. A speculative frame is dropped when frame t stops, and recomputed
  when the loop guard changes the AEOS bias, so the output is what the sequential loop would produce.
- **Voice prefix KV cache.** A cloned voice's reference clip is prefilled once; later requests restore its KV and only
  prefill their own text.
- **Optional 8-bit weights** for the backbone and the depth decoder: about half the bytes per frame.

| Module | What it is |
| --- | --- |
| `engine.py` | One request at a time: prompt → backbone → first-code sampling and stop rule → depth decoder → feedback. Also the duration and loop guards, text splitting and voices; picks the backend |
| `mlx_engine.py` | The MLX engine (`MayaMLX`, a `MayaTTS`): pipelined frame loop, voice prefix cache, all MLX work on one thread it owns |
| `mlx_model.py` | The model in MLX, loaded straight from the release safetensors: Qwen3 backbone, compiled frame step (cb0 sampler, depth decoder, feedback), prompt composition |
| `mlx_kernels.py` | Custom Metal kernels for the depth decoder: LayerNorm + matvec + epilogues (fp16 / 8-bit), tiny causal attention, top-k sampler |
| `fast_backbone.py`, `fast_depth.py` | The PyTorch MPS engine's lean backbone and depth decoder |
| `codec.py` | Streaming Mimi decoder; runs on the CPU in a worker thread |
| `serving.py` | Queued engine for the server: one worker thread owns the model, runs requests in order, and handles cancellation and voice registration |
| `server.py`, `client.py` | Upstream maya3 server and client, unchanged protocol, wired to `serving.py` |
| `tts/`, `prompt.py`, `guards.py`, `text/`, `reference.py`, `configs/` | Vendored from [MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native) (Apache-2.0, `LICENSE.upstream`) |

MLX findings (0.32.4):

- Streams are thread-local: arrays and compiled graphs built on one thread fail on another ("There is no Stream(gpu, 1)
  in current thread"). The engine therefore does all MLX work on one thread of its own.
- `audio_in.norm` is `nn.RMSNorm(eps=None)`, whose epsilon follows the input dtype (fp16: 9.8e-4). That is ~35 % of a
  feedback frame's mean square, so the MLX engine uses the same epsilon rather than a fixed one.
- A one-pass matvec kernel that re-reads its input rows is fine for a few positions and 7x slower than MLX's tiled
  GEMM at 18-26 positions, so the long first passes of the depth bands use MLX's matmul.

MPS findings with torch 2.8.0:

- `scaled_dot_product_attention` returns wrong attention for this backbone in fp16/bf16 (~95 % relative error), so
  the model babbles or stops early. The backbone uses its own attention, which is within 0.4 % of fp32.
- HF's Qwen3 RMSNorm takes 133 µs per call on MPS, and `F.rms_norm` takes 6 µs for the same output.
- Indexing with a 0-dim device tensor forces a host sync (~0.7 ms). `F.embedding` avoids it.

## Tests

```bash
uv run --group dev pytest tests/                                  # Metal kernels vs MLX ops (no weights needed)
MAYA_MODEL_DIR=models uv run --group dev pytest tests/            # + MLX vs torch fp32 parity on the real weights
```

## Licence

Apache-2.0, like the upstream code and weights. Weights: © Maya Research, see
[maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native).
