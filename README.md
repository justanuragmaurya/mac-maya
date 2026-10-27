---
license: apache-2.0
base_model: maya-research/maya2-native
pipeline_tag: text-to-speech
language: [as, bn, en, gu, hi, kn, ml, mr, or, pa, ta, te]
tags: [text-to-speech, tts, apple-silicon, mps, macos, indic, voice-cloning]
---

# maya-mac: Maya TTS on Apple Silicon

Run [maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native) on a Mac. It covers 12 Indic languages
and English, has 21 built-in voices, and supports zero-shot voice cloning. The official `maya3` package needs an
NVIDIA GPU because it uses Triton kernels and CUDA graphs. This package is a rewrite of that inference path for
Apple's Metal (MPS) backend.

This repo contains **code only**. The weights download from the original model repos the first time you run it
(~5.5 GB, cached by Hugging Face).

## Requirements

- Mac with Apple Silicon (M1 or newer), macOS 14+
- Python 3.10–3.12
- ~6 GB free disk
- RAM: tested on 24 GB. Loading peaks at ~12 GB and steady state is ~6 GB of GPU memory, so 16 GB Macs may work but are
  untested.

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
- **Engine settings:** see `maya_mac.engine.EngineConfig`, and pass them as `from_pretrained(**settings)`.
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

| | |
| --- | --- |
| Real-time factor, short sentences | ~0.95–1.0 (about real time) |
| Real-time factor, 40 s paragraph | ~1.04 |
| First audio | ~0.4 s |
| Load time, including warm-up | ~10–20 s |

Older or smaller chips will be slower than real time. `generate()` still works at any speed, but live streaming
may stutter.

## How it differs from the CUDA engine

Model, prompt format, sampler, stop rule and guards are the same as upstream. This is the inference code only:

| Module | What it is |
| --- | --- |
| `engine.py` | One request at a time: prompt → backbone → first-code sampling and stop rule → depth decoder → feedback. Also the duration and loop guards, text splitting and voices |
| `fast_backbone.py` | Lean Qwen3-1.7B decode step: preallocated KV cache, fused q\|k\|v and gate\|up, fused RMSNorm, attention written as matmuls |
| `fast_depth.py` | Depth decoder (codebook levels 1–31): per-group KV cache, speaker cross-attention as a constant, residual adds fused into the matmuls |
| `codec.py` | Streaming Mimi decoder; runs on the CPU in a worker thread |
| `serving.py` | Queued engine for the server: one worker thread owns the model, runs requests in order, and handles cancellation and voice registration |
| `server.py`, `client.py` | Upstream maya3 server and client, unchanged protocol, wired to `serving.py` |
| `tts/`, `prompt.py`, `guards.py`, `text/`, `reference.py`, `configs/` | Vendored from [MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native) (Apache-2.0, `LICENSE.upstream`) |

MPS findings with torch 2.8.0, each measured while building this:

- `scaled_dot_product_attention` returns wrong attention for this backbone in fp16/bf16 (~95 % relative error), so
  the model babbles or stops early. The backbone uses its own attention, which is within 0.4 % of fp32.
- HF's Qwen3 RMSNorm takes 133 µs per call on MPS, and `F.rms_norm` takes 6 µs for the same output.
- Indexing with a 0-dim device tensor forces a host sync (~0.7 ms). `F.embedding` avoids it.

## Licence

Apache-2.0, like the upstream code and weights. Weights: © Maya Research, see
[maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native).
