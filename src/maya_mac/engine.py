"""Single-request Maya inference engine for Apple Silicon (MPS) and CPU.

Same model, prompt geometry, cb0 sampler, controlled stop and guards as the upstream CUDA engine
(maya3/engine.py), minus everything that only exists on NVIDIA: no Triton kernels, no CUDA graphs, no seat batching.

Per frame (80 ms of audio):
  1. backbone : one Qwen3 decode step over the HF KV cache -> hidden h
  2. cb0      : cb0 logits -> presence penalty / temperature / top-k / top-p -> controlled stop + sample (host side)
  3. depth    : levels 1..31 from h and cb0 (FastDepth: per-group KV cache, constant speaker cross-attention)
  4. feedback : the frame's 32 codes are embedded and fed back to the backbone
Mimi decodes the codes to 24 kHz audio (whole piece, or streamed by StreamingMimi).
"""
from __future__ import annotations

import collections
import dataclasses
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from maya_mac import guards as G  # noqa: E402
from maya_mac import reference as S  # noqa: E402
from maya_mac.prompt import DEFAULT_BUILDER, MODES, Voice, VoiceRegistry, load_named_voices, norm_mode  # noqa: E402
from maya_mac.tts.tokens import AEOS_CLASS, N_Q  # noqa: E402

SR = 24000
FRAME = 1920
HERE = Path(__file__).resolve().parent
# where each component comes from: a Hugging Face repo id, the files to fetch, and the files that must exist after
REPOS = {
    "maya2-native": ("maya-research/maya2-native", ["*.safetensors", "*.safetensors.index.json", "maya3_config.json",
                                                    "backbone/*", "voices/*", "LICENSE"],
                     ["model.safetensors.index.json", "maya3_config.json", "backbone/config.json",
                      "backbone/tokenizer.json"]),
    "mimi": ("kyutai/mimi", ["config.json", "model.safetensors", "preprocessor_config.json"],
             ["config.json", "model.safetensors"]),
    "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", ["embedding_model.ckpt", "hyperparams.yaml"],
              ["embedding_model.ckpt"]),
}


def _missing(d, required):
    d = Path(d)
    out = [f for f in required if not (d / f).is_file()]
    index = d / "model.safetensors.index.json"
    if index.is_file():      # every shard the index names
        import json
        out += sorted({f for f in json.loads(index.read_text())["weight_map"].values() if not (d / f).is_file()})
    return out


def resolve_models(models_dir=None):
    """-> {component: local dir}. models_dir (or $MAYA_MODEL_DIR) holding maya2-native/, mimi/ and ecapa/ is used as
    is; otherwise each component is fetched into the Hugging Face cache (files already cached are not re-downloaded;
    with no network, or HF_HUB_OFFLINE=1, the cache alone is used)."""
    models_dir = models_dir or os.environ.get("MAYA_MODEL_DIR")
    if models_dir:
        d = Path(models_dir).expanduser()
        out = {n: d / n for n in REPOS}
    else:
        from huggingface_hub import snapshot_download
        out = {}
        for name, (repo, patterns, _) in REPOS.items():
            try:
                out[name] = Path(snapshot_download(repo, allow_patterns=patterns))
            except Exception:   # offline: whatever the cache holds (checked below)
                out[name] = Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
    for name, (repo, _, required) in REPOS.items():
        gone = _missing(out[name], required)
        if gone:
            raise FileNotFoundError(f"{name} ({repo}) at {out[name]} lacks {gone}")
    return out


def mlx_available():
    import platform
    import sys
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclasses.dataclass
class EngineConfig:
    depth_temp: float = 0.7              # depth sampler (upstream EngineConfig.temp / top_k)
    depth_top_k: int = 64
    depth_dtype: str = "fp16"            # fp16 | bf16 | fp32 (upstream default fp16)
    backbone_dtype: str = "fp16"         # bf16 | fp16 | fp32 (CPU always runs fp32)
    # MPS SDPA returns wrong attention for this model in fp16/bf16 (~95 % relative error on the hidden states,
    # measured with torch 2.8.0); eager attention is accurate (fp16: 0.4 %, bf16: 2.5 % vs fp32)
    backbone_attn: str = "auto"          # auto (eager on MPS, sdpa elsewhere) | eager | sdpa
    cb0_temp: float = 0.9                # cb0 sampler defaults = upstream sweep winner
    cb0_top_k: int = 64
    cb0_top_p: float = 1.0
    cb0_rep: float = 0.8                 # presence penalty over the last cb0_rep_win frames
    cb0_rep_win: int = 12
    hard_cap: int = 1500                 # frames (120 s) per piece
    capacity: int = 4096                 # backbone KV positions (prompt + generated frames of one piece)
    gap_s: float = 0.15                  # silence between the text pieces of one request
    chunk_s: float = 20.0                # long text is split into pieces of about this many seconds
    stream_frames: int = 3               # frames per streamed audio chunk (240 ms)
    codec_device: str = "cpu"            # Mimi decode/encode device
    backend: str = "auto"                # auto (mlx when installed on Apple Silicon, else torch) | mlx | torch
    backbone_bits: int | None = None     # mlx only: 8 | 4 = quantized backbone weights (None: backbone_dtype)
    depth_bits: int | None = None        # mlx only: 8 | 4 = quantized depth decoder weights (None: depth_dtype)
    warmup: bool = True                  # run a short generation at load
    seed: int = 20261024
    guards: G.GuardConfig = dataclasses.field(default_factory=G.GuardConfig)


def load_audio(x, sr=None):
    """path | (array, sr) | array (assumed 24 kHz) -> float32 mono 24 kHz."""
    if isinstance(x, (str, os.PathLike)):
        import soundfile as sf
        w, sr = sf.read(str(x), dtype="float32")
    elif isinstance(x, tuple):
        w, sr = x
    else:
        w = x
    w = np.asarray(w, np.float32)
    if w.ndim > 1:
        w = w.mean(1) if w.shape[1] < w.shape[0] else w.mean(0)
    sr = sr or SR
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        w = resample_poly(w, SR // g, int(sr) // g).astype(np.float32)
    return w


class Ecapa:
    """ECAPA-TDNN speaker embedding (speechbrain spkrec-ecapa-voxceleb; 16 kHz fbank-80, sentence norm)."""

    def __init__(self, ecapa_dir, device):
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization
        self.fbank = Fbank(n_mels=80).eval().to(device)
        self.norm = InputNormalization(norm_type="sentence", std_norm=False).eval().to(device)
        emb = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                         dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192).eval()
        emb.load_state_dict(torch.load(Path(ecapa_dir) / "embedding_model.ckpt", map_location="cpu", weights_only=True),
                            strict=True)
        self.emb = emb.to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, w24):
        from scipy.signal import resample_poly
        if w24 is None or len(w24) < 2400:
            raise ValueError("reference audio is too short (need >= 0.1 s; 3-10 s recommended)")
        w16 = torch.from_numpy(resample_poly(np.asarray(w24, np.float32), 2, 3).astype(np.float32)).to(self.device)[None]
        L = torch.ones(1, device=self.device)
        e = self.emb(self.norm(self.fbank(w16), L), L).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()


class Cb0Sampler:
    """Upstream Engine._cb0 for one request, on the host: presence penalty, temperature, top-k, top-p, then the
    controlled stop (argmax AEOS / p_aeos >= 0.35 / hazard >= 0.6, after `minimum` frames) and a draw with AEOS removed."""

    def __init__(self, cfg: EngineConfig, minimum, sampling, gen):
        self.temp = float(sampling.get("temperature") or cfg.cb0_temp)
        self.top_k = int(sampling.get("top_k") or cfg.cb0_top_k)
        self.top_p = float(sampling.get("top_p") or cfg.cb0_top_p)
        rp = sampling.get("repetition_penalty")
        self.rep = float(rp if rp is not None else cfg.cb0_rep)
        self.hist = np.full(max(1, cfg.cb0_rep_win), -1, np.int64)
        self.win = cfg.cb0_rep_win
        self.minimum = minimum
        self.hazard = 0.0
        self.frames = 0
        self.aeos_bias = 0.0
        self.gen = gen

    def __call__(self, logits):
        """logits: (2049,) float32 CPU tensor -> (stop reason or None, cb0)."""
        logits = logits / self.temp
        logits[AEOS_CLASS] += self.aeos_bias
        if self.win > 0 and self.rep > 0:
            seen = self.hist[self.hist >= 0]
            if seen.size:
                seen = torch.from_numpy(np.unique(seen))
                seen = seen[seen != AEOS_CLASS]
                logits[seen] -= self.rep
        k = min(self.top_k, logits.shape[0])
        kth = logits.topk(k).values[-1]
        logits = logits.masked_fill(logits < kth, -float("inf"))
        if self.top_p < 1.0:
            sp, si = logits.softmax(-1).sort(descending=True)
            drop = (sp.cumsum(-1) - sp) > self.top_p
            logits[si[drop]] = -float("inf")
        probs = logits.softmax(-1)
        pa = float(probs[AEOS_CLASS])
        stop = None
        if self.frames >= self.minimum:
            self.hazard = 1 - (1 - self.hazard) * (1 - pa)
            if int(probs.argmax()) == AEOS_CLASS:
                stop = "aeos_argmax"
            elif pa >= 0.35:
                stop = "aeos_tau"
            elif self.hazard >= 0.6:
                stop = "aeos_hazard"
        q = probs.clone()
        q[AEOS_CLASS] = 0
        q = q / q.sum().clamp_min(1e-12)
        cb0 = int(torch.multinomial(q, 1, generator=self.gen))
        if self.win > 0:
            self.hist[:-1] = self.hist[1:]
            self.hist[-1] = cb0
        self.frames += 1
        return stop, cb0


class MayaTTS:
    """from maya_mac import MayaTTS
    tts = MayaTTS.from_pretrained()                      # downloads on first use, MPS when available
    wav = tts.generate("नमस्ते, आप कैसे हैं?", lang="hi", voice="Ahana", normalize=True)   # float32, 24 kHz
    for chunk in tts.stream("Hello there.", lang="en", voice="Ahana"): ...              # float32 chunks
    tts.register_voice("me", ref_audio="me.wav", ref_text="...", lang="en")            # zero-shot clone
    """

    def __init__(self, model_dir, mimi_dir, ecapa_dir=None, device=None, config: EngineConfig = None, log=print):
        from transformers import MimiModel
        from maya_mac.tts.data.packing_v2 import PackerV2
        from maya_mac.tts.model.load_s3 import load_model_for_eval
        from maya_mac.codec import StreamingMimi
        from maya_mac.fast_backbone import LeanBackbone
        from maya_mac.fast_depth import FastDepth

        self.device = device = device or default_device()
        self.cfg = cfg = config or EngineConfig()
        self.log = log
        t0 = time.perf_counter()
        attn = cfg.backbone_attn if cfg.backbone_attn != "auto" else ("eager" if device == "mps" else "sdpa")
        mcfg = {"model": {"backbone": str(Path(model_dir) / "backbone"), "attn": attn}, "data": {"max_tokens": 8192}}
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        bb_dtype = torch.float32 if device == "cpu" else dtypes[cfg.backbone_dtype]
        self.dtype = bb_dtype
        self.model, _ = load_model_for_eval(str(model_dir), device=device, dtype=bb_dtype, cfg=mcfg,
                                            config=str(HERE / "configs/s3_fresh_v1.yaml"), amortize_frac=1.0,
                                            require_anchor=True, verbose=False)
        self.model.depth.float()
        depth_dtype = dtypes[cfg.depth_dtype]
        if device == "cpu":
            depth_dtype = torch.float32
        self.depth = FastDepth(self.model.depth, dtype=depth_dtype, temp=cfg.depth_temp, top_k=cfg.depth_top_k)
        d = self.model.depth   # only spk_proj_* / null_mem are used from here on; FastDepth holds its own copy
        for name in ("groups", "cross", "heads", "code_lo", "code_up", "in_proj", "level_emb", "norm"):
            setattr(d, name, None)
        # the lean backbone keeps its own fused q|k|v and gate|up weights (and references the rest); the HF decoder
        # layers are never called after this, so drop them instead of holding two copies
        self.backbone = LeanBackbone(self.model.backbone, capacity=cfg.capacity)
        self.model.backbone.lm.model.layers = torch.nn.ModuleList()
        if device == "mps":
            torch.mps.empty_cache()
        self.packer = PackerV2(self.model.backbone.tokenizer, self.model.backbone.token_ids, max_tokens=8192)
        self.prior = S.load_prior(str(HERE / "configs/text_frames_prior.json"))
        self.dur = G.DurationPrior()
        # Mimi runs on the CPU in a worker thread: off the GPU's critical path (MPS pays per kernel), and exact
        self.mimi = MimiModel.from_pretrained(str(mimi_dir)).to(cfg.codec_device).eval()
        self.codec = StreamingMimi(self.mimi)
        self._codec_pool = ThreadPoolExecutor(1, thread_name_prefix="maya-mimi")
        self.ecapa_dir = ecapa_dir
        self._ecapa = None
        self.registry = VoiceRegistry()
        voices_json = Path(model_dir) / "voices" / "voices.json"
        if voices_json.is_file():
            load_named_voices(self.registry, str(voices_json))
        self._fb_offsets = torch.arange(N_Q, device=device) * self.model.audio_in.n_codes
        self._fb_scale = float(N_Q) ** 0.5
        if cfg.warmup:      # MPS compiles its kernels on first use: pay that at load, not on the first request
            list(self.stream("Hello, this is a warmup.", "en", max_frames=6, seed=1))
        self.load_s = time.perf_counter() - t0
        self._seed_rng = np.random.default_rng(cfg.seed)
        self.last_info = None

    @classmethod
    def from_pretrained(cls, models_dir=None, device=None, **config):
        """Weights come from the Hugging Face cache (downloaded on first use, ~5.5 GB), or from models_dir /
        $MAYA_MODEL_DIR holding maya2-native/, mimi/ and ecapa/. config: EngineConfig fields.
        device "mlx" (or backend="mlx") selects the MLX engine; with backend "auto" and no device it is used whenever
        MLX is installed on Apple Silicon, and an explicit torch device (mps | cpu | cuda) selects the torch engine."""
        d = resolve_models(models_dir)
        cfg = EngineConfig(**config)
        if cls is MayaTTS:
            if device == "mlx" or cfg.backend == "mlx" or (cfg.backend == "auto" and device is None and mlx_available()):
                from maya_mac.mlx_engine import MayaMLX
                cls = MayaMLX
        return cls(d["maya2-native"], d["mimi"], d["ecapa"], device=device, config=cfg)

    # ------------------------------------------------------------------ voices
    @property
    def voices(self):
        return self.registry.ids()

    @torch.inference_mode()
    def prepare_voice(self, voice: Voice):
        """Pack the voice's reference prefix and compute its depth speaker memory (once per voice)."""
        with voice.lock:
            if voice.mem is not None:
                return voice
            _, use_mem = MODES[voice.mode]
            builder = voice.builder or DEFAULT_BUILDER
            voice.prefix = builder.prefix(self.packer, voice)
            voice.spk_refs = (np.asarray(voice.ecapa, np.float32).reshape(1, 192) if voice.ecapa is not None
                              else np.zeros((1, 192), np.float32))
            refs_t = torch.as_tensor(voice.spk_refs, device=self.device)[None]
            voice.mem = self.model.depth.spk_memory(refs_t, "ecapa",
                                                    null_mask=torch.tensor([not use_mem], device=self.device))
            voice.rmask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
            voice.ts = int(voice.prefix["tokens"].shape[0]) if voice.prefix is not None else 0
            return voice

    def _embed(self, w):
        if self._ecapa is None:
            if self.ecapa_dir is None:
                raise ValueError("ecapa_dir is required for reference-audio voices")
            self._ecapa = Ecapa(self.ecapa_dir, self.device)
        return self._ecapa(w)

    @torch.inference_mode()
    def make_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", meta=None):
        """Build a Voice from a reference clip (audio + its transcript + its language); mode plain needs no clip."""
        mode = norm_mode(mode)
        if mode == "plain":
            return self.prepare_voice(Voice(voice_id, mode, meta=meta or {}))
        if ref_audio is None:
            raise ValueError(f"mode {mode} needs reference audio")
        w = load_audio(ref_audio)
        if len(w) > 30 * SR:
            raise ValueError("reference audio longer than 30 s; 3-10 s works best")
        x = torch.as_tensor(w, device=self.cfg.codec_device)[None, None]
        codes = self.mimi.encode(x, num_quantizers=N_Q).audio_codes[0].cpu().numpy().astype(np.int64)
        ecapa = self._embed(w) if mode in ("spk", "icl_spk") else None
        clips = []
        if mode in ("icl", "icl_spk"):
            if not ref_text or not lang:
                raise ValueError("ref_text and lang (the reference clip's transcript and language) are required")
            clips = [{"text": ref_text, "codes": codes, "lang": lang}]
        return self.prepare_voice(Voice(voice_id, mode, clips=clips, ecapa=ecapa, meta=meta or {}))

    def register_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", meta=None):
        self.registry.add(self.make_voice(voice_id, ref_audio, ref_text, lang, mode, meta))
        return voice_id

    def remove_voice(self, voice_id):
        return self.registry.remove(voice_id) is not None

    def close(self):
        self._codec_pool.shutdown(wait=True)

    # ------------------------------------------------------------------ synthesis
    def _plan(self, text, lang, voice, max_frames):
        cfg, g = self.cfg, self.cfg.guards
        builder = voice.builder or DEFAULT_BUILDER
        ex = builder.example(self.packer, voice, text, lang)
        prompt, _ = S.prompt_from_example(ex, self.model, self.device, self.dtype, N_Q)
        ef, er = S.expected_frames(self.prior, lang, len(text))
        loose = min(cfg.hard_cap, int(2.0 * (ef + er) + 40))
        cap = loose
        if g.enabled:
            cap = min(cap, max(int(ef * 0.5) + 8, self.dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
        if max_frames:
            cap = min(cap, max(1, int(max_frames)))
        minimum = max(4, int(0.5 * ef))
        suspend = G.suspend_windows_v2(text, lang, self.dur) if (g.enabled and g.loop and g.aware_v2) else ()
        return prompt.to(self.dtype), cap, loose, minimum, suspend

    @torch.inference_mode()
    def _frames(self, text, lang, voice, sampling, max_frames, info):
        """Yield the (32,) code tensor of each frame of one text piece; info receives the stop record."""
        cfg, g = self.cfg, self.cfg.guards
        prompt, cap, loose, minimum, suspend = self._plan(text, lang, voice, max_frames)
        seed = sampling.get("seed")
        seed = int(seed) if seed is not None else int(self._seed_rng.integers(1, 2 ** 31 - 1))
        cpu_gen = torch.Generator().manual_seed(seed)
        self.depth.set_speaker(voice.mem, seed=(seed * 69069 + 1) % (2 ** 31 - 1))
        sampler = Cb0Sampler(cfg, minimum, sampling, cpu_gen)
        if prompt.shape[1] + cap + 1 > cfg.capacity:
            raise ValueError(f"prompt of {prompt.shape[1]} tokens + {cap} frames exceeds KV capacity {cfg.capacity}")
        backbone = self.backbone
        backbone.reset()
        h = backbone.forward(prompt)
        cb0_hist = np.full(g.hist, -1, np.int64)
        frames, reason, boost_at, n_boost = 0, "cap", None, 0
        while True:
            logits = self.model.backbone.cb0_logits(h)[0].float().cpu()   # the one host sync per frame
            stop, cb0 = sampler(logits)
            if stop:
                reason = stop
                break
            codes = self.depth(h, cb0)                             # (32,) long, on device
            yield codes
            frames += 1
            cb0_hist[:-1] = cb0_hist[1:]
            cb0_hist[-1] = cb0
            if frames >= cap:
                reason = "max_frames" if max_frames and frames >= max_frames else (
                    "cap_guard" if g.enabled and cap < loose else "cap")
                break
            if (g.enabled and g.loop and frames >= g.start and frames % g.every == 0
                    and not any(a <= frames <= b for a, b in suspend)):
                hit = G.loop_detect_batch(cb0_hist[None], np.array([frames]), g)[0]
                if hit is not None and g.mode == "boost":
                    if boost_at is None:
                        boost_at, n_boost = frames, n_boost + 1
                        sampler.aeos_bias = g.boost
                    if frames - boost_at >= g.backstop:
                        reason = "loop_backstop"
                        break
                elif hit is not None:
                    reason = "loop_" + hit.split(":")[0]
                    break
                elif boost_at is not None:
                    sampler.aeos_bias, boost_at = 0.0, None
            h = backbone.forward(self._feedback(codes)[None])
        info.update({"reason": reason, "frames": frames, "cap": cap, "n_boost": n_boost})

    def _feedback(self, codes):
        """== model.audio_in.frames(codes[None], 32) without its host-synchronizing range check (depth codes are
        always in range)."""
        a = self.model.audio_in
        return a.norm(a.table(codes + self._fb_offsets)[None].sum(1) / self._fb_scale)

    def _voice(self, voice, ref_audio, ref_text, ref_lang, mode):
        if ref_audio is not None:
            if voice not in (None, "plain"):
                raise ValueError("pass either voice=<named/registered voice> or ref_audio=, not both")
            return self.make_voice("oneoff", ref_audio, ref_text, ref_lang, mode)
        v = self.registry.get(voice or "plain")
        return self.prepare_voice(v) if v.mem is None else v

    def _prep(self, text, lang, normalize):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        text = text.strip()
        if normalize:
            from maya_mac.text import normalise_text
            text = normalise_text(text, lang)
        return S.split_text(text, lang, self.prior, self.cfg.chunk_s) if self.cfg.chunk_s > 0 else [text]

    def stream(self, text, lang="en", voice=None, ref_audio=None, ref_text=None, ref_lang=None, mode="icl_spk",
               normalize=False, max_frames=None, seed=None, temperature=None, top_k=None, top_p=None,
               repetition_penalty=None, info=None):
        """Yield float32 PCM chunks (24 kHz mono) as frames are generated."""
        v = self._voice(voice, ref_audio, ref_text, ref_lang or lang, mode)
        sampling = {"seed": seed, "temperature": temperature, "top_k": top_k, "top_p": top_p,
                    "repetition_penalty": repetition_penalty}
        pieces = self._prep(text, lang, normalize)
        t0, t_first, total, stops = time.perf_counter(), None, 0, []
        for k, piece in enumerate(pieces):
            if k:
                yield np.zeros(int(self.cfg.gap_s * SR), np.float32)
            sp = dict(sampling)
            if seed is not None:       # each piece gets its own derived stream (upstream rule)
                sp["seed"] = (int(seed) * 1000003 + k * 7919) % (2 ** 31 - 1)
            rec = {}
            left = (max_frames - total) if max_frames else None
            self._codec_pool.submit(self.codec.reset).result()
            buf, pending, n = [], collections.deque(), self.cfg.stream_frames
            for codes in self._frames(piece, lang, v, sp, left, rec):
                buf.append(codes)
                # a frame's codes are ready once the NEXT frame has been produced (its cb0 read syncs the device),
                # so chunks are copied to the host one frame late and never stall the GPU pipeline
                if len(buf) > n:
                    chunk, buf = torch.stack(buf[:n], -1).cpu(), buf[n:]
                    pending.append(self._codec_pool.submit(self.codec.decode, chunk))
                while pending and pending[0].done():
                    if t_first is None:
                        t_first = time.perf_counter()
                    yield pending.popleft().result()
            if buf:
                pending.append(self._codec_pool.submit(self.codec.decode, torch.stack(buf, -1).cpu()))
            while pending:
                pcm = pending.popleft().result()
                if t_first is None:
                    t_first = time.perf_counter()
                yield pcm
            total += rec["frames"]
            stops.append(rec)
            if max_frames and total >= max_frames:
                break
        wall = time.perf_counter() - t0
        audio_s = total * FRAME / SR
        self.last_info = {"reason": stops[-1]["reason"] if stops else None, "frames": total, "audio_s": audio_s,
                          "wall_s": wall, "rtf": wall / audio_s if audio_s else None,
                          "ttfa_s": (t_first - t0) if t_first else None, "pieces": len(pieces), "stops": stops}
        if info is not None:
            info.update(self.last_info)

    def generate(self, text, lang="en", voice=None, return_info=False, **kw):
        """Synthesise text -> float32 waveform (24 kHz). kw: see stream()."""
        info = {}
        chunks = list(self.stream(text, lang, voice, info=info, **kw))
        wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
        return (wav, info) if return_info else wav
