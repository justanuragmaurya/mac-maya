"""Queued engine for the server: the interface maya_mac.server expects (upstream maya3.engine.Engine's), on top of
MayaTTS.

One worker thread owns the model and runs requests one at a time in arrival order (MayaTTS generates a single stream at
about real time on Apple Silicon; there is no batching).  Voice registration also runs on that thread, between
requests, so the GPU is only ever driven from one thread.

    engine = ServingEngine(MayaTTS.from_pretrained())
    req = engine.submit("Hello there.", "en", "Ahana", on_event=lambda kind, payload: ...)
    req.cancel()
Events, called from the worker thread (keep them cheap and non-blocking):
    ('audio', np.int16 [n])   PCM 24 kHz mono, a few frames at a time
    ('end', dict)             reason, frames, audio_s, ttfa_s, wall_s, pieces, stops
    ('error', dict)           code, error
"""
from __future__ import annotations

import collections
import itertools
import threading
import time

import numpy as np

SAMPLING_KEYS = {"seed", "temperature", "top_k", "top_p", "repetition_penalty"}


def _check_sampling(sp):
    """Same ranges as upstream maya3.engine._check_sampling."""
    if not sp:
        return {}
    bad = set(sp) - SAMPLING_KEYS
    if bad:
        raise ValueError(f"unknown sampling keys {sorted(bad)}; allowed {sorted(SAMPLING_KEYS)}")
    out = {k: v for k, v in sp.items() if v is not None}
    if "temperature" in out and not 0.05 <= float(out["temperature"]) <= 2.0:
        raise ValueError("temperature must be in [0.05, 2.0]")
    if "top_k" in out and not 1 <= int(out["top_k"]) <= 2049:
        raise ValueError("top_k must be in [1, 2049]")
    if "top_p" in out and not 0.0 < float(out["top_p"]) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if "repetition_penalty" in out and not 0.0 <= float(out["repetition_penalty"]) <= 5.0:
        raise ValueError("repetition_penalty must be in [0, 5]")
    if "seed" in out:
        out["seed"] = int(out["seed"])
    return out


class Request:
    _ids = itertools.count()

    def __init__(self, text, lang, voice, on_event, max_frames=None, rid=None, sampling=None):
        self.id = rid or f"r{next(self._ids)}"
        self.text, self.lang, self.voice = text, lang, voice
        self.on_event = on_event
        self.max_frames = max_frames
        self.sampling = sampling or {}
        self.cancelled = False
        self.t_submit = time.perf_counter()

    def cancel(self):
        self.cancelled = True


class ServingEngine:
    def __init__(self, tts, log=None):
        self.tts = tts
        self.registry = tts.registry
        self.failed = None
        self._q = collections.deque()          # Request | (fn, box, threading.Event)
        self._cv = threading.Condition()
        self._stop = False
        self._busy = False
        self.stats = collections.Counter()
        self._thread = threading.Thread(target=self._loop, name="maya-mac-engine", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ public (thread-safe)
    def submit(self, text, lang, voice="plain", on_event=None, max_frames=None, rid=None, split=True, sampling=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        sampling = _check_sampling(sampling)
        vid = voice if isinstance(voice, str) else voice.id
        self.registry.get(vid)                 # KeyError for an unknown voice, before anything is queued
        req = Request(text.strip(), lang, vid, on_event or (lambda k, p: None), max_frames, rid, sampling)
        with self._cv:
            if self.failed:
                raise RuntimeError(f"engine failed: {self.failed}")
            if self._stop:
                raise RuntimeError("engine stopped")
            self._q.append(req)
            self._cv.notify()
        return req

    def call(self, fn, timeout=300):
        """Run fn() on the worker thread (between requests) and return its result."""
        if threading.current_thread() is self._thread:
            return fn()
        box, ev = {}, threading.Event()
        with self._cv:
            self._q.append((fn, box, ev))
            self._cv.notify()
        if not ev.wait(timeout):
            raise TimeoutError("engine call timed out")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def health(self):
        with self._cv:
            pending = sum(isinstance(x, Request) for x in self._q)
        out = {"ready": self.failed is None and not self._stop, "failed": self.failed, "device": self.tts.device,
               "active": int(self._busy), "pending": pending, "max_seats": 1, **self.stats}
        try:
            import torch
            if self.tts.device == "mps":
                out["mem_alloc_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
        except Exception:
            pass
        return out

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=30)

    # ------------------------------------------------------------------ worker
    def _loop(self):
        try:
            while True:
                with self._cv:
                    while not self._q and not self._stop:
                        self._cv.wait()
                    if self._stop:
                        break
                    item = self._q.popleft()
                    self._busy = isinstance(item, Request)
                if isinstance(item, Request):
                    self._run(item)
                else:
                    fn, box, ev = item
                    try:
                        box["result"] = fn()
                    except Exception as exc:
                        box["error"] = exc
                    ev.set()
                self._busy = False
        except BaseException as exc:  # pragma: no cover - a bug in the loop itself
            import traceback
            traceback.print_exc()
            self.failed = f"{type(exc).__name__}: {exc}"
        finally:
            with self._cv:
                left = list(self._q)
                self._q.clear()
            for item in left:
                if isinstance(item, Request):
                    self._emit(item, "error", {"code": "unavailable", "error": self.failed or "engine stopped"})
                else:
                    item[1]["error"] = RuntimeError("engine stopped")
                    item[2].set()

    def _emit(self, req, kind, payload):
        try:
            req.on_event(kind, payload)
        except Exception:                      # a broken consumer must not kill the engine
            req.cancelled = True

    def _run(self, req):
        if req.cancelled:
            self._emit(req, "error", {"code": "cancelled", "error": "cancelled"})
            return
        self.stats["admitted"] += 1
        info, t_first = {}, None
        gen = self.tts.stream(req.text, lang=req.lang, voice=req.voice, max_frames=req.max_frames, info=info,
                              **req.sampling)
        try:
            for chunk in gen:
                if req.cancelled:
                    break
                if t_first is None:
                    t_first = time.perf_counter()
                self._emit(req, "audio", (np.clip(chunk, -1, 1) * 32767).round().astype(np.int16))
        except (ValueError, KeyError) as exc:
            self._emit(req, "error", {"code": "invalid_request", "error": f"{type(exc).__name__}: {exc}"})
            return
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.stats["errors"] += 1
            self._emit(req, "error", {"code": "internal", "error": f"{type(exc).__name__}: {exc}"})
            return
        finally:
            gen.close()
        if req.cancelled:
            self._emit(req, "error", {"code": "cancelled", "error": "cancelled"})
            return
        self.stats["completed"] += 1
        self._emit(req, "end", {"reason": info.get("reason"), "frames": info.get("frames", 0),
                                "audio_s": info.get("audio_s", 0.0),
                                "ttfa_s": (t_first - req.t_submit) if t_first else None,
                                "wall_s": time.perf_counter() - req.t_submit, "pieces": info.get("pieces"),
                                "stops": info.get("stops", [])})


class ServingTTS:
    """What maya_mac.server needs from a model object: .engine, register_voice, remove_voice, close."""

    def __init__(self, tts):
        self.tts = tts
        self.engine = ServingEngine(tts)

    def register_voice(self, voice_id, ref_audio=None, ref_text=None, lang=None, mode="icl_spk", meta=None):
        return self.engine.call(lambda: self.tts.register_voice(voice_id, ref_audio, ref_text, lang, mode, meta))

    def remove_voice(self, voice_id):
        return self.tts.remove_voice(voice_id)

    def close(self):
        self.engine.close()
        self.tts.close()
