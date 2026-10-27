"""maya "text" -l hi -v Ahana -o speech.wav [--normalize] [--seed N] [--ref-audio a.wav --ref-text "..."]"""
import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="maya", description="Maya TTS on Apple Silicon / CPU")
    ap.add_argument("text", nargs="?", help="text to speak")
    ap.add_argument("-l", "--lang", default="en", help="as bn en gu hi kn ml mr or pa ta te")
    ap.add_argument("-v", "--voice", default=None, help="named voice (see --list-voices); default: plain")
    ap.add_argument("-o", "--out", default="speech.wav")
    ap.add_argument("--normalize", action="store_true", help="expand numbers, currency, dates, URLs ...")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ref-audio", help="reference clip for zero-shot cloning")
    ap.add_argument("--ref-text", help="transcript of the reference clip")
    ap.add_argument("--ref-lang", help="language of the reference clip (default: --lang)")
    ap.add_argument("--models", default=None, help="dir holding maya2-native/, mimi/, ecapa/ (default: Hugging Face cache)")
    ap.add_argument("--device", default=None,
                    help="mlx | mps | cpu | cuda (default: mlx when installed on Apple Silicon, else best available)")
    ap.add_argument("--bits", type=int, choices=(8,), default=None,
                    help="mlx: 8-bit weights for the backbone and depth decoder (~1.4x faster, ~3%% logit drift)")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args(argv)

    import soundfile as sf
    from maya_mac.engine import SR, MayaTTS

    tts = MayaTTS.from_pretrained(a.models, device=a.device, **({"backbone_bits": a.bits, "depth_bits": a.bits}
                                                               if a.bits else {}))
    print(f"loaded on {tts.device} in {tts.load_s:.1f}s")
    if a.list_voices:
        print("\n".join(tts.voices))
        return
    if not a.text:
        ap.error("text is required")
    wav, info = tts.generate(a.text, lang=a.lang, voice=a.voice, normalize=a.normalize, seed=a.seed,
                             ref_audio=a.ref_audio, ref_text=a.ref_text, ref_lang=a.ref_lang, return_info=True)
    sf.write(a.out, wav, SR)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, ensure_ascii=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
