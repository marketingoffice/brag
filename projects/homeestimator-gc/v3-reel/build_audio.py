"""Audio for the 30s 'From Conversation to Proposal' reel.

lines.json rows: [id, start_s, kokoro_voice, speed, text]
Usage: python3 build_audio.py [--tts]
"""
import json, subprocess, sys, os
import numpy as np, soundfile as sf, librosa

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100
DUR = 30.0
MUSIC_OFFSET = 10.0  # original drop 19.92s -> 9.92s here
MUSIC = os.path.join(HERE, "..", "vo-src", "music-original.mp3")
SFX_DIR = os.path.join(HERE, "..", "..", "..", "skills", "brag", "assets", "sfx")
VO_DIR = os.path.join(HERE, "vo-src")
OUT = os.path.join(HERE, "composition", "assets", "audio")

# tactile scope confirmations + UI accents: (time, file, gain)
SFX = [
    (0.10, "interface/bong_001.ogg", 0.5),              # phone wakes / listening
    *[(t, "interface/click_003.ogg", 0.8) for t in (2.05, 2.30, 2.55, 2.80)],  # cabinets, counters, floor, lighting
    (3.75, "interface/switch_002.ogg", 0.5),            # finish tier -> better
    (6.60, "interface/switch_007.ogg", 0.5),            # occupied-home condition
    (6.85, "ui/click2.ogg", 0.5),
    (9.92, "impact/impactSoft_heavy_001.ogg", 0.7),     # drop
    *[(t, "impact/impactSoft_medium_002.ogg", 0.55) for t in (11.59, 12.82, 14.05, 15.30)],  # pipeline stations
    (15.45, "casino/card-slide-1.ogg", 0.4),            # proposal resolves
    (18.40, "impact/impactSoft_medium_004.ogg", 0.6),   # ~10 MIN
    (19.64, "ui/click2.ogg", 0.5),
    (20.87, "ui/click2.ogg", 0.5),
    (22.71, "impact/impactSoft_heavy_004.ogg", 0.6),    # less re-entry
    (26.20, "impact/impactBell_heavy_000.ogg", 0.5),    # CTA
]


def load(path, sr=SR):
    x, s = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(1)
    return librosa.resample(x, orig_sr=s, target_sr=sr) if s != sr else x


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(VO_DIR, exist_ok=True)
    lines = json.load(open(os.path.join(HERE, "lines.json")))
    n = int(DUR * SR)
    if "--tts" in sys.argv:
        for lid, _, voice, speed, text in lines:
            p = os.path.join(VO_DIR, lid + ".wav")
            subprocess.run(["npx", "hyperframes", "tts", text, "-v", voice, "-s", str(speed), "-o", p], check=True, capture_output=True)
            d, s = sf.read(p)
            idx = np.where(np.abs(d) > 0.01)[0]
            d = d[max(idx[0] - int(0.03 * s), 0): min(idx[-1] + int(0.12 * s), len(d))]
            f = int(0.01 * s)
            d[:f] *= np.linspace(0, 1, f)
            d[-f:] *= np.linspace(1, 0, f)
            sf.write(p, d, s)

    vo = np.zeros(n)
    act = np.zeros(n)
    timing = []
    for lid, t, voice, speed, text in lines:
        x = load(os.path.join(VO_DIR, lid + ".wav"))
        if lid.startswith("c"):
            # contractor on a jobsite phone: gentle band-limit so it reads as a different mic
            x = librosa.effects.preemphasis(x, coef=0.5) * 0.9
        i = int(t * SR)
        seg = x[: max(0, n - i)]
        vo[i: i + len(seg)] += seg
        act[max(0, int((t - 0.12) * SR)): i + len(seg) + int(0.2 * SR)] = 1
        timing.append({"id": lid, "start": t, "end": round(t + len(x) / SR, 2), "text": text})
    vo *= 0.85 / max(1e-6, np.abs(vo).max())

    m, _ = librosa.load(MUSIC, sr=SR, mono=False, offset=MUSIC_OFFSET, duration=DUR)
    m = (m if m.ndim == 2 else np.stack([m, m])).T
    m = np.pad(m, ((0, max(0, n - len(m))), (0, 0)))[:n]
    k = int(0.2 * SR)
    duck = 1 - 0.62 * np.convolve(act, np.ones(k) / k, "same")  # voice-first mix
    fade = np.ones(n)
    fl = int(1.4 * SR)
    fade[-fl:] = np.linspace(1, 0, fl) ** 1.5
    m = m * (duck * fade)[:, None] * 0.8

    fx = np.zeros(n)
    for t, f, g in SFX:
        x = load(os.path.join(SFX_DIR, f))
        x = x / max(1e-6, np.abs(x).max()) * g
        i = int(t * SR)
        seg = x[: max(0, n - i)]
        fx[i: i + len(seg)] += seg

    mix = m + (vo + fx * 0.6)[:, None]
    mix *= min(1.0, 0.95 / np.abs(mix).max())
    wav = os.path.join(OUT, "mix.wav")
    sf.write(wav, mix, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-b:a", "256k", os.path.join(OUT, "mix.mp3")], check=True)
    os.remove(wav)
    json.dump(timing, open(os.path.join(HERE, "vo-timing.json"), "w"), indent=1)
    for tm in timing:
        print(tm["id"], tm["start"], "->", tm["end"], tm["text"])


if __name__ == "__main__":
    main()
