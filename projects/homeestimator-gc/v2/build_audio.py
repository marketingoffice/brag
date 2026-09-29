"""Build the v2 audio mix: music (offset to land the drop at 9.92s), VO lines, SFX.

Usage: python3 build_audio.py [--tts]   (--tts regenerates VO lines with Kokoro)
Writes composition/assets/audio/mix.mp3 and vo-timing.json.
"""
import json, subprocess, sys, os
import numpy as np, soundfile as sf, librosa

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100
DUR = 45.5
MUSIC_OFFSET = 10.0          # original drop at 19.92s -> lands at 9.92s
MUSIC = os.path.join(HERE, "..", "vo-src", "music-original.mp3")
SFX_DIR = os.path.join(HERE, "..", "..", "..", "skills", "brag", "assets", "sfx")
VO_DIR = os.path.join(HERE, "vo-src")
OUT = os.path.join(HERE, "composition", "assets", "audio")
VOICE, SPEED = "am_michael", 1.05

# sound-event hooks: (time, file, gain)
SFX = [
    (0.05, "interface/bong_001.ogg", 0.55),            # notification
    (2.62, "impact/impactSoft_heavy_003.ogg", 0.7),    # impact: FIRST NUMBER
    (3.30, "impact/impactWood_medium_001.ogg", 0.45),  # number fractures
    *[(4.30 + i * 0.62, "interface/glitch_002.ogg", 0.25) for i in range(5)],  # missing scope ticks
    (9.92, "impact/impactSoft_heavy_001.ogg", 0.75),   # drop: gold cut
    *[(10.10 + i * 0.62, "interface/click_003.ogg", 0.7) for i in range(5)],   # five inputs
    (12.95, "impact/impactSoft_medium_001.ogg", 0.6),  # converge
    (15.30, "impact/impactSoft_medium_002.ogg", 0.55), # scope-add: second floor
    (16.65, "interface/switch_007.ogg", 0.45),         # scope-add: cape roof
    (17.65, "ui/click2.ogg", 0.55),                    # occupied chip
    (18.85, "interface/switch_002.ogg", 0.45),         # finish tier
    (22.35, "ui/click2.ogg", 0.6),                     # select wall insulation
    (23.25, "interface/switch_007.ogg", 0.45),         # drywall -> paint
    (24.85, "impact/impactSoft_medium_004.ogg", 0.6),  # gut-to-studs
    *[(25.15 + i * 0.31, "interface/switch_005.ogg", 0.35) for i in range(6)],  # dependency-lock
    (29.73, "impact/impactSoft_heavy_004.ogg", 0.7),   # hit: price logic
    (30.40, "interface/click_002.ogg", 0.6),
    (31.02, "interface/click_002.ogg", 0.6),
    (32.70, "interface/error_005.ogg", 0.25),          # drag attempt rejected
    (33.15, "interface/switch_002.ogg", 0.6),          # margin-lock
    (35.45, "casino/card-slide-1.ogg", 0.45),          # proposal-resolve
    (36.25, "interface/bong_001.ogg", 0.45),
    (39.05, "impact/impactBell_heavy_000.ogg", 0.55),  # cta-hit / logo
    (40.45, "ui/click2.ogg", 0.5),
]


def load(path, sr=SR):
    x, s = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(1)
    if s != sr:
        x = librosa.resample(x, orig_sr=s, target_sr=sr)
    return x


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(VO_DIR, exist_ok=True)
    lines = json.load(open(os.path.join(HERE, "vo-lines.json")))
    n = int(DUR * SR)

    if "--tts" in sys.argv:
        for lid, _, text in lines:
            p = os.path.join(VO_DIR, lid + ".wav")
            subprocess.run(["npx", "hyperframes", "tts", text, "-v", VOICE, "-s", str(SPEED), "-o", p],
                           check=True, capture_output=True)
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
    for lid, t, text in lines:
        x = load(os.path.join(VO_DIR, lid + ".wav"))
        i = int(t * SR)
        seg = x[: max(0, n - i)]
        vo[i: i + len(seg)] += seg
        act[max(0, int((t - 0.12) * SR)): i + len(seg) + int(0.2 * SR)] = 1
        timing.append({"id": lid, "start": t, "end": round(t + len(x) / SR, 2), "text": text})
    vo *= 0.85 / max(1e-6, np.abs(vo).max())

    m, _ = librosa.load(MUSIC, sr=SR, mono=False, offset=MUSIC_OFFSET, duration=DUR)
    m = m if m.ndim == 2 else np.stack([m, m])
    m = m.T
    if len(m) < n:
        m = np.pad(m, ((0, n - len(m)), (0, 0)))
    m = m[:n]
    k = int(0.2 * SR)
    duck = 1 - 0.55 * np.convolve(act, np.ones(k) / k, "same")
    fade = np.ones(n)
    fl = int(1.6 * SR)
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
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix *= 0.95 / peak
    wav = os.path.join(OUT, "mix.wav")
    sf.write(wav, mix, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-b:a", "256k",
                    os.path.join(OUT, "mix.mp3")], check=True)
    os.remove(wav)
    json.dump(timing, open(os.path.join(HERE, "vo-timing.json"), "w"), indent=1)
    for tm in timing:
        print(tm["id"], tm["start"], "->", tm["end"])


if __name__ == "__main__":
    main()
