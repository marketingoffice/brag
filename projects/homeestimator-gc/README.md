# HomeEstimator.ai — General Contractor promo (motion graphics)

Built with /brag + Hyperframes. Rewrite of *HomeEstimatorAI_Cinematic_Script_v2* for a GC audience,
locked to the supplied 60.7s music track (~96 BPM).

| Path | What |
|---|---|
| `HomeEstimatorAI_GC_MotionScript_v3.docx` | Rewritten script: beats, VO with timecodes, visuals, change log |
| `composition/index.html` | Vertical 1080x1920 composition (IG / FB Reels, X, YouTube Shorts) |
| `composition/assets/` | Logo (icon / wordmark), fonts, GSAP, ducked music, VO mix |
| `vo-lines.json` | Every VO line: id, start time (s), text, duration |
| `vo-src/` | Per-line Kokoro WAVs (voice `am_michael`, 0.95x) + original music |
| `beats.json` | Beat grid detected from the music |
| `share-copy.txt` | Captions per platform |

## Structure (music-locked)

| Time | Beat | Music |
|---|---|---|
| 0.0–9.7 | Problem: flat-rate template → 30–60% ERROR, margin drains | sparse intro |
| 9.7–19.9 | Pivot: bid vs build, change orders / callbacks / lost jobs, "Until now." | build → natural silence 19.1–19.8 |
| 19.9–31.2 | Logo reveal (beat-locked 19.92 drop), 15-minute estimates, 23+ years field data | full section |
| 31.2–39.7 | Engine: <5% variance gauge, 50+ room cards, 3 finish tiers, live inflation | full section |
| 39.7–47.6 | GC payoff (beat-locked 39.73): Bid faster / tighter / win more; No architect / plans / waiting | second section |
| 47.6–50.5 | Stop guessing. Start winning. | energy peak |
| 50.5–60.7 | CTA: HomeEstimator.ai + 3-day money-back guarantee, 5s static end hold | outro |

## Rebuild

```bash
cd composition
npx hyperframes check
npx hyperframes render -o ../renders/HomeEstimator_GC_Vertical_9x16.mp4 -f 30 -q delivery
```

Needs `ffmpeg`/`ffprobe` on PATH. VO regenerate: `npx hyperframes tts "<line>" -v am_michael -s 0.95` (needs `pip install kokoro-onnx soundfile`).
