"""Deterministic procedural (non-AI) gradient background renderer."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

PALETTES = [
    [(12, 20, 45), (62, 72, 134), (203, 128, 181), (255, 225, 195)],
    [(14, 36, 49), (29, 128, 136), (151, 213, 195), (250, 246, 220)],
    [(32, 19, 48), (104, 46, 126), (231, 84, 142), (255, 198, 114)],
    [(241, 221, 233), (195, 207, 245), (189, 232, 222), (255, 243, 204)],
    [(12, 12, 18), (57, 51, 68), (136, 109, 95), (235, 208, 161)],
    [(14, 44, 85), (38, 112, 165), (91, 202, 198), (235, 247, 217)],
    [(35, 17, 63), (91, 35, 116), (188, 71, 136), (252, 177, 151)],
    [(250, 244, 224), (235, 201, 180), (174, 189, 190), (75, 125, 140)],
    [(22, 25, 32), (62, 69, 78), (139, 132, 123), (235, 222, 198)],
    [(16, 60, 55), (43, 120, 98), (134, 195, 144), (249, 238, 175)],
]


def render(i, dest, w=6000, h=4000):
    if not 0 <= i < 10000:
        raise ValueError("index outside campaign")
    if w < 1 or h < 1:
        raise ValueError("invalid dimensions")
    rng = np.random.default_rng(int.from_bytes(
        hashlib.sha256(("gradient-v4-" + str(i)).encode()).digest()[:8], "big"))
    pal = np.asarray(PALETTES[int(rng.integers(len(PALETTES)))], dtype=np.float32)
    if rng.random() < 0.5:
        pal = pal[::-1].copy()
    angle = float(rng.uniform(-math.pi, math.pi))
    cx, cy = rng.uniform(.12, .88, size=2)
    mode = i % 7
    waves = rng.uniform(2, 11, size=3)
    phase = rng.uniform(-math.pi, math.pi, size=3)
    # Independent localized highlights create distinct spatial geometry, not
    # just different colors applied to the same gradient.
    spots = [
        (float(rng.uniform(.05, .95)), float(rng.uniform(.05, .95)),
         float(rng.uniform(.12, .42)), float(rng.uniform(.12, .48)),
         float(rng.uniform(-.35, .35)))
        for _ in range(int(rng.integers(2, 6)))
    ]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    image = Image.new("RGB", (w, h))
    for top in range(0, h, 96):
        n = min(96, h - top)
        y = ((np.arange(top, top + n, dtype=np.float32) + .5) / h)[:, None]
        dx, dy = x - cx, y - cy
        if mode == 0:
            t = .5 + .65 * (dx * math.cos(angle) + dy * math.sin(angle))
        elif mode == 1:
            t = np.sqrt((dx * 1.1) ** 2 + (dy * .9) ** 2) * 1.3
        elif mode == 2:
            t = .5 + .36 * np.sin(waves[0] * x + waves[1] * y + phase[0])
        elif mode == 3:
            t = .5 + .5 * np.arctan2(dy, dx) / math.pi
        elif mode == 4:
            t = .5 + .55 * (dx * math.cos(angle) + dy * math.sin(angle))
            t = t + .13 * np.sin(waves[0] * x + waves[1] * y + phase[1])
        elif mode == 5:
            t = .25 + .7 * np.sqrt((dx * 1.5) ** 2 + (dy * .6) ** 2)
        else:
            t = .5 + .23 * np.sin(waves[0] * x + phase[0]) * np.cos(waves[1] * y + phase[1])
        t = np.broadcast_to(t, (n, w)).astype(np.float32).copy()
        for sx, sy, rx, ry, strength in spots:
            t += strength * np.exp(-.5 * (((x - sx) / rx) ** 2 + ((y - sy) / ry) ** 2))
        t += .035 * np.sin(waves[2] * x - waves[1] * y + phase[2])
        z = np.clip(t, 0, 1) * 3
        k = np.minimum(z.astype(np.int8), 2)
        f = (z - k)[..., None]
        f = f * f * (3 - 2 * f)
        rgb = pal[k] * (1 - f) + pal[k + 1] * f
        dither = rng.uniform(-.48, .48, size=(n, w, 1)).astype(np.float32)
        image.paste(Image.fromarray(
            np.uint8(np.clip(rgb + dither + .5, 0, 255)), "RGB"), (0, top))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(dest, "JPEG", quality=97, subsampling=0, icc_profile=icc, optimize=True)
    return {"index": i, "filename": dest.name,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--width", type=int, default=6000)
    p.add_argument("--height", type=int, default=4000)
    p.add_argument("--out", default="output")
    a = p.parse_args()
    if a.start < 0 or not 1 <= a.count <= 50 or a.start + a.count > 10000:
        p.error("invalid campaign range")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = [
        render(i, out / f"gradient_{i + 1:05d}.jpg", a.width, a.height)
        for i in range(a.start, a.start + a.count)
    ]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
