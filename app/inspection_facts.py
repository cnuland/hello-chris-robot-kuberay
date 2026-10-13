"""Inspection-station facts (v006): the 16x16 height scan of a passing part turned into plain-language measurements
for DiffusionGemma - the same "semantic facts" recipe as the cell supervisor (v004/v005). DiffusionGemma names the part
from these facts; the student's part-ID head is then trained on (raw scan -> DiffusionGemma's answer).
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

N_HI = 48
CELL = 0.12 / N_HI                                # station scanner pitch (m): 2.5 mm
PART_SPECS = {                                    # the work instruction DiffusionGemma reads
    "bracket": "rectangular bar, about 6 cm x 3 cm, solid (convex), with one fastener hole in the middle",
    "cap": "octagonal cap, about 3.5 cm across in every direction (nearly round), convex",
    "lbracket": "L-shaped bracket, about 6 cm x 5 cm overall, NOT convex (one corner of its bounding box is empty)",
    "plate": "square plate, about 4.5 cm x 4.5 cm, convex",
    "rail": "long narrow bar, about 9 cm x 2 cm, convex",
}


def _hull_area(pts: np.ndarray) -> float:
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    h = np.array(lo[:-1] + hi[:-1], float)
    return 0.5 * abs(np.dot(h[:, 0], np.roll(h[:, 1], 1)) - np.dot(h[:, 1], np.roll(h[:, 0], 1)))


def measure(scan: np.ndarray) -> Dict[str, float]:
    """Footprint measurements from the 48x48 station scan: minimum-area bounding rectangle (rotating calipers over
    1-degree steps), fill = area / rectangle, solidity = area / convex hull."""
    n = int(round(np.sqrt(scan.size))); cell = 0.12 / n * 100                  # cm per cell
    m = scan.reshape(n, n) > 0.3
    ij = np.argwhere(m).astype(float)
    if len(ij) < 3:
        return {"cells": int(len(ij))}
    pts = np.concatenate([ij + d for d in ([0, 0], [0, 1], [1, 0], [1, 1])]) * cell
    best = None
    for a in np.radians(np.arange(0, 90, 1.0)):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); q = pts @ R
        e = q.max(0) - q.min(0)
        if best is None or e[0] * e[1] < best[0] * best[1]:
            best = e
    L, W = float(max(best)), float(min(best)); area = len(ij) * cell ** 2
    hull = _hull_area(pts / cell) * cell ** 2
    return {"cells": int(len(ij)), "length_cm": L, "width_cm": W, "area_cm2": area,
            "fill": area / (L * W), "solidity": area / hull if hull else 1.0}


def facts(scan: np.ndarray) -> List[str]:
    """Rounded, plain-language measurements (rounding makes repeated parts produce the same facts -> cacheable)."""
    q = measure(scan)
    if q["cells"] < 3:
        return ["The scanner saw nothing on the belt."]
    r = lambda v: round(v * 2) / 2
    f = [f"The scanned outline is about {r(q['length_cm']):.1f} cm long and {r(q['width_cm']):.1f} cm wide.",
         f"Its footprint area is about {round(q['area_cm2']):.0f} square cm, filling {round(q['fill'] * 10) * 10:.0f}% of its bounding box."]
    f.append("The outline is convex (no notches)." if q["solidity"] > 0.80 else
             "The outline is NOT convex: part of its bounding box is empty, like a notch or a missing corner.")
    if q["width_cm"] / q["length_cm"] > 0.85:
        f.append("It is about as wide as it is long.")
    return f
