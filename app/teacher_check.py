#!/usr/bin/env python
"""Run the multi-part teacher on a list of seeds: app/venv/bin/python app/teacher_check.py 1-24 [--parts a,b,c] [--render out.mp4]"""
import argparse, sys, time
import numpy as np
from multi_env import MultiAssemblyEnv
import multi_teacher as T

ap = argparse.ArgumentParser(); ap.add_argument("seeds"); ap.add_argument("--parts", default="bracket,cap,lbracket")
ap.add_argument("--render", default=None); ap.add_argument("--max-steps", type=int, default=0)
a = ap.parse_args()
lo, hi = (a.seeds.split("-") + [None])[:2]; seeds = range(int(lo), int(hi) + 1) if hi else [int(s) for s in a.seeds.split(",")]
parts = tuple(a.parts.split(",")); ms = a.max_steps or (2600 if len(parts) <= 3 else 4400)
e = MultiAssemblyEnv({"parts": parts, "max_steps": ms, "render": bool(a.render), "camera": "demo", "width": 640, "height": 480})
ok = 0
for seed in seeds:
    e.reset(seed=seed); trace = []; last = None; frames = []; t0 = time.time()
    for k in range(ms):
        _, _, te, tr, i = e.step(T.act(e))
        if (e.part, e.stage) != last:
            last = (e.part, e.stage); trace.append(f"{k}:{last[0]}:{last[1]}")
        if a.render and k % 2 == 0:
            frames.append(e.render())
        if te or tr:
            break
    done = i["stage_i"] >= i["n_stages"]; ok += done
    print(f"seed {seed} {e.order}: stages {i['stage_i']}/{i['n_stages']} steps {k + 1} dropped {i['dropped']} ({time.time() - t0:.1f}s)"
          + ("" if done else f"  last: {' '.join(trace[-4:])}"), flush=True)
    if a.render:
        import imageio.v2 as iio; iio.mimwrite(a.render, frames, fps=10, codec="libx264", quality=8)
print(f"teacher complete: {ok}/{len(seeds)}")
