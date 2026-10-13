#!/usr/bin/env python
"""Evaluate a multi-part student on seeds; optionally render successes. Usage: student_eval.py ckpt 777000-777027 [--render-dir d]"""
import argparse, os
import numpy as np
from multi_env import MultiAssemblyEnv
from student_multi import MultiStudent
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("seeds"); ap.add_argument("--render-dir", default=None)
ap.add_argument("--parts", default="bracket,cap,lbracket"); a = ap.parse_args()
lo, hi = a.seeds.split("-"); pol = MultiStudent.load(open(a.ckpt, "rb").read())
ok = 0; n = 0
for seed in range(int(lo), int(hi) + 1):
    e = MultiAssemblyEnv({"parts": tuple(a.parts.split(",")), "render": bool(a.render_dir), "camera": "demo", "width": 640, "height": 480})
    o, _ = e.reset(seed=seed); frames = []
    for k in range(e.cfg["max_steps"]):
        o, _, te, tr, i = e.step(pol.act(o)[0])
        if a.render_dir and k % 2 == 0: frames.append(e.render())
        if te or tr: break
    done = i["stage_i"] >= i["n_stages"]; ok += done; n += 1
    print(f"seed {seed} {e.order}: stages {i['stage_i']}/{i['n_stages']} attached {len(e.joined)} steps {k+1} {'SUCCESS' if done else ''}", flush=True)
    if a.render_dir and done:
        import imageio.v2 as iio; os.makedirs(a.render_dir, exist_ok=True)
        iio.mimwrite(os.path.join(a.render_dir, f"student_success_seed{seed}.mp4"), frames, fps=10, codec="libx264", quality=8)
print(f"full assemblies {ok}/{n}")
