#!/usr/bin/env python
"""Parallel student-only evaluation: eval_parallel.py ckpt [first_seed] [n] [procs] -> full-assembly rate on unseen seeds."""
import sys
from multiprocessing import Pool
import numpy as np

def run(args):
    ckpt, seed = args
    from multi_env import MultiAssemblyEnv
    from student_multi import MultiStudent
    pol = MultiStudent.load(open(ckpt, "rb").read()); e = MultiAssemblyEnv(); o, _ = e.reset(seed=seed)
    for k in range(e.cfg["max_steps"]):
        o, _, te, tr, i = e.step(pol.act(o)[0])
        if te or tr: break
    return seed, int(i["stage_i"]), int(i["n_stages"]), len(e.joined), k + 1

if __name__ == "__main__":
    ckpt = sys.argv[1]; s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 777000; n = int(sys.argv[3]) if len(sys.argv) > 3 else 28
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 14
    with Pool(procs) as p:
        res = p.map(run, [(ckpt, s) for s in range(s0, s0 + n)])
    ok = [r for r in res if r[1] >= r[2]]
    print(f"{ckpt}: full assemblies {len(ok)}/{n} ({len(ok)/n:.0%}), stages {np.mean([r[1]/r[2] for r in res]):.0%}, "
          f"parts {np.mean([r[3] for r in res]):.2f}/3, success seeds {[r[0] for r in ok]}")
