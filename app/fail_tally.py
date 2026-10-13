#!/usr/bin/env python
"""Where does a student fail? fail_tally.py ckpt first_seed n -> counts of (part, stage, drop|timeout) for failed runs."""
import sys, collections
from multiprocessing import Pool

def run(args):
    ckpt, seed = args
    from multi_env import MultiAssemblyEnv
    from student_multi import MultiStudent
    pol = MultiStudent.load(open(ckpt, "rb").read()); e = MultiAssemblyEnv(); o, _ = e.reset(seed=seed)
    first = {}
    for k in range(e.cfg["max_steps"]):
        o, _, te, tr, i = e.step(pol.act(o)[0])
        if te or tr: break
    return seed, i["stage_i"] >= i["n_stages"], e.part, e.stage, "drop" if i["dropped"] else "timeout", e.order.index(e.part) if e.part else -1

if __name__ == "__main__":
    ckpt, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with Pool(14) as p:
        res = p.map(run, [(ckpt, s) for s in range(s0, s0 + n)])
    fails = collections.Counter((r[2], r[3], r[4], f"part#{r[5]+1}") for r in res if not r[1])
    print(f"success {sum(r[1] for r in res)}/{n}")
    for k, v in fails.most_common(): print(v, k)
