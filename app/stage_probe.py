#!/usr/bin/env python
"""Per-stage competence of a student: the teacher drives to stage s, the student must complete that single stage within
a step budget. Usage: app/venv/bin/python app/stage_probe.py student.pt [seeds]"""
import sys, collections
import numpy as np
from multi_env import MultiAssemblyEnv
import multi_teacher as T
from student_multi import MultiStudent
pol = MultiStudent.load(open(sys.argv[1], "rb").read()); seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
res = collections.defaultdict(list)
for seed in range(seeds):
    e = MultiAssemblyEnv(); o, _ = e.reset(seed=seed); n = len(e.plan)
    for s in range(n - 1):
        e.reset(seed=seed)
        while e.stage_i < s:
            o, _, te, _, _ = e.step(T.act(e))
            if te: break
        if e.stage_i != s: continue
        name = e.stage; ok = False
        for k in range(400):
            o, _, te, tr, i = e.step(pol.act(o)[0])
            if e.stage_i > s: ok = True; break
            if te or tr: break
        res[name].append(ok)
for st, v in sorted(res.items(), key=lambda x: np.mean(x[1])):
    print(f"{st:12s} {np.mean(v):5.0%}  (n={len(v)})")
