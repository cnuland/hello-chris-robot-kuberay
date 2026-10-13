#!/usr/bin/env python
"""DAgger on KubeRay for the multi-part cell (v006): distil THREE teachers into one lightweight student.

  * DiffusionGemma vision  -> part-ID labels  (offline set: vision_labeler.py; RGB inspection frame -> part name)
  * DiffusionGemma text    -> skill labels    (cell supervisor over plain-language facts; cluster-wide cache)
  * privileged teacher     -> action labels   (multi_teacher.py)

Round 0 the teacher drives; round k the student drives with probability 1 - beta_k (beta_k = 0.5^k) and every
visited state is labelled. The part-ID head is trained on (height scan -> DiffusionGemma's vision answer) from the
vision set; the identity the skill/action heads are conditioned on is the ID head's own prediction (so training
matches what the student sees at run time). Each round: student-only evaluation + one rendered episode (progression).

Usage (cluster): RUN_TAG=multi-v1 k8s/ray/submit-dagger.sh dagger_multi.py --vision ignored/vision/v1/vision_labels.npz
       (local)  : app/venv/bin/python app/dagger_multi.py --local --workers 6 --rounds 2 --episodes 6 --no-dgemma
"""
from __future__ import annotations
import argparse, io, json, os, sys, tempfile, time
import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@ray.remote(max_concurrency=8)
class SkillCache:
    """DiffusionGemma cell supervisor behind one cache keyed by task-state class."""
    def __init__(self, url: str):
        from multi_skill_client import MultiSkillClient
        self.client = MultiSkillClient(url); self.cache = {}; self.audit = []; self.down_until = 0.0

    def ask(self, key: str, facts, fallback: str):
        if key in self.cache:
            return self.cache[key]
        if time.time() < self.down_until:
            return {"skill": fallback, "source": "privileged"}
        ans = self.client.ask(facts)
        self.audit.append({"flags": key, "dgemma": ans["skill"], "privileged": fallback, "source": ans["source"],
                           "latency": round(ans["latency"], 2), "facts": facts["facts"], "text": ans.get("text", ans.get("error", ""))})
        if ans["source"] == "dgemma":
            self.cache[key] = {"skill": ans["skill"], "source": "dgemma"}; return self.cache[key]
        self.down_until = time.time() + 60.0
        return {"skill": fallback, "source": "privileged"}

    def report(self):
        dg = [a for a in self.audit if a["source"] == "dgemma"]
        agree = sum(a["dgemma"] == a["privileged"] for a in dg)
        return {"state_classes": len(self.audit), "dgemma_answers": len(dg), "agreement_with_stage_machine": agree / len(dg) if dg else None,
                "audit": self.audit}


@ray.remote(num_cpus=1)
def rollout(blob, beta: float, seed: int, cache, render: bool, max_steps: int, parts, sheet_answers=None, start_stage: int = 0):
    from multi_env import MultiAssemblyEnv, SKILLS
    import multi_teacher as T
    from multi_skill_client import describe, flags
    pol = None
    if blob is not None:
        from student_multi import MultiStudent
        pol = MultiStudent.load(blob)
    env = MultiAssemblyEnv({"parts": tuple(parts), "render": render, "camera": "demo", "width": 640, "height": 480, "max_steps": max_steps})
    obs, info = env.reset(seed=seed); rng = np.random.default_rng(seed + 7)
    O, SK, SRC, A, ST = [], [], [], [], []; frames = []; ret = 0.0; student_steps = 0; seen = {}
    if isinstance(start_stage, str):                                       # focus curriculum: "part:stage" -> plan index
        q, st = start_stage.split(":")
        start_stage = next((i for i, (pp, ss) in enumerate(env.plan) if pp == q and ss == st), 0)
    while env.stage_i < start_stage and env.steps < max_steps - 400:       # start-stage curriculum: teacher drives there
        obs, _, term, _, info = env.step(T.act(env))
        if term:
            break
    for k in range(env.steps, max_steps):
        r = env._readings()
        if r["scanned"] and env.part and env.part not in seen:                 # identity as DiffusionGemma named it
            from inspection_facts import facts as scan_facts
            ans = (sheet_answers or {}).get(" ".join(scan_facts(env.scans_hi[env.part])))
            seen[env.part] = ans if ans in parts else (parts[pol.act(obs)[2]] if pol is not None else None)
        seen_as = seen.get(env.part) if env.part else None
        if cache is not None and env.stage != "done":
            lab = ray.get(cache.ask.remote(str(flags(r, seen_as)), describe(r, seen_as), env.stage))
        else:
            lab = {"skill": env.stage, "source": "privileged"}
        a_t = T.act(env)
        if pol is None or rng.random() < beta:
            a = a_t
        else:
            a = pol.act(obs)[0]; student_steps += 1
        O.append(obs); SK.append(SKILLS.index(lab["skill"])); SRC.append(lab["source"] == "dgemma"); A.append(a_t)
        ST.append(env.stage_i)
        obs, rew, term, trunc, info = env.step(a); ret += rew
        if render and k % 2 == 0:
            frames.append(env.render())
        if term or trunc:
            break
    video = None
    if frames:
        import imageio.v2 as imageio
        tmp = os.path.join(tempfile.gettempdir(), f"mev-{os.getpid()}-{seed}.mp4")
        imageio.mimwrite(tmp, frames, fps=10, codec="libx264", quality=8); video = open(tmp, "rb").read()
    n = info["n_stages"]
    return {"obs": np.asarray(O, np.float32), "skill": np.asarray(SK, np.int64), "dg": np.asarray(SRC, bool),
            "act": np.asarray(A, np.float32), "stage": np.asarray(ST, np.int64),
            "ep": {"seed": seed, "stages": int(info["stage_i"]), "n_stages": n, "success": bool(info["stage_i"] >= n),
                   "steps": k + 1, "return": round(ret, 2), "dropped": bool(info["dropped"]), "student_steps": student_steps,
                   "joined": int(info["n_joined"]), "order": info["order"]}, "video": video}


def train(pol, D, V, epochs: int, lr: float = 1e-3, log=print):
    """skill CE (DiffusionGemma text) + 4 x action MSE (teacher) + part-ID CE (DiffusionGemma vision, on the vision set)."""
    import torch, torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("TRAIN_THREADS", "4")))
    from student_multi import SCAN_DIM
    obs = torch.as_tensor(D["obs"]); act = torch.as_tensor(D["act"]); sk = torch.as_tensor(D["skill"])
    vs = torch.as_tensor(V["scans"]); vy = torch.as_tensor(V["y"])
    if float(pol.obs_std.sum()) == float(pol.obs_std.numel()):
        pol.set_norm(D["obs"])
    opt = torch.optim.Adam(pol.parameters(), lr=lr); n = len(obs); bs = 2048; pol.train()
    for ep in range(epochs):
        perm = torch.randperm(n); tot = np.zeros(4)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.no_grad():
                part = pol.id_head(obs[idx][:, -SCAN_DIM:]).argmax(-1)              # condition on the student's own ID
            out = pol(obs[idx], skill=sk[idx], part=part)
            l_sk = F.cross_entropy(out["skill_logits"], sk[idx])
            l_a = F.mse_loss(out["motion"], act[idx][:, :4].clamp(-1, 1)) \
                + 0.25 * F.binary_cross_entropy_with_logits(out["grip_logit"], (act[idx][:, 4] > 0).float())
            vi = torch.randint(0, len(vs), (256,))
            l_id = F.cross_entropy(pol.id_head(vs[vi]), vy[vi])
            loss = l_sk + 4.0 * l_a + l_id
            opt.zero_grad(); loss.backward(); opt.step()
            tot += [float(l_sk) * len(idx), float(l_a) * len(idx), float(l_id) * len(idx),
                    float((out["skill_logits"].argmax(-1) == sk[idx]).float().sum())]
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"    epoch {ep:3d}: skill CE {tot[0]/n:.3f}  action MSE {tot[1]/n:.4f}  part-ID CE {tot[2]/n:.3f}  skill acc {tot[3]/n:.3f}")
    pol.eval(); return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--episodes", type=int, default=112)
    ap.add_argument("--eval-episodes", type=int, default=28); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=2600); ap.add_argument("--parts", default="bracket,cap,lbracket")
    ap.add_argument("--vision", default="ignored/vision/v1/vision_labels.npz")
    ap.add_argument("--sheet-answers", default=None, help="DiffusionGemma inspection answers per fact sheet (vision_labeler)")
    ap.add_argument("--run-tag", default=time.strftime("multi-%Y%m%d-%H%M%S"))
    ap.add_argument("--dgemma-url", default=os.environ.get("DGEMMA_URL", "http://dgemma.dgemma.svc:8000"))
    ap.add_argument("--no-dgemma", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--init-policy", default=None); ap.add_argument("--init-data", default=None)
    ap.add_argument("--beta0", type=float, default=0.25); ap.add_argument("--round0", type=int, default=0)
    ap.add_argument("--tail-frac", type=float, default=0.5, help="episodes that start at a random stage (teacher drives there)")
    ap.add_argument("--max-samples", type=int, default=1_500_000)
    ap.add_argument("--focus", default="", help="comma list of part:stage start points for hard-stage practice")
    ap.add_argument("--focus-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate for warm-started rounds (fine-tuning: ~2e-4)")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out-dir", default=os.environ.get("ROBOT_OUT", "/tmp/robot-runs"))
    args = ap.parse_args(); parts = args.parts.split(",")
    ray.init(num_cpus=args.workers) if args.local else ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    from dagger_assembly import Store
    from student_multi import MultiStudent
    store = Store(args.run_tag, args.out_dir); lines = []
    def log(m):
        print(m, flush=True); lines.append(m)
    z = np.load(args.vision, allow_pickle=True)
    vlab = np.array([str(x) for x in z["dgemma"]]); keep = np.isin(vlab, parts)
    V = {"scans": z["scans"][keep].astype(np.float32), "y": np.array([parts.index(x) for x in vlab[keep]], np.int64)}
    vtrue = np.array([str(x) for x in z["true"]])[keep]
    log(f"=== multi-part DAgger {args.run_tag}: parts={parts} rounds={args.rounds} episodes={args.episodes} "
        f"cpus={ray.cluster_resources().get('CPU')} vision set={len(V['y'])} (DiffusionGemma vision accuracy "
        f"{float(np.mean(vlab[keep] == vtrue)):.1%} vs ground truth)")
    cache = None if args.no_dgemma else SkillCache.remote(args.dgemma_url)
    sheets = json.load(open(args.sheet_answers or os.path.join(os.path.dirname(args.vision), "sheet_answers.json"))) \
        if (args.sheet_answers or os.path.exists(os.path.join(os.path.dirname(args.vision), "sheet_answers.json"))) else {}
    log(f"DiffusionGemma inspection answers for {len(sheets)} fact sheets")
    D = None; pol = None; metrics = []; best = -1.0
    if args.init_policy:
        pol = MultiStudent.load(open(args.init_policy, "rb").read()); log(f"warm start from {args.init_policy}")
    if args.init_data:
        z0 = np.load(args.init_data); D = {k: z0[k] for k in z0.files}; log(f"aggregating onto {args.init_data}: {len(D['obs'])} samples")
    for rnd in range(args.round0, args.round0 + args.rounds):
        k = rnd - args.round0
        beta = (args.beta0 * 0.5 ** k) if args.init_policy else (1.0 if rnd == 0 else 0.5 ** rnd)
        blob = pol.blob() if pol is not None else None
        t0 = time.time()
        n_tail = int(args.episodes * args.tail_frac) if pol is not None else 0
        srng = np.random.default_rng(rnd)
        focus = [f for f in args.focus.split(",") if f]
        n_focus = int(args.episodes * args.focus_frac) if (pol is not None and focus) else 0
        starts = [focus[i % len(focus)] if i < n_focus else (int(srng.integers(1, 25)) if i < n_focus + n_tail else 0)
                  for i in range(args.episodes)]
        res = ray.get([rollout.remote(blob, beta, 1000 + rnd * 10000 + i, cache, False, args.max_steps, parts, sheets, starts[i])
                       for i in range(args.episodes)])
        new = {k: np.concatenate([r[k] for r in res]) for k in ("obs", "skill", "dg", "act", "stage")}
        D = new if D is None else {k: np.concatenate([D[k], new[k]]) for k in D}
        if len(D["obs"]) > args.max_samples:                                   # cap memory: thin older rounds, keep this one
            n_old = len(D["obs"]) - len(new["obs"]); keep_old = max(0, args.max_samples - len(new["obs"]))
            idx = np.concatenate([np.sort(np.random.default_rng(rnd).choice(n_old, keep_old, replace=False)),
                                  np.arange(n_old, len(D["obs"]))])
            D = {k: v[idx] for k, v in D.items()}
        log(f"round {rnd} beta={beta:.3f}: {len(new['obs'])} samples in {time.time() - t0:.0f}s (total {len(D['obs'])}, "
            f"DiffusionGemma skill labels {new['dg'].mean():.0%})"
            + (f"  teacher success {np.mean([r['ep']['success'] for r in res]):.0%}" if rnd == 0 else ""))
        if pol is None:
            from build_multi_cell import PARTS
            tmap = [[float(PARTS[q]["join"] == "screw"), float(PARTS[q]["join"] == "weld")] for q in parts]   # driver, torch
            pol = MultiStudent(D["obs"].shape[1], len(parts), tool_map=tmap)
        first = rnd == args.round0 and not args.init_policy                    # a fresh network gets the full schedule
        pol = train(pol, D, V, args.epochs if first else max(12, args.epochs // 2), lr=args.lr if not first else 1e-3, log=log)
        blob = pol.blob(); store.put(f"student_round{rnd}.pt", blob)
        dbuf = io.BytesIO(); np.savez_compressed(dbuf, **D); store.put("dataset_latest.npz", dbuf.getvalue())   # resumable
        import torch
        with torch.no_grad():
            id_acc = float((pol.id_head(torch.as_tensor(V["scans"])).argmax(-1).numpy() == np.array([parts.index(t) for t in vtrue])).mean())
        ev = ray.get([rollout.remote(blob, 0.0, 777000 + i, None, i == 0, args.max_steps, parts, sheets) for i in range(args.eval_episodes)])
        succ = float(np.mean([e["ep"]["success"] for e in ev])); frac = float(np.mean([e["ep"]["stages"] / e["ep"]["n_stages"] for e in ev]))
        joined = float(np.mean([e["ep"]["joined"] for e in ev])); ok = [e["ep"]["steps"] for e in ev if e["ep"]["success"]]
        m = {"round": rnd, "beta": beta, "samples": int(len(D["obs"])), "eval_success_rate": succ, "eval_stage_frac": frac,
             "eval_parts_joined": joined, "eval_steps_ok": float(np.mean(ok)) if ok else None, "part_id_acc_vs_truth": id_acc}
        metrics.append(m)
        log(f"round {rnd} EVAL student-only: full assemblies {succ:.0%}  stages {frac:.0%}  parts attached {joined:.2f}/{len(parts)}  "
            f"steps(ok) {m['eval_steps_ok']}  part-ID acc {id_acc:.1%}")
        if succ * 10 + frac > best:
            best = succ * 10 + frac; store.put("student_best.pt", blob); store.put("best.json", json.dumps(m).encode())
            log(f"round {rnd}: new best student")
        if ev[0]["video"]:
            store.put(f"eval_round{rnd:02d}_joined{ev[0]['ep']['joined']}_stages{ev[0]['ep']['stages']}.mp4", ev[0]["video"])
        store.put("metrics.json", json.dumps(metrics, indent=2).encode())
    if cache is not None:
        rep = ray.get(cache.report.remote()); store.put("dgemma_audit.json", json.dumps(rep, indent=2).encode())
        log(f"DiffusionGemma supervisor: {rep['dgemma_answers']} answers over {rep['state_classes']} task-state classes, "
            f"agreement with the stage machine {rep['agreement_with_stage_machine']}")
    store.put("log.txt", "\n".join(lines).encode()); log(f"=== done: {store.dir}")


if __name__ == "__main__":
    main()
