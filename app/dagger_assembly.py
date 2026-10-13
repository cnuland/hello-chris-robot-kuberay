#!/usr/bin/env python
"""DAgger on KubeRay (v004): distill the privileged teacher + the DiffusionGemma skill supervisor into the lightweight
student, round by round, with a sampled evaluation video per round (the progression reel).

Round 0      : teacher drives (beta = 1) on every Ray worker; behaviour cloning on the aggregate.
Round k >= 1 : the STUDENT drives with probability 1 - beta_k (beta_k = 0.5^k); every visited state is labelled by
               the teacher (action) and DiffusionGemma (skill); aggregate; retrain.
Each round   : student-only evaluation (success rate, stages reached, steps) + one rendered evaluation episode.

DiffusionGemma sits behind one Ray actor (SkillCache) that asks the vLLM service once per task-state class (the coarse
flags in skill_client.flags) and serves every worker from that cache - one T4 replica cannot answer 50+ concurrent
episodes at 1 Hz each, and the skill choice depends on the coarse task state anyway.

Usage (cluster): RUN_TAG=dagger-v1 k8s/ray/submit-dagger.sh --rounds 6 --episodes 56
       (local)  : app/venv/bin/python app/dagger_assembly.py --local --rounds 2 --episodes 4 --workers 4
"""
from __future__ import annotations
import argparse, io, json, os, sys, tempfile, time
import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@ray.remote(max_concurrency=8)
class SkillCache:
    """Cluster-wide DiffusionGemma cache keyed by task-state class."""
    def __init__(self, url: str):
        from skill_client import SkillClient
        self.client = SkillClient(url, timeout_s=60.0); self.cache = {}; self.calls = 0; self.audit = []; self.down_until = 0.0

    def ask(self, key, readings_small, fallback_stage: str):
        import time as _t
        k = str(key)
        if k in self.cache:
            return self.cache[k]
        if _t.time() < getattr(self, "down_until", 0.0):
            return {"skill": fallback_stage, "source": "privileged"}            # service down: answer instantly
        self.calls += 1
        ans = self.client.ask(readings_small, fallback_stage, force=True)
        self.audit.append({"flags": k, "dgemma": ans["skill"], "privileged": fallback_stage, "source": ans["source"],
                           "latency": round(ans.get("latency", 0.0), 2)})
        if ans["source"] == "dgemma":
            self.cache[k] = {"skill": ans["skill"], "source": "dgemma"}         # cache only genuine answers
            return self.cache[k]
        self.down_until = _t.time() + 60.0                                      # fail fast for a minute, then retry
        return {"skill": fallback_stage, "source": "privileged"}

    def report(self):
        n = len(self.audit); dg = [a for a in self.audit if a["source"] == "dgemma"]
        agree = sum(a["dgemma"] == a["privileged"] for a in dg)
        return {"state_classes": n, "dgemma_answers": len(dg), "fallbacks": n - len(dg),
                "agreement_with_stage_machine": (agree / len(dg)) if dg else None, "audit": self.audit}


def _small(r):
    keys = ("tcp", "part_pos", "key_err_deg", "part_held", "part_in_pocket", "on_stop", "on_belt", "tool_held",
            "tool_in_holder", "weld_on", "twist", "finger_gap")
    return {k: (np.asarray(r[k]).tolist() if isinstance(r[k], np.ndarray) else r[k]) for k in keys}


@ray.remote(num_cpus=1)
def rollout(policy_blob, beta: float, seed: int, cache, render: bool, max_steps: int):
    import numpy as np
    from assembly_env import AssemblyEnv, STAGES
    import assembly_teacher as T
    from skill_client import SKILL_NAMES, flags
    policy = None
    if policy_blob is not None:
        import torch
        from student_policy import StudentPolicy
        path = os.path.join(tempfile.gettempdir(), f"pol-{os.getpid()}.pt"); open(path, "wb").write(policy_blob)
        policy = StudentPolicy.load(path)
    env = AssemblyEnv({"render": render, "width": 480, "height": 368, "max_steps": max_steps})
    obs, info = env.reset(seed=seed); rng = np.random.default_rng(seed + 7)
    O, SK, SRC, A, ST = [], [], [], [], []; frames = []; ret = 0.0; student_steps = 0
    for k in range(max_steps):
        r = env._readings()
        lab = ray.get(cache.ask.remote(flags(r), _small(r), env.stage)) if cache is not None else {"skill": env.stage, "source": "privileged"}
        a_t = T.act(env)
        if policy is None or rng.random() < beta:
            a = a_t
        else:
            a, _ = policy.act(obs); student_steps += 1
        O.append(obs); SK.append(SKILL_NAMES.index(lab["skill"])); SRC.append(lab["source"] == "dgemma")
        A.append(a_t); ST.append(STAGES.index(env.stage) if env.stage in STAGES else len(STAGES) - 1)
        obs, rew, term, trunc, info = env.step(a); ret += rew
        if render and k % 2 == 0:
            frames.append(env.render())
        if term or trunc:
            break
    video = None
    if frames:
        import imageio.v2 as imageio
        tmp = os.path.join(tempfile.gettempdir(), f"ev-{os.getpid()}-{seed}.mp4")
        imageio.mimwrite(tmp, frames, fps=10, codec="libx264", quality=8); video = open(tmp, "rb").read()
    return {"obs": np.asarray(O, np.float32), "skill": np.asarray(SK, np.int64), "dg": np.asarray(SRC, bool),
            "act": np.asarray(A, np.float32), "stage": np.asarray(ST, np.int64),
            "ep": {"seed": seed, "stages": int(info["stage_i"]), "success": bool(info["stage_i"] >= len(STAGES)),
                   "steps": k + 1, "return": round(ret, 2), "dropped": bool(info["dropped"]), "student_steps": student_steps},
            "video": video}


def train(policy, D, epochs: int, lr: float = 1e-3, log=print):
    import torch, torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get("TRAIN_THREADS", "4")))         # pod CPU limit, not the host core count
    obs = torch.as_tensor(D["obs"]); act = torch.as_tensor(D["act"]); sk = torch.as_tensor(D["skill"])
    if float(policy.obs_std.sum()) == float(policy.obs_std.numel()):            # normalise once (fresh net); never shift it later
        policy.set_norm(D["obs"])
    opt = torch.optim.Adam(policy.parameters(), lr=lr); n = len(obs); bs = 1024
    policy.train()
    for ep in range(epochs):
        perm = torch.randperm(n); tot = [0.0, 0.0, 0.0]
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = policy(obs[idx], skill=sk[idx])                                # teacher-forced skill conditioning
            l_sk = F.cross_entropy(out["skill_logits"], sk[idx]); l_a = F.mse_loss(out["mean"], act[idx].clamp(-1, 1))
            loss = l_sk + 4.0 * l_a
            opt.zero_grad(); loss.backward(); opt.step()
            tot[0] += float(l_sk.detach()) * len(idx); tot[1] += float(l_a.detach()) * len(idx)
            tot[2] += float((out["skill_logits"].argmax(-1) == sk[idx]).float().sum())
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"    train epoch {ep:3d}: skill CE {tot[0]/n:.3f}  action MSE {tot[1]/n:.4f}  skill acc {tot[2]/n:.3f}")
    policy.eval(); return policy


class Store:
    """Results to the head's local dir and to MinIO (s3://robot/<run_tag>/)."""
    def __init__(self, run_tag: str, out_dir: str):
        self.run_tag, self.dir = run_tag, os.path.join(out_dir, run_tag); os.makedirs(self.dir, exist_ok=True); self.s3 = None
        if os.environ.get("S3_ENDPOINT_URL"):
            try:
                import boto3
                self.s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
                                       aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"), region_name="us-east-1")
            except Exception as exc:
                print("s3 disabled:", exc)

    def put(self, name: str, data: bytes):
        open(os.path.join(self.dir, name), "wb").write(data)
        if self.s3:
            try:
                self.s3.put_object(Bucket=os.environ.get("S3_BUCKET", "robot"), Key=f"{self.run_tag}/{name}", Body=data)
            except Exception as exc:
                print("s3 put failed:", name, exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--episodes", type=int, default=56)
    ap.add_argument("--eval-episodes", type=int, default=28); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=900); ap.add_argument("--run-tag", default=time.strftime("dagger-%Y%m%d-%H%M%S"))
    ap.add_argument("--dgemma-url", default=os.environ.get("DGEMMA_URL", "http://dgemma.dgemma.svc:8000"))
    ap.add_argument("--no-dgemma", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out-dir", default=os.environ.get("ROBOT_OUT", "/tmp/robot-runs"))
    ap.add_argument("--init-policy", default=None, help="warm-start DAgger from a student checkpoint (student drives from round 0)")
    ap.add_argument("--beta0", type=float, default=None, help="beta for round 0 when warm-starting (default 0.5)")
    ap.add_argument("--init-data", default=None, help="aggregate onto a previous run's dataset_latest.npz")
    args = ap.parse_args()
    ray.init(num_cpus=args.workers) if args.local else ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    import torch
    from student_policy import StudentPolicy
    store = Store(args.run_tag, args.out_dir); log_lines = []
    def log(msg):
        print(msg, flush=True); log_lines.append(msg)
    cache = None if args.no_dgemma else SkillCache.remote(args.dgemma_url)
    log(f"=== DAgger {args.run_tag}: rounds={args.rounds} episodes/round={args.episodes} cluster_cpus={ray.cluster_resources().get('CPU')} "
        f"dgemma={'off' if cache is None else args.dgemma_url}")
    D = None; policy = None; metrics = []; seed_base = 1000
    if args.init_policy:
        policy = StudentPolicy.load(args.init_policy); log(f"warm start from {args.init_policy}")
    if args.init_data:
        z = np.load(args.init_data); D = {k: z[k] for k in z.files}; log(f"aggregating onto {args.init_data}: {len(D['obs'])} samples")
    for rnd in range(args.rounds):
        if args.init_policy:
            beta = (args.beta0 if args.beta0 is not None else 0.5) * (0.5 ** rnd)
        else:
            beta = 1.0 if rnd == 0 else 0.5 ** rnd
        blob = None
        if policy is not None:
            buf = io.BytesIO(); torch.save({"state": policy.state_dict(), "obs_dim": policy.obs_dim, "act_dim": policy.act_dim}, buf); blob = buf.getvalue()
        t0 = time.time()
        res = ray.get([rollout.remote(blob, beta, seed_base + rnd * 10000 + i, cache, False, args.max_steps) for i in range(args.episodes)])
        new = {k: np.concatenate([r[k] for r in res]) for k in ("obs", "skill", "dg", "act", "stage")}
        D = new if D is None else {k: np.concatenate([D[k], new[k]]) for k in D}
        coll = {"collect_s": round(time.time() - t0, 1), "samples_new": int(len(new["obs"])), "samples_total": int(len(D["obs"])),
                "dgemma_label_frac": float(new["dg"].mean()), "teacher_success_in_collection": float(np.mean([r["ep"]["success"] for r in res])) if rnd == 0 else None}
        log(f"round {rnd} beta={beta:.3f}: collected {coll['samples_new']} samples in {coll['collect_s']}s (total {coll['samples_total']}, "
            f"DiffusionGemma-labelled {coll['dgemma_label_frac']:.0%})")
        policy = StudentPolicy(D["obs"].shape[1]) if policy is None else policy
        policy = train(policy, D, args.epochs if (rnd == 0 and not args.init_policy) else max(15, args.epochs // 2), log=log)
        buf = io.BytesIO(); torch.save({"state": policy.state_dict(), "obs_dim": policy.obs_dim, "act_dim": policy.act_dim}, buf); blob = buf.getvalue()
        store.put(f"student_round{rnd}.pt", blob)
        dbuf = io.BytesIO(); np.savez_compressed(dbuf, **D); store.put("dataset_latest.npz", dbuf.getvalue())   # resumable
        t1 = time.time()
        ev = ray.get([rollout.remote(blob, 0.0, 777000 + i, None, i == 0, args.max_steps) for i in range(args.eval_episodes)])
        succ = float(np.mean([e["ep"]["success"] for e in ev])); stages = float(np.mean([e["ep"]["stages"] for e in ev]))
        steps_ok = [e["ep"]["steps"] for e in ev if e["ep"]["success"]]
        m = {"round": rnd, "beta": beta, **coll, "eval_success_rate": succ, "eval_mean_stages": stages,
             "eval_steps_when_success": (float(np.mean(steps_ok)) if steps_ok else None), "eval_s": round(time.time() - t1, 1),
             "eval_stage_hist": np.bincount([e["ep"]["stages"] for e in ev], minlength=9).tolist()}
        metrics.append(m)
        log(f"round {rnd} EVAL student-only: success {succ:.0%}  mean stages {stages:.2f}/8  hist {m['eval_stage_hist']}")
        score = succ * 10 + stages
        if score > getattr(main, "_best", -1):
            main._best = score; store.put("student_best.pt", blob)
            store.put("best.json", json.dumps({"round": rnd, "eval_success_rate": succ, "eval_mean_stages": stages}).encode())
            log(f"round {rnd}: new best student (success {succ:.0%}, stages {stages:.2f})")
        if ev[0]["video"]:
            store.put(f"eval_round{rnd:02d}_seed777000_stages{ev[0]['ep']['stages']}.mp4", ev[0]["video"])
        store.put("metrics.json", json.dumps(metrics, indent=2).encode())
    if cache is not None:
        rep = ray.get(cache.report.remote()); store.put("dgemma_audit.json", json.dumps(rep, indent=2).encode())
        log(f"DiffusionGemma: {rep['dgemma_answers']} answers over {rep['state_classes']} task-state classes, "
            f"agreement with the stage machine {rep['agreement_with_stage_machine']}")
    store.put("log.txt", "\n".join(log_lines).encode())
    log(f"=== done: results in {store.dir} (and s3://robot/{args.run_tag}/ when MinIO is configured)")


if __name__ == "__main__":
    main()
