#!/usr/bin/env python
"""PPO fine-tuning of the distilled multi-part student (v006c) - the RL step after DAgger.

The env rewards every stage (dense shaping toward the stage goal, +10 per completed stage, -10 drop, -5 regression,
small time penalty). Ray workers collect rollouts with Gaussian exploration on the 4-D motion only - the gripper stays
the deterministic classifier (an exploratory open drops a part). Lessons from v005: the value head reads DETACHED
trunk features, a critic warmup precedes policy updates, and an anchor term keeps the motion near the distilled
policy. Start-stage curriculum (random + focus points) as in dagger_multi. Evaluation every `--eval-every` iterations
(student-only, deterministic) with a rendered episode.

Usage: app/venv/bin/python app/ppo_multi.py --local --workers 14 --init ignored/checkpoints/multi_final_student.pt
"""
from __future__ import annotations
import argparse, io, json, os, sys, tempfile, time
import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@ray.remote(num_cpus=1)
def collect(blob, seed, max_steps, deterministic, render, start_stage, std):
    import torch
    from multi_env import MultiAssemblyEnv
    import multi_teacher as T
    from student_multi import MultiStudent
    pol = MultiStudent.load(blob); torch.manual_seed(seed)
    env = MultiAssemblyEnv({"render": render, "camera": "demo", "width": 640, "height": 480, "max_steps": max_steps})
    obs, info = env.reset(seed=seed)
    if isinstance(start_stage, str):
        q, st = start_stage.split(":")
        start_stage = next((i for i, (pp, ss) in enumerate(env.plan) if pp == q and ss == st), 0)
    while env.stage_i < start_stage and env.steps < max_steps - 400:
        obs, _, term, _, info = env.step(T.act(env))
        if term:
            break
    O, A, LP, R, V, D = [], [], [], [], [], []; frames = []; ret = 0.0; s0 = env.stage_i; term = False
    sd = torch.full((4,), std)
    for k in range(env.steps, max_steps):
        with torch.no_grad():
            out = pol(torch.as_tensor(obs)[None]); mean = out["motion"][0]; v = float(out["value"][0])
            grip = 1.0 if float(out["grip_logit"][0]) > 0 else -1.0
        m = mean if deterministic else mean + sd * torch.randn_like(mean)
        lp = float(torch.distributions.Normal(mean, sd).log_prob(m).sum())
        a = np.concatenate([m.clamp(-1, 1).numpy(), [grip]]).astype(np.float32)
        nobs, rew, term, trunc, info = env.step(a); ret += rew
        O.append(obs); A.append(m.numpy()); LP.append(lp); R.append(rew); V.append(v); D.append(term)
        obs = nobs
        if render and k % 2 == 0:
            frames.append(env.render())
        if term or trunc:
            break
    with torch.no_grad():
        last_v = 0.0 if term else float(pol(torch.as_tensor(obs)[None])["value"][0])
    video = None
    if frames:
        import imageio.v2 as imageio
        tmp = os.path.join(tempfile.gettempdir(), f"ppo-m-{os.getpid()}-{seed}.mp4")
        imageio.mimwrite(tmp, frames, fps=10, codec="libx264", quality=8); video = open(tmp, "rb").read()
    n = info["n_stages"]
    return {"obs": np.asarray(O, np.float32), "act": np.asarray(A, np.float32), "logp": np.asarray(LP, np.float32),
            "rew": np.asarray(R, np.float32), "val": np.asarray(V, np.float32), "done": np.asarray(D, bool), "last_v": last_v,
            "ep": {"seed": seed, "stages": int(info["stage_i"]), "n": n, "start": s0, "success": bool(info["stage_i"] >= n),
                   "joined": int(info["n_joined"]), "steps": env.steps, "return": round(ret, 2)}, "video": video}


def gae(t, gamma=0.995, lam=0.95):
    r, v, d = t["rew"], t["val"], t["done"]; n = len(r); adv = np.zeros(n, np.float32); last = 0.0
    for i in reversed(range(n)):
        nv = t["last_v"] if i == n - 1 else v[i + 1]; nt = 0.0 if d[i] else 1.0
        delta = r[i] + gamma * nv * nt - v[i]; last = delta + gamma * lam * nt * last; adv[i] = last
    return adv, adv + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=56); ap.add_argument("--eval-episodes", type=int, default=28)
    ap.add_argument("--eval-every", type=int, default=4); ap.add_argument("--max-steps", type=int, default=2600)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--anchor", type=float, default=0.5); ap.add_argument("--critic-warmup", type=int, default=2)
    ap.add_argument("--std", type=float, default=0.04); ap.add_argument("--tail-frac", type=float, default=0.5)
    ap.add_argument("--focus", default="lbracket:place,lbracket:get_tool,cap:press")
    ap.add_argument("--run-tag", default=time.strftime("ppo-multi-%Y%m%d-%H%M%S")); ap.add_argument("--local", action="store_true")
    ap.add_argument("--workers", type=int, default=14); ap.add_argument("--out-dir", default="ignored/runs")
    args = ap.parse_args()
    ray.init(num_cpus=args.workers) if args.local else ray.init(address="auto")
    import torch
    torch.set_num_threads(int(os.environ.get("TRAIN_THREADS", "8")))
    from student_multi import MultiStudent
    from dagger_assembly import Store
    store = Store(args.run_tag, args.out_dir); lines = []
    def log(m):
        print(m, flush=True); lines.append(m)
    blob0 = open(args.init, "rb").read(); pol = MultiStudent.load(blob0); ref = MultiStudent.load(blob0)
    v_params = list(pol.value_head.parameters()); v_ids = {id(p) for p in v_params}
    pi_params = [p for n, p in pol.named_parameters() if id(p) not in v_ids and not n.startswith(("id_head", "skill_head", "log_std"))]
    opt = torch.optim.Adam(pi_params, lr=args.lr); v_opt = torch.optim.Adam(v_params, lr=1e-3)
    focus = [f for f in args.focus.split(",") if f]; metrics = []
    log(f"=== PPO {args.run_tag}: init={args.init} iters={args.iters} episodes={args.episodes} lr={args.lr} std={args.std}")

    def evaluate(it):
        b = pol.blob()
        ev = ray.get([collect.remote(b, 777000 + i, args.max_steps, True, i == 0, 0, args.std) for i in range(args.eval_episodes)])
        succ = float(np.mean([e["ep"]["success"] for e in ev])); st = float(np.mean([e["ep"]["stages"] / e["ep"]["n"] for e in ev]))
        jn = float(np.mean([e["ep"]["joined"] for e in ev]))
        if ev[0]["video"]:
            store.put(f"ppo_eval_iter{it:03d}_joined{ev[0]['ep']['joined']}.mp4", ev[0]["video"])
        store.put(f"ppo_iter{it:03d}.pt", b)
        return succ, st, jn

    succ, st, jn = evaluate(0)
    log(f"iter   0 (DAgger student) EVAL: full assemblies {succ:.0%}  stages {st:.0%}  parts {jn:.2f}/3")
    metrics.append({"iter": 0, "eval_success": succ, "eval_stages": st, "eval_parts": jn})
    pol.train()
    for it in range(1, args.iters + 1):
        t0 = time.time(); b = pol.blob(); rng = np.random.default_rng(it)
        n_tail = int(args.episodes * args.tail_frac)
        starts = [(focus[i % len(focus)] if (i % 2 == 0 and focus) else int(rng.integers(1, 25))) if i < n_tail else 0
                  for i in range(args.episodes)]
        trajs = ray.get([collect.remote(b, 50000 * it + i, args.max_steps, False, False, starts[i], args.std) for i in range(args.episodes)])
        adv, ret = zip(*[gae(t) for t in trajs])
        obs = torch.as_tensor(np.concatenate([t["obs"] for t in trajs])); act = torch.as_tensor(np.concatenate([t["act"] for t in trajs]))
        olp = torch.as_tensor(np.concatenate([t["logp"] for t in trajs])); adv = torch.as_tensor(np.concatenate(adv))
        ret = torch.as_tensor(np.concatenate(ret)); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        with torch.no_grad():
            ref_m = ref(obs)["motion"]
        sd = torch.full((4,), args.std); warm = it <= args.critic_warmup; n = len(obs)
        for _ in range(4):
            perm = torch.randperm(n)
            for i in range(0, n, 4096):
                idx = perm[i:i + 4096]; out = pol(obs[idx])
                v_loss = ((out["value"] - ret[idx]) ** 2).mean()
                if warm:
                    loss = v_loss
                else:
                    lp = torch.distributions.Normal(out["motion"], sd).log_prob(act[idx]).sum(-1)
                    ratio = torch.exp(lp - olp[idx])
                    pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[idx]).mean()
                    loss = pg + 0.5 * v_loss + args.anchor * ((out["motion"] - ref_m[idx]) ** 2).mean()
                opt.zero_grad(); v_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(v_params, 10.0); v_opt.step()
                if not warm:
                    torch.nn.utils.clip_grad_norm_(pi_params, 1.0); opt.step()
        full = [t for t in trajs if t["ep"]["start"] == 0]
        fs = float(np.mean([t["ep"]["success"] for t in full])) if full else float("nan")
        tail = [t for t in trajs if t["ep"]["start"] > 0]
        tc = float(np.mean([(t["ep"]["stages"] - t["ep"]["start"]) / max(1, t["ep"]["n"] - t["ep"]["start"]) for t in tail])) if tail else float("nan")
        m = {"iter": it, "samples": n, "s": round(time.time() - t0), "train_full_success": fs, "tail_completion": tc,
             "mean_return": float(np.mean([t["ep"]["return"] for t in trajs])), "critic_warmup": warm}
        line = (f"iter {it:3d}{' (critic warmup)' if warm else ''}: {n} samples {m['s']}s  full-episode success {fs:.0%}  "
                f"mid-task completion {tc:.0%}  return {m['mean_return']:.1f}")
        if it % args.eval_every == 0 or it == args.iters:
            pol.eval(); succ, st, jn = evaluate(it); pol.train()
            m.update(eval_success=succ, eval_stages=st, eval_parts=jn)
            line += f"  | EVAL full assemblies {succ:.0%} stages {st:.0%} parts {jn:.2f}/3"
        metrics.append(m); log(line); store.put("ppo_metrics.json", json.dumps(metrics, indent=2).encode())
    store.put("ppo_log.txt", "\n".join(lines).encode()); log(f"=== done: {store.dir}")


if __name__ == "__main__":
    main()
