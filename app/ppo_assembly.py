#!/usr/bin/env python
"""PPO fine-tuning on KubeRay (v005): start from the DAgger-distilled student and improve it on the per-stage rewards.

The environment rewards every stage (reach, grasp, lift, rotate, place, get_tool, fasten, finish): dense shaping toward
the current stage goal, +10 per completed stage, -10 for a drop, -5 for a regression, a small time penalty. Ray workers
collect stochastic rollouts in parallel; the driver (Ray head) computes GAE and runs clipped PPO. A short critic warmup
and an anchor to the DAgger policy keep the first updates from wrecking the distilled behaviour. Metrics every
iteration and a sampled deterministic evaluation video every `--video-every` iterations (the progression reel).

Usage (cluster): RUN_TAG=ppo-v1 k8s/ray/submit-dagger.sh ppo_assembly.py --init s3://robot/dagger-v1/student_round5.pt
       (local)  : app/venv/bin/python app/ppo_assembly.py --local --workers 6 --iters 3 --init <student.pt>
"""
from __future__ import annotations
import argparse, io, json, os, sys, tempfile, time
import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_blob(src: str) -> bytes:
    if src.startswith("s3://"):
        import boto3
        b, k = src[5:].split("/", 1)
        s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"), region_name="us-east-1")
        return s3.get_object(Bucket=b, Key=k)["Body"].read()
    return open(src, "rb").read()


def _policy_from_blob(blob: bytes):
    from student_policy import StudentPolicy
    path = os.path.join(tempfile.gettempdir(), f"ppo-{os.getpid()}-{time.time_ns()}.pt"); open(path, "wb").write(blob)
    return StudentPolicy.load(path)


@ray.remote(num_cpus=1)
def collect(blob: bytes, seed: int, max_steps: int, deterministic: bool, render: bool, start_stage: str = None,
            time_penalty: float = 0.01):
    import torch
    from assembly_env import AssemblyEnv
    import assembly_teacher as T
    pol = _policy_from_blob(blob); torch.manual_seed(seed)
    env = AssemblyEnv({"render": render, "width": 480, "height": 368, "max_steps": max_steps, "time_penalty": time_penalty})
    obs, info = env.reset(seed=seed)
    if start_stage:                                     # tail curriculum: the teacher drives up to `start_stage`
        while env.stage != start_stage and env.steps < max_steps - 200:
            obs, _, term, _, info = env.step(T.act(env))
            if term:
                break
    O, A, LP, R, V, D = [], [], [], [], [], []; frames = []; ret = 0.0
    std = torch.exp(pol.log_std).detach().clone(); std[4] = 1e-6          # no exploration on the binary gripper command
    for k in range(max_steps):
        with torch.no_grad():
            out = pol(torch.as_tensor(obs)[None]); mean = out["mean"][0]; v = float(out["value"][0])
        raw = mean if deterministic else mean + std * torch.randn_like(mean)
        lp = float(torch.distributions.Normal(mean, std).log_prob(raw).sum())
        a = raw.clamp(-1, 1).numpy()
        nobs, rew, term, trunc, info = env.step(a); ret += rew
        O.append(obs); A.append(raw.numpy()); LP.append(lp); R.append(rew); V.append(v); D.append(term)
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
        tmp = os.path.join(tempfile.gettempdir(), f"ppo-ev-{os.getpid()}-{seed}.mp4")
        imageio.mimwrite(tmp, frames, fps=10, codec="libx264", quality=8); video = open(tmp, "rb").read()
    return {"obs": np.asarray(O, np.float32), "act": np.asarray(A, np.float32), "logp": np.asarray(LP, np.float32),
            "rew": np.asarray(R, np.float32), "val": np.asarray(V, np.float32), "done": np.asarray(D, bool), "last_v": last_v,
            "ep": {"seed": seed, "stages": int(info["stage_i"]), "success": bool(info["stage_i"] >= 8), "steps": k + 1,
                   "return": round(ret, 2), "dropped": bool(info["dropped"])}, "video": video}


def gae(traj, gamma=0.99, lam=0.95):
    r, v, d = traj["rew"], traj["val"], traj["done"]; n = len(r); adv = np.zeros(n, np.float32); last = 0.0
    for t in reversed(range(n)):
        nv = traj["last_v"] if t == n - 1 else v[t + 1]; nonterm = 0.0 if d[t] else 1.0
        delta = r[t] + gamma * nv * nonterm - v[t]; last = delta + gamma * lam * nonterm * last; adv[t] = last
    return adv, adv + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=56); ap.add_argument("--eval-episodes", type=int, default=28)
    ap.add_argument("--max-steps", type=int, default=900); ap.add_argument("--video-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--anchor", type=float, default=0.5); ap.add_argument("--critic-warmup", type=int, default=2)
    ap.add_argument("--init-std", type=float, default=0.05); ap.add_argument("--tail-frac", type=float, default=0.5)
    ap.add_argument("--tail-stage", default="finish"); ap.add_argument("--time-penalty", type=float, default=0.01)
    ap.add_argument("--run-tag", default=time.strftime("ppo-%Y%m%d-%H%M%S")); ap.add_argument("--local", action="store_true")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out-dir", default=os.environ.get("ROBOT_OUT", "/tmp/robot-runs"))
    args = ap.parse_args()
    ray.init(num_cpus=args.workers) if args.local else ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    import torch
    from dagger_assembly import Store
    store = Store(args.run_tag, args.out_dir); lines = []
    def log(m):
        print(m, flush=True); lines.append(m)
    init_blob = _load_blob(args.init); pol = _policy_from_blob(init_blob); ref = _policy_from_blob(init_blob)
    with torch.no_grad():
        pol.log_std.fill_(float(np.log(args.init_std)))                     # gentle exploration for a long-horizon task
    pol.train(); metrics = []
    v_params = list(pol.value_head.parameters()); v_ids = {id(q) for q in v_params}
    pi_params = [q for q in pol.parameters() if id(q) not in v_ids]
    opt = torch.optim.Adam(pi_params, lr=args.lr); v_opt = torch.optim.Adam(v_params, lr=1e-3)   # critic clipped separately
    log(f"=== PPO {args.run_tag}: init={args.init} iters={args.iters} episodes/iter={args.episodes} cpus={ray.cluster_resources().get('CPU')}")

    def blob_of(p):
        b = io.BytesIO(); torch.save({"state": p.state_dict(), "obs_dim": p.obs_dim, "act_dim": p.act_dim}, b); return b.getvalue()

    def evaluate(it, video: bool):
        b = blob_of(pol)
        ev = ray.get([collect.remote(b, 555000 + i, args.max_steps, True, video and i == 0) for i in range(args.eval_episodes)])
        succ = float(np.mean([e["ep"]["success"] for e in ev])); st = float(np.mean([e["ep"]["stages"] for e in ev]))
        ok = [e["ep"]["steps"] for e in ev if e["ep"]["success"]]
        if video and ev[0]["video"]:
            store.put(f"ppo_eval_iter{it:03d}_stages{ev[0]['ep']['stages']}.mp4", ev[0]["video"])
        return succ, st, (float(np.mean(ok)) if ok else None), np.bincount([e["ep"]["stages"] for e in ev], minlength=9).tolist()

    succ, st, stp, hist = evaluate(0, True)
    log(f"iter   0 (DAgger init) EVAL: success {succ:.0%}  mean stages {st:.2f}/8  steps(ok) {stp}  hist {hist}")
    metrics.append({"iter": 0, "eval_success_rate": succ, "eval_mean_stages": st, "eval_steps_when_success": stp, "eval_stage_hist": hist})
    for it in range(1, args.iters + 1):
        t0 = time.time(); b = blob_of(pol)
        n_tail = int(args.episodes * args.tail_frac)
        trajs = ray.get([collect.remote(b, 10000 * it + i, args.max_steps, False, False, args.tail_stage if i < n_tail else None, args.time_penalty)
                         for i in range(args.episodes)])
        advs, rets = zip(*[gae(t) for t in trajs])
        obs = torch.as_tensor(np.concatenate([t["obs"] for t in trajs])); act = torch.as_tensor(np.concatenate([t["act"] for t in trajs]))
        old_lp = torch.as_tensor(np.concatenate([t["logp"] for t in trajs])); adv = torch.as_tensor(np.concatenate(advs))
        ret = torch.as_tensor(np.concatenate(rets)); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        with torch.no_grad():
            ref_out = ref(obs); ref_mean = ref_out["mean"]; ref_skill = ref_out["skill_logits"].argmax(-1)
        n = len(obs); bs = 4096; warm = it <= args.critic_warmup
        for _ in range(4):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; out = pol(obs[idx]); std = torch.exp(pol.log_std)
                v_loss = ((out["value"] - ret[idx]) ** 2).mean()
                if warm:
                    loss = v_loss                                                  # critic warmup: policy untouched
                else:
                    lp = torch.distributions.Normal(out["mean"], std).log_prob(act[idx]).sum(-1)
                    ratio = torch.exp(lp - old_lp[idx])
                    pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[idx]).mean()
                    anchor = ((out["mean"] - ref_mean[idx]) ** 2).mean()
                    skill_anchor = torch.nn.functional.cross_entropy(out["skill_logits"], ref_skill[idx])  # keep DGemma's skills
                    loss = pg + 0.5 * v_loss + args.anchor * (anchor + 0.1 * skill_anchor) - 0.001 * torch.distributions.Normal(out["mean"], std).entropy().sum(-1).mean()
                opt.zero_grad(); v_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(v_params, 10.0); v_opt.step()
                if not warm:
                    torch.nn.utils.clip_grad_norm_(pi_params, 1.0); opt.step()
        full = trajs[n_tail:] or trajs; tail = trajs[:n_tail]
        tr_succ = float(np.mean([t["ep"]["success"] for t in full])); tr_ret = float(np.mean([t["ep"]["return"] for t in full]))
        tr_st = float(np.mean([t["ep"]["stages"] for t in full]))
        tail_succ = float(np.mean([t["ep"]["success"] for t in tail])) if tail else None
        m = {"iter": it, "samples": int(n), "collect_update_s": round(time.time() - t0, 1), "train_success_rate": tr_succ,
             "tail_success_rate": tail_succ,
             "train_mean_return": round(tr_ret, 2), "train_mean_stages": tr_st, "critic_warmup": warm}
        line = (f"iter {it:3d}{' (critic warmup)' if warm else ''}: {n} samples  full-episode success {tr_succ:.0%}  stages {tr_st:.2f}  "
                f"return {tr_ret:.1f}  | {args.tail_stage}-start success {tail_succ if tail_succ is None else f'{tail_succ:.0%}'}")
        if it % args.video_every == 0 or it == args.iters:
            succ, st, stp, hist = evaluate(it, True)
            m.update(eval_success_rate=succ, eval_mean_stages=st, eval_steps_when_success=stp, eval_stage_hist=hist)
            line += f"  | EVAL success {succ:.0%} stages {st:.2f} steps(ok) {stp} hist {hist}"
            store.put(f"ppo_iter{it:03d}.pt", blob_of(pol))
        metrics.append(m); log(line); store.put("ppo_metrics.json", json.dumps(metrics, indent=2).encode())
    store.put("ppo_log.txt", "\n".join(lines).encode()); log(f"=== done: {store.dir}")


if __name__ == "__main__":
    main()
