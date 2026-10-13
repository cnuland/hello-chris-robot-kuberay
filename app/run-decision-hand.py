#!/usr/bin/env python
"""Runner - the run-decision-dd.py of this project, reduced to its skeleton.

Per episode: load a fixture scene (or reset), then each step:
  1. perception -> structured obs
  2. planner-first: if no plan is executing, ask the teacher for one (budgeted); replay-verified plans execute
     unconditionally (plans outrank reflexes, reflexes outrank the student - the dd precedence)
  3. otherwise ask the student, pass it through the reflexes
  4. log a DAgger record (student action vs the teacher's label) when both exist
  5. trace violations on one line; snapshot a fixture every N steps and at every failure
Episodes run on Ray workers; --local runs one in-process.

Usage: python run-decision-hand.py --local --stage lift --max-steps 400 --run-tag t1
       (cluster) RUN_TAG=v001 k8s/ray/submit-decision-run.sh --num-envs 20 --stage lift
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_env import HandEnv, STAGES
from grasp_planner import GraspPlanner, MPCPlanner
from decision_client import DecisionClient
from perception import observe, trace_line
from reflex import joint_limit_guard, force_guard
from dagger import DaggerLog

FIXTURE_EVERY = int(os.environ.get("ROBOT_FIXTURE_EVERY", "100"))


def build_env_config(stage: str, max_steps: int, render: bool) -> Dict[str, Any]:
    return {"stage": stage, "max_steps": max_steps, "render": render}


def save_fixture(env, out_dir: str, tag: str, step: int, reason: str) -> Optional[str]:
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    snap, steps, stage, ok = env.get_state()
    path = os.path.join(out_dir, f"{tag}_step{step:05d}_{reason}.npz")
    np.savez(path, qpos=snap[0], qvel=snap[1], act=snap[2], time=snap[3], steps=steps, stage=stage, success_steps=ok)
    return path


def upload_outputs(fixture_dir: str, run_tag: str, worker: int) -> int:
    """Cluster runs: copy this worker's fixtures / video / DAgger log to MinIO (s3://$S3_BUCKET/<run_tag>/)."""
    ep, bucket = os.environ.get("S3_ENDPOINT_URL"), os.environ.get("S3_BUCKET")
    if not (ep and bucket and fixture_dir and os.path.isdir(fixture_dir)):
        return 0
    import boto3
    s3 = boto3.client("s3", endpoint_url=ep, aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"), region_name="us-east-1")
    n = 0
    for f in os.listdir(fixture_dir):
        if f.startswith(f"{run_tag}_w{worker:03d}") or f.startswith(f"dagger-{run_tag}_w{worker:03d}"):
            s3.upload_file(os.path.join(fixture_dir, f), bucket, f"{run_tag}/{f}"); n += 1
    return n


def run_episode(env_config: Dict[str, Any], endpoints: List[str], run_tag: str, worker: int = 0, plan_time_s: float = 60.0,
                fixture_dir: str = "", teacher: str = "mpc", seed: int = 0) -> Dict[str, Any]:
    env = HandEnv(env_config); planner = GraspPlanner(env); client = DecisionClient(endpoints, action_dim=env.action_space.shape[0])
    mpc = MPCPlanner(env, seed=seed + worker) if teacher == "mpc" else None   # v002: receding-horizon teacher
    run_tag_w = f"{run_tag}_w{worker:03d}"
    dagger = DaggerLog(fixture_dir, run_tag_w) if fixture_dir else None
    t_start = time.time(); stage_steps: Dict[str, int] = {}
    obs, info = env.reset(); plan: List[np.ndarray] = []; plan_block_until = -1; steps = 0; done = False
    frames: List[np.ndarray] = []                                 # --video: one frame per step, written at the end
    last_name = "none"; violations = 0; stages_done: List[str] = []
    while not done:
        perc = observe(env)                                       # (the student input; unused by the skeleton)
        if mpc is not None:
            # planner-first for every step (dd precedence): the MPC teacher acts; every action was simulated from the
            # true state first. Each step is a labelled demonstration for the student (DAgger / BC data).
            action = mpc.act(); last_name = "mpc"
            if dagger:
                dagger.add(obs, env.stage, action, np.zeros(0), steps)
        elif not plan and steps >= plan_block_until:
            seq = planner.plan(time_s=plan_time_s)
            if seq:
                plan = list(seq); save_fixture(env, fixture_dir, run_tag, steps, f"plan{planner.plans}-{env.stage}")
            else:
                plan_block_until = steps + 20                     # dd: do not burn another budget right away
        if mpc is not None:
            pass
        elif plan:
            action = plan.pop(0); last_name = "plan"
        else:
            d = client.choose_action(obs, info, env.render())
            action = d.action; last_name = d.source
            if d.source == "fallback":
                # dd lesson (v154r): a meaningless fallback between plans is what wrecked the model-less Double Dragon runs.
                # Zeros here would drive every joint to mid-range; hold the current targets instead.
                action = env.hold_action()
            for guard in (lambda a: joint_limit_guard(env, a), lambda a: force_guard(info, a, env.hold_action(), env.cfg["force_max"])):
                alt = guard(action)
                if alt is not None:
                    action = alt; last_name = "reflex"
        obs, reward, terminated, truncated, info = env.step(action); steps += 1
        if env.cfg["render"]:
            frames.append(env.render())
        if info["dropped"] or info["force_violation"]:
            violations += 1
            print(trace_line(steps, info, len(plan), last_name), flush=True)
            save_fixture(env, fixture_dir, run_tag_w, steps, "violation")
        if info["stage_done"] and env.stage not in stages_done:
            stages_done.append(env.stage); stage_steps[env.stage] = steps
            print(f"stage-done {env.stage} at step {steps} ({time.time() - t_start:.0f}s)", flush=True)
            save_fixture(env, fixture_dir, run_tag_w, steps, f"done-{env.stage}")
            nxt = STAGES.index(env.stage) + 1
            if nxt < len(STAGES):
                env.stage = STAGES[nxt]; env.success_steps = 0; plan = []
            else:
                done = True
        if steps % FIXTURE_EVERY == 0:
            save_fixture(env, fixture_dir, run_tag_w, steps, f"run-{env.stage}")
            if mpc is not None:
                print(f"step {steps} stage={env.stage} lift_dz={info['lift_dz']*100:.1f}cm ang={info['angle_err_deg']:.0f} "
                      f"depth={info['insert_depth']*100:.1f}cm lat={info['insert_latera'+'l']*1000:.1f}mm cost={mpc.last_cost:.1f}", flush=True)
        done = done or terminated or truncated
    if frames and fixture_dir:
        import imageio.v2 as imageio
        imageio.mimwrite(os.path.join(fixture_dir, f"{run_tag_w}.mp4"), frames, fps=env.cfg["control_hz"], codec="libx264", quality=8)
    result = {"run_tag": run_tag, "worker": worker, "teacher": teacher, "steps": steps, "stages_done": stages_done,
              "stage_steps": stage_steps, "violations": violations, "plans": (mpc.plans if mpc else planner.plans),
              "dagger_records": dagger.n if dagger else 0, "wall_s": round(time.time() - t_start, 1)}
    if fixture_dir:
        with open(os.path.join(fixture_dir, f"{run_tag_w}_summary.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    result["uploaded"] = upload_outputs(fixture_dir, run_tag, worker)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true"); ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--stage", default="lift", choices=STAGES); ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--endpoints", default=os.environ.get("STUDENT_ENDPOINTS", "http://127.0.0.1:18011"))
    ap.add_argument("--run-tag", default=time.strftime("%Y%m%d-%H%M%S")); ap.add_argument("--plan-time-s", type=float, default=60.0)
    ap.add_argument("--fixture-dir", default=os.environ.get("ROBOT_SAVE_LOCAL_DIR", ""))
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--video", action="store_true", help="write an mp4 per episode (local)")
    ap.add_argument("--teacher", default=os.environ.get("ROBOT_TEACHER", "mpc"), choices=("mpc", "cem"))
    args = ap.parse_args()
    cfg = build_env_config(args.stage, args.max_steps, render=args.video); eps = [e for e in args.endpoints.split(",") if e]
    if args.dry_run:
        print(json.dumps({"env_config": cfg, "endpoints": eps, "num_envs": args.num_envs, "run_tag": args.run_tag}, indent=2)); return
    if args.local:
        print(json.dumps(run_episode(cfg, eps, args.run_tag, plan_time_s=args.plan_time_s, fixture_dir=args.fixture_dir,
                                     teacher=args.teacher), indent=2)); return
    import ray
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    remote = ray.remote(num_cpus=1)(run_episode)
    results = ray.get([remote.remote(cfg, eps, args.run_tag, w, args.plan_time_s, args.fixture_dir, args.teacher)
                       for w in range(args.num_envs)])
    print("=== decision-model run summary ===")
    full = sum(1 for r in results if len(r["stages_done"]) == len(STAGES))
    print(json.dumps({"episodes": len(results), "full_task_success": full, "stages_done": [r["stages_done"] for r in results],
                      "stage_steps": [r["stage_steps"] for r in results], "violations": sum(r["violations"] for r in results),
                      "uploaded_files": sum(r.get("uploaded", 0) for r in results)}, indent=2))


if __name__ == "__main__":
    main()
