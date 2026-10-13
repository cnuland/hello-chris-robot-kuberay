"""Local MPC teacher trial for one stage: reports progress per 20 steps, writes a video. (dev tool)"""
import sys, time, json, os, numpy as np, imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_env import HandEnv
from grasp_planner import MPCPlanner
stage = sys.argv[1] if len(sys.argv) > 1 else "lift"; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
out = sys.argv[4] if len(sys.argv) > 4 else "ignored/runs/mpc"
env = HandEnv({"stage": stage, "max_steps": 10**6, "render": True, **kw.get("env", {})}); env.reset()
mpc = MPCPlanner(env, **kw.get("mpc", {})); frames = []; best = {"lift_dz": -1, "angle": 999, "depth": -1}; t0 = time.time()
for k in range(steps):
    a = mpc.act(); _, _, term, _, info = env.step(a); frames.append(env.render())
    best = {"lift_dz": max(best["lift_dz"], info["lift_dz"]), "angle": min(best["angle"], info["angle_err_deg"]),
            "depth": max(best["depth"], info["insert_depth"])}
    if k % 20 == 19 or term:
        print(f"step {k+1}: lift_dz={info['lift_dz']*100:.1f}cm ang={info['angle_err_deg']:.0f} depth={info['insert_depth']*100:.1f}cm "
              f"lat={info['insert_lateral']*1000:.1f}mm ok={info['stage_ok']} cost={mpc.last_cost:.1f} drop={info['dropped']} "
              f"({(time.time()-t0)/(k+1):.2f}s/step)", flush=True)
    if term: print("TERMINATED", info["dropped"], info["force_violation"]); break
os.makedirs(out, exist_ok=True); p = os.path.join(out, f"mpc_{stage}.mp4"); imageio.mimwrite(p, frames, fps=20, codec="libx264", quality=8)
print("BEST", {k: round(v, 4) for k, v in best.items()}, "video", p)
