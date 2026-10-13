"""Run the privileged teacher for one episode; print stage transitions, write a video. (dev tool)"""
import sys, os, time, numpy as np, imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assembly_env import AssemblyEnv
import assembly_teacher as T
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0; out = sys.argv[2] if len(sys.argv) > 2 else "ignored/runs/teacher"
video = "--no-video" not in sys.argv
env = AssemblyEnv({"render": video}); obs, info = env.reset(seed=seed); frames = []; t0 = time.time(); ret = 0.0
print(f"seed {seed}: part yaw {np.degrees(info['part_yaw']):.0f} deg")
for k in range(env.cfg["max_steps"]):
    a = T.act(env); obs, rew, term, trunc, info = env.step(a); ret += rew
    if video and k % 2 == 0: frames.append(env.render())
    if info["advanced"]: print(f"  step {k+1:4d}  -> {info['stage']:9s} (return {ret:.1f})", flush=True)
    if term or trunc: break
print(f"END stage={info['stage']} steps={k+1} return={ret:.1f} dropped={info['dropped']} twist={info['twist']:.0f} weld={info['weld_on']} "
      f"key_err={info['key_err_deg']:.1f} wall={time.time()-t0:.0f}s")
if video:
    os.makedirs(out, exist_ok=True); p = os.path.join(out, f"teacher_s{seed}.mp4"); imageio.mimwrite(p, frames, fps=10, codec="libx264", quality=8); print("video", p)
