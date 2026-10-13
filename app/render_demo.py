"""Render a short clip of the Allegro Hand in MuJoCo - the video path of this template.

Two modes:
  * default: the bare Menagerie hand doing a finger wave (the original v000a clip -> allegro_demo.mp4)
  * --cell:  the assembly cell (app/ignored/assets/allegro_cell.xml, `make assets`): open -> thumb opposition ->
             wrap grasp on the part -> hold -> release (-> allegro_cell_demo.mp4). Scripted motion only; it shows
             the scene, the physics contacts and the render pipeline, not a learned or planned policy.
Usage: app/venv/bin/python app/render_demo.py [--cell] [out_dir] [seconds]"""
import os, sys, numpy as np, mujoco, imageio.v2 as imageio

here = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
cell = "--cell" in sys.argv
xml = (os.path.join(here, "ignored", "assets", "allegro_cell.xml") if cell
       else os.path.join(here, "ignored", "menagerie", "wonik_allegro", "scene_right.xml"))
out_dir = args[0] if len(args) > 0 else os.path.join(here, "ignored", "videos")
seconds = float(args[1]) if len(args) > 1 else 8.0
os.makedirs(out_dir, exist_ok=True)

model = mujoco.MjModel.from_xml_path(xml); data = mujoco.MjData(model)
fps = 30; substeps = max(1, int(round(1.0 / (model.opt.timestep * fps))))
renderer = mujoco.Renderer(model, height=480, width=640)
lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
print(f"model: {model.nu} actuators, {model.nbody} bodies; {seconds}s at {fps} fps")


def cell_targets(t: float) -> np.ndarray:
    """Open (0-1 s) -> thumb opposition (1-2.5 s) -> wrap grasp (2.5-4 s) -> hold -> release (last 1.5 s)."""
    open_ = np.clip(np.zeros(model.nu), lo, hi)
    thumb = open_.copy(); thumb[12] = 1.0; thumb[13] = lo[13] + 0.2 * (hi[13] - lo[13]); thumb[14] = thumb[15] = 1.0
    grasp = thumb.copy()
    for f in range(3):
        grasp[f * 4 + 1] = grasp[f * 4 + 2] = 1.6; grasp[f * 4 + 3] = 1.44
    grasp = np.clip(grasp, lo, hi); thumb = np.clip(thumb, lo, hi)
    def ramp(a, b, t0, t1):
        u = np.clip((t - t0) / (t1 - t0), 0, 1); u = u * u * (3 - 2 * u); return a + u * (b - a)
    if t < 1.0: return open_
    if t < 2.5: return ramp(open_, thumb, 1.0, 2.5)
    if t < seconds - 1.5: return ramp(thumb, grasp, 2.5, 4.0)
    return ramp(grasp, open_, seconds - 1.5, seconds - 0.3)


frames = []
cam = None
if not cell:
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 150, -25, 0.45; cam.lookat[:] = [0.0, 0.0, 0.08]
for k in range(int(seconds * fps)):
    t = k / fps
    if cell:
        data.ctrl[:] = cell_targets(t)
    else:
        phase = np.linspace(0, 2 * np.pi, model.nu, endpoint=False)
        data.ctrl[:] = lo + (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t + phase)) * (hi - lo)
    for _ in range(substeps):
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera=("wrist" if cell else cam)); frames.append(renderer.render().copy())

name = "allegro_cell_demo" if cell else "allegro_demo"
mp4 = os.path.join(out_dir, f"{name}.mp4"); imageio.mimwrite(mp4, frames, fps=fps, codec="libx264", quality=8)
for i in (0, len(frames) // 3, 2 * len(frames) // 3):
    imageio.imwrite(os.path.join(out_dir, f"{name}_frame_{i:03d}.png"), frames[i])
print("wrote", mp4, "and", len(frames), "frames")
