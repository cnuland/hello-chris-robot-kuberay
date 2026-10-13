"""The sprite-radar analogue: a structured, student-facing summary of the scene, plus the per-step trace line.

In Double Dragon the radar turned OAM sprites into {player, enemies, pits, stalactites}; here `observe()` turns the
simulator (or, on the real cell, ROS topics) into {joints, part pose, tip positions, contacts}. The student sees
this (and a camera); the planner scores on `info` directly."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np


def observe(env) -> Dict[str, Any]:
    d, m = env.data, env.model
    tips = {}
    for name in ("ff_tip", "mf_tip", "rf_tip", "th_tip"):             # TODO: site names from the Allegro MJCF
        try:
            import mujoco
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name); tips[name] = d.site_xpos[sid].copy()
        except Exception:
            pass
    pos, quat = env._part_pose()
    return {"q": d.qpos[: m.nu].copy(), "dq": d.qvel[: m.nu].copy(), "part_pos": pos, "part_quat": quat,
            "tips": tips, "ncon": int(d.ncon), "contact_force_max": env._contact_force_max()}


def trace_line(step: int, info: Dict[str, Any], plan_left: int, last_action: str, note: str = "") -> str:
    """One line whenever an invariant breaks (dd: `hp-drop step=... area=... hp 7->6 ... stal=[...]`).
    Print it with everything a reader needs to reproduce from the nearest fixture."""
    return (f"violation step={step} stage={info['stage']} drop={info['dropped']} force={info['contact_force_max']:.1f} "
            f"ang={info['angle_err_deg']:.1f} z={info['part_pos'][2]:.3f} plan_left={plan_left} last={last_action} {note}")
