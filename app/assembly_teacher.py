"""Privileged teacher for the assembly cell (v003) - the robot analogue of the Double Dragon planners.

`act(env)` returns the 5-D action for the CURRENT state from privileged readings only (exact poses, contact flags,
the stage counter) and keeps no hidden memory, so it can label any state a student visits - the DAgger requirement.
Each stage is a small closed-loop controller (approach above, align, descend, act, retreat); everything it does is
simulated by the same MuJoCo model the student trains in.
"""
from __future__ import annotations
import numpy as np
from assembly_env import AssemblyEnv, HOME_TCP, wrap

HOVER = 0.10          # approach height above a grasp / place point
CARRY_Z = 0.20        # carry height between stations


PREOPEN = -0.25       # ~5 cm finger gap: enough for the 3 cm part / 2.2 cm driver, clears the stopper gate


def _move(env: AssemblyEnv, r, goal_pos, goal_yaw=None, grip=PREOPEN, gain=1.0) -> np.ndarray:
    tcp, yaw = r["tcp"], r["tcp_yaw"]
    d = (np.asarray(goal_pos) - tcp) / env.cfg["max_dpos"] * gain
    if r["part_held"] or r["tool_held"]:
        n = np.linalg.norm(d)
        if n > 0.4:
            d = d * (0.4 / n)                                                  # carry speed cap (~12 cm/s): no slip
    dyaw = 0.0 if goal_yaw is None else wrap(goal_yaw - yaw) / env.cfg["max_dyaw"]
    return np.clip(np.array([d[0], d[1], d[2], dyaw, grip]), -1, 1).astype(np.float32)


def _grasp_yaw(r) -> float:
    """TCP yaw that puts the fingers across the part's short axis, picking the closer of the two equivalents."""
    y = r["part_yaw"]; cands = [y, wrap(y + np.pi)]
    return min(cands, key=lambda c: abs(wrap(c - r["tcp_yaw"])))


def _carrying_tool(r) -> bool:
    """Robust to one-step contact flicker: contact, or fingers closed with the driver right at the TCP."""
    return bool(r["tool_held"] or (r["finger_gap"] < 0.035 and np.linalg.norm(r["tool_pos"] - r["tcp"]) < 0.04))


def act(env: AssemblyEnv) -> np.ndarray:
    r = env._readings(); st = env.stage; tcp = r["tcp"]
    if st == "reach":
        p = r["part_pos"]; gy = _grasp_yaw(r)
        if r["on_belt"] and not r["on_stop"]:                                  # wait over the pick zone
            return _move(env, r, [p[0], -0.15, CARRY_Z], gy)
        above = np.array([p[0], p[1], p[2] + HOVER])
        if np.linalg.norm(tcp[:2] - p[:2]) > 0.008 or abs(wrap(r["tcp_yaw"] - gy, np.pi)) > np.radians(5):
            return _move(env, r, [p[0], p[1], p[2] + 0.06], gy)              # canonical approach height
        return _move(env, r, [p[0], p[1], p[2]], gy, gain=0.6)
    if st == "grasp":
        return _move(env, r, r["part_pos"], None, grip=1.0, gain=0.3)
    if st == "lift":
        return _move(env, r, [tcp[0], tcp[1], 0.07 + env.cfg["lift_height"] + 0.04], None, grip=1.0, gain=0.35)
    if st == "rotate":
        # rotate the part onto the pocket key (yaw 0 mod pi) IN PLACE first, then carry toward the housing
        target_yaw = r["tcp_yaw"] - wrap(r["part_yaw"], np.pi)
        pk = r["pocket"]
        if r["key_err_deg"] > 10.0:
            a = _move(env, r, tcp, target_yaw, grip=1.0); a[3] = np.clip(a[3], -0.5, 0.5); return a
        return _move(env, r, [pk[0], pk[1], CARRY_Z], target_yaw, grip=1.0, gain=0.5)
    if st == "place":
        pk = r["pocket"]; p = r["part_pos"]
        off = p - tcp                                                          # part position relative to the TCP
        key_yaw = r["tcp_yaw"] - wrap(r["part_yaw"], np.pi)
        if r["part_held"]:
            xy_err = np.linalg.norm(p[:2] - pk[:2]); goal = pk - off
            if xy_err > 0.0015:                                                 # align above the pocket first
                goal[2] = goal[2] + 0.025 if xy_err < 0.02 else CARRY_Z         # canonical: carry height far, 2.5 cm above near
                return _move(env, r, goal, key_yaw, grip=1.0, gain=0.5 if xy_err < 0.02 else 0.8)
            if p[2] - pk[2] > 0.002:
                goal[2] = tcp[2] - 0.004                                       # slow descent, keep correcting xy
                return _move(env, r, goal, key_yaw, grip=1.0, gain=1.0)
            return _move(env, r, tcp, None, grip=PREOPEN)                         # seated: release
        if r["finger_gap"] < 0.045:
            return _move(env, r, tcp, None, grip=PREOPEN)                         # finish opening before moving
        return _move(env, r, [tcp[0], tcp[1], r["pocket"][2] + 0.10], None, grip=PREOPEN)  # released: back off to a fixed height
    if st == "get_tool":
        g = env.data.site_xpos[env.sid["tool_grip"]].copy()
        R_t = env.data.xmat[env.bid["tool"]].reshape(3, 3)
        ty = float(np.arctan2(R_t[1, 0], R_t[0, 0]))                           # square handle: align to a face (mod 90)
        gy = min([ty + k * np.pi / 2 for k in range(-2, 3)], key=lambda c: abs(wrap(c - r["tcp_yaw"])))
        if not r["tool_held"]:
            near = np.linalg.norm(tcp[:2] - g[:2])
            if near > 0.003 or abs(wrap(r["tcp_yaw"] - gy)) > np.radians(3):
                hz = CARRY_Z if near > 0.03 else g[2] + 0.05                    # canonical transit / approach heights
                return _move(env, r, [g[0], g[1], hz], gy, grip=PREOPEN, gain=0.6 if near < 0.02 else 1.0)
            if tcp[2] - g[2] > 0.003:
                return _move(env, r, g, gy, grip=PREOPEN, gain=0.6)
            return _move(env, r, g, gy, grip=1.0, gain=0.3)
        return _move(env, r, [tcp[0], tcp[1], 0.20], None, grip=1.0, gain=0.4)   # lift the driver to a fixed height
    if st == "fasten":
        f, tip = r["fastener"], r["tool_tip"]
        goal = tcp + (f - tip)
        lat = r["tip_lateral"]
        if lat > 0.004:
            goal[2] = tcp[2] + (f[2] - tip[2]) + (0.03 if lat < 0.02 else 0.06)  # canonical hover above the fastener
            return _move(env, r, goal, None, grip=1.0, gain=0.4 if lat > 0.02 else 0.3)
        if r["tip_vertical"] > 0.003 and tip[2] > f[2]:
            goal[2] -= 0.001
            return _move(env, r, goal, None, grip=1.0, gain=0.5)
        a = _move(env, r, goal + [0, 0, -0.001], None, grip=1.0, gain=0.5)
        a[3] = 1.0                                                             # twist the driver at full rate
        return a
    if st == "finish":
        h = r["holder"]
        if _carrying_tool(r):
            tool_off = r["tool_pos"] - tcp
            hd = np.linalg.norm(r["tool_pos"][:2] - h[:2])
            if hd > 0.006:                                                     # carry over the holder at a safe height
                above = np.array([h[0], h[1], 0.22]) - tool_off * [1, 1, 0]
                return _move(env, r, above, None, grip=1.0, gain=0.4 if hd > 0.02 else 0.3)
            seat = np.array([h[0], h[1], 0.069 + 0.003]) - tool_off             # descend while still centring (no hard gate)
            if r["tool_pos"][2] > 0.072:
                if r["tool_pos"][2] < 0.12 and hd > 0.0035:
                    seat[2] = tcp[2]                                           # near the cup rim: centre before going lower
                return _move(env, r, seat, None, grip=1.0, gain=0.5)
            return _move(env, r, tcp, None, grip=PREOPEN)                      # seated: release
        if np.linalg.norm(tcp[:2] - h[:2]) < 0.06 and tcp[2] < 0.20:
            return _move(env, r, [tcp[0], tcp[1], 0.24], None, grip=PREOPEN)   # clear the tool vertically first
        return _move(env, r, HOME_TCP, None, grip=PREOPEN)
    return np.zeros(5, dtype=np.float32)
