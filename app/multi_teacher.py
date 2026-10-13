"""Privileged teacher for the multi-part cell (v006) - generalises assembly_teacher to any part / tool / joining method.

`act(env)` reads only the current privileged state (part identity, exact poses, contacts, plan position) and keeps no
hidden memory, so it can label any state a student visits (the DAgger requirement). One small closed-loop controller
per skill: inspect (wait clear of the line), reach, grasp, lift, rotate, place, get_tool, fasten (press + twist),
weld (trace the seam marks with the torch tip, dwelling on each), press (closed fingers push the cap home),
return_tool, finish.
"""
from __future__ import annotations
import numpy as np
from build_multi_cell import PARTS
from multi_env import MultiAssemblyEnv, HOME_TCP, wrap

CARRY_Z = 0.20
PREOPEN = -0.25
WAIT = np.array([0.60, -0.17, CARRY_Z])      # over the stop, downstream of the inspection gantry


def _move(env, r, goal, goal_yaw=None, grip=PREOPEN, gain=1.0) -> np.ndarray:
    tcp, yaw = r["tcp"], r["tcp_yaw"]
    d = (np.asarray(goal) - tcp) / env.cfg["max_dpos"] * gain
    if r["held"] or r["tool_held"]:
        n = np.linalg.norm(d)
        if n > 0.4:
            d = d * (0.4 / n)
    dyaw = 0.0 if goal_yaw is None else wrap(goal_yaw - yaw) / env.cfg["max_dyaw"]
    return np.clip(np.array([d[0], d[1], d[2], dyaw, grip]), -1, 1).astype(np.float32)


def _transit(env, r, goal, goal_yaw=None, grip=PREOPEN, gain=1.0) -> np.ndarray:
    """Move to `goal`, but retract straight up to transit height first when starting low and far away (an industrial
    arm program never drags open jaws sideways out of a fixture)."""
    tcp = r["tcp"]; goal = np.asarray(goal, float)
    if np.linalg.norm(goal[:2] - tcp[:2]) > 0.03:                              # far: travel at transit height
        if tcp[2] < CARRY_Z - 0.03:
            return _move(env, r, [tcp[0], tcp[1], CARRY_Z], None, grip=grip, gain=gain)
        return _move(env, r, [goal[0], goal[1], max(goal[2], CARRY_Z)], goal_yaw, grip=grip, gain=gain)
    return _move(env, r, goal, goal_yaw, grip=grip, gain=gain)


def _funnel(err: float, tol: float, height: float, slope: float = 3.0) -> float:
    """Hover height above a target as a smooth function of the remaining lateral error: 0 within `tol`, rising with
    `slope`, capped at `height`. Descend-while-centring, with no threshold switch for a student to dither across."""
    return float(np.clip(slope * (err - tol), 0.0, height))


def _grasp_yaw(r) -> float:
    step = MultiAssemblyEnv.grasp_period(r["part"])
    cands = [r["part_yaw"] + k * step for k in range(-8, 9)]
    if step < np.pi / 2:                                                       # cap: the pair of flats nearest "parallel to the stopper"
        return min(cands, key=lambda c: abs(wrap(c - np.pi / 2, np.pi)) + 0.01 * abs(wrap(c - r["tcp_yaw"])))
    return min(cands, key=lambda c: abs(wrap(c - r["tcp_yaw"])))


def _key_yaw(r) -> float:
    """TCP yaw that brings the held part onto its nest key."""
    sym = PARTS[r["part"]]["sym"]
    return r["tcp_yaw"] if sym == 0 else r["tcp_yaw"] - wrap(r["part_yaw"], sym)


def _carrying(r, t) -> bool:
    T = r["tools"][t]
    return bool(r["tool_held"] == t or (r["finger_gap"] < 0.035 and np.linalg.norm(T["pos"] - r["tcp"]) < 0.06))


def act(env: MultiAssemblyEnv) -> np.ndarray:
    r = env._readings(); st = env.stage; tcp = r["tcp"]
    if st == "inspect":
        return _transit(env, r, WAIT, None)
    if st == "reach":
        p = r["part_pos"]; gy = _grasp_yaw(r)
        if r["on_belt"] and not r["on_stop"]:
            return _transit(env, r, [p[0], -0.17, CARRY_Z], gy)
        xy = np.linalg.norm(tcp[:2] - p[:2]); ye = abs(wrap(r["tcp_yaw"] - gy, np.pi))
        hover = max(_funnel(xy, 0.002, CARRY_Z - p[2]), _funnel(ye, np.radians(2), 0.06, slope=0.15))
        # close as soon as the jaws are around the part: the close decision belongs to this pose, not to a 5-step dwell
        # (a dwell labels the same pose "open" then "close" across the stage boundary - unlearnable for the student)
        close = xy < 0.004 and ye < np.radians(5) and abs(tcp[2] - p[2]) < 0.004
        return _move(env, r, [p[0], p[1], p[2] + hover], gy, grip=1.0 if close else PREOPEN, gain=0.6)
    if st == "grasp":
        return _move(env, r, r["part_pos"], None, grip=1.0, gain=0.3)
    if st == "lift":
        return _move(env, r, [tcp[0], tcp[1], env.cfg["lift_z"] + 0.04], None, grip=1.0, gain=0.35)
    if st == "rotate":
        ky = _key_yaw(r); nest = r["nest"]
        if r["key_err_deg"] > 10.0:
            a = _move(env, r, tcp, ky, grip=1.0); a[3] = np.clip(a[3], -0.5, 0.5); return a
        return _move(env, r, [nest[0], nest[1], CARRY_Z], ky, grip=1.0, gain=0.5)
    if st == "place":
        nest = r["nest"]; p = r["part_pos"]; off = p - tcp; ky = _key_yaw(r)
        in_hand = r["held"] or (r["finger_gap"] < 0.04 and env.grip_cmd > 0 and np.linalg.norm(off) < 0.02)  # contact flicker
        if in_hand:
            # smooth funnel: hover height above the seat grows with the remaining xy error (2 mm -> 0, 2 cm -> 2.5 cm),
            # so the arm descends WHILE centring - one continuous map instead of hover/descend switches (easier to imitate)
            xy = np.linalg.norm(p[:2] - nest[:2]); goal = nest - off
            goal[2] += _funnel(xy, 0.0015, CARRY_Z - nest[2]) + 0.001          # ONE continuous map out to carry height
            if p[2] - nest[2] > 0.003 or xy > 0.0035:                         # release window = the env's seat tolerance
                return _move(env, r, goal, ky, grip=1.0, gain=0.35)
            if env.grip_cmd <= 0 and r["finger_gap"] < 0.03:                    # open commanded, jaws pinned: free them
                return _move(env, r, tcp + [0, 0, 0.004], None, grip=PREOPEN, gain=0.25)
            return _move(env, r, tcp, None, grip=PREOPEN)                         # seated and centred: release
        if r["finger_gap"] < 0.045:
            if env.grip_cmd <= 0 and r["finger_gap"] < 0.03 and r["in_nest"]:
                # jaws commanded open but pinned against a nest wall: back straight up slowly to free them
                return _move(env, r, tcp + [0, 0, 0.004], None, grip=PREOPEN, gain=0.25)
            return _move(env, r, tcp, None, grip=PREOPEN)
        return _move(env, r, [tcp[0], tcp[1], nest[2] + 0.10], None, grip=PREOPEN)
    if st == "get_tool":
        t = r["tool_need"]; T = r["tools"][t]; g = T["grip"]
        gy = min([T["yaw"] + k * np.pi / 2 for k in range(-4, 5)], key=lambda c: abs(wrap(c - r["tcp_yaw"])))
        if r["tool_held"] != t:
            near = np.linalg.norm(tcp[:2] - g[:2]); ye = abs(wrap(r["tcp_yaw"] - gy))
            hover = max(_funnel(near, 0.0015, max(CARRY_Z, g[2] + 0.05) - g[2]), _funnel(ye, np.radians(1.5), 0.05, slope=0.15))
            close = near < 0.003 and ye < np.radians(3) and tcp[2] - g[2] < 0.004    # usable window; funnel keeps centring
            return _move(env, r, [g[0], g[1], g[2] + hover], gy, grip=1.0 if close else PREOPEN, gain=0.5)
        return _move(env, r, [tcp[0], tcp[1], 0.22], None, grip=1.0, gain=0.4)
    if st == "fasten":
        f, tip = r["join_target"], r["tool_tip"]; goal = tcp + (f - tip); lat = r["tip_lateral"]
        hover = _funnel(lat, 0.002, 0.06)
        goal[2] += hover - 0.001
        a = _move(env, r, goal, None, grip=1.0, gain=0.4)
        if lat < 0.004 and r["tip_vertical"] < 0.005:                         # inside the env's engage tolerance (6 mm)
            a[3] = 1.0                                                         # engaged: twist the driver at full rate
        return a
    if st == "weld":
        s, tip = r["join_target"], r["tool_tip"]; goal = tcp + (s - tip) + [0, 0, 0.004]; lat = r["tip_lateral"]
        goal[2] += _funnel(lat, 0.004, 0.03)                                   # continuous: hover far, trace at weld height
        return _move(env, r, goal, None, grip=1.0, gain=0.35)
    if st == "press":
        top = r["join_target"]
        if r["finger_gap"] > 0.012 and tcp[2] < top[2] + 0.028:               # still around the cap: rise with the jaws open
            return _move(env, r, [tcp[0], tcp[1], top[2] + 0.035], None, grip=PREOPEN, gain=0.6)
        if r["finger_gap"] > 0.012:                                            # close the fingers above the cap first
            return _move(env, r, [top[0], top[1], max(tcp[2], top[2] + 0.03)], None, grip=1.0, gain=0.5)
        hover = _funnel(np.linalg.norm(tcp[:2] - top[:2]), 0.003, 0.03)
        if hover > 0.0:
            return _move(env, r, [top[0], top[1], top[2] + hover], None, grip=1.0, gain=0.4)
        # force-limited press: creep down until the fingertip force passes the threshold, then hold (memoryless: reads F)
        dz = 0.0 if r["press_force"] > env.cfg["press_force"] * 1.2 else -0.0015
        return _move(env, r, [top[0], top[1], tcp[2] + dz], None, grip=1.0, gain=1.0)
    if st == "return_tool":
        t = r["tool_need"]; T = r["tools"][t]; h = T["holder"]; seat_z = env.seat_z(t)
        if _carrying(r, t):
            off = T["pos"] - tcp; hd = np.linalg.norm(T["pos"][:2] - h[:2])
            seat = np.array([h[0], h[1], seat_z + 0.003]) - off
            if hd > 0.03:
                seat[2] = seat_z + 0.15 - off[2]
                return _move(env, r, seat, None, grip=1.0, gain=0.4)
            seat[2] += _funnel(hd, 0.002, 0.15, slope=6.0)                    # funnel into the cup
            if T["pos"][2] > seat_z + 0.003 or hd > 0.0035:
                return _move(env, r, seat, None, grip=1.0, gain=0.4)
            return _move(env, r, tcp, None, grip=PREOPEN)
        if r["finger_gap"] < 0.045:
            return _move(env, r, tcp, None, grip=PREOPEN)
        return _move(env, r, [tcp[0], tcp[1], T["pos"][2] + 0.16], None, grip=PREOPEN)
    if st == "finish":
        return _transit(env, r, HOME_TCP, None)
    return np.zeros(5, np.float32)
