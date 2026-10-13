"""Panda assembly cell as a Gymnasium env (v003) - the robot task the distilled student has to learn.

Stages (each with a success test and a shaped reward - the "rooms" of this project):
  0 reach     TCP above the part's grasp point, yaw aligned with the part's short axis
  1 grasp     both fingers in contact with the part, gripper closed
  2 lift      part held >= 5 cm above the belt
  3 rotate    part yaw matches the housing pocket key (mod 180 deg) within 5 deg, still held
  4 place     part seated in the pocket (xy < 4 mm, z < 4 mm) and released
  5 get_tool  driver held and lifted out of its holder
  6 fasten    driver tip on the fastener, pressed, twisted >= 90 deg while engaged -> weld attaches part to housing
  7 finish    driver back in its holder, released, TCP home

Action (normalized, 5-D): [dx, dy, dz, dyaw, grip] -> Cartesian TCP target deltas (<= 1.5 cm, <= 8 deg per 50 ms
control step) solved to joint targets with damped least-squares IK (TCP kept pointing down); grip is binary
(> 0 closes, otherwise ~5 cm open).
`info` carries the privileged readings (the RAM map); `obs` is what the student sees.
"""
from __future__ import annotations
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
import mujoco

from sim_state import snapshot, restore

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("reach", "grasp", "lift", "rotate", "place", "get_tool", "fasten", "finish")
HOME_Q = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854])
HOME_TCP = np.array([0.45, 0.0, 0.32])
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_xml": os.path.join(HERE, "ignored", "assets", "assembly_cell.xml"),
    "control_hz": 20, "max_steps": 900,
    "max_dpos": 0.015, "max_dyaw": np.radians(8.0),
    "belt_speed": 0.10,
    "part_yaw_range": (30.0, 90.0),     # degrees off-key, random sign
    "lift_height": 0.05, "rotate_tol_deg": 5.0, "place_tol": 0.004,
    "fasten_twist_deg": 90.0, "fasten_tol": 0.006,
    "render": False, "camera": "overview", "width": 480, "height": 360,
}


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a: float, period: float = 2 * np.pi) -> float:
    return float((a + period / 2) % period - period / 2)


class AssemblyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        m = self.model = mujoco.MjModel.from_xml_path(self.cfg["model_xml"])
        self.data = mujoco.MjData(m)
        self.substeps = max(1, int(round(1.0 / (m.opt.timestep * self.cfg["control_hz"]))))
        nm = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, S, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_GEOM
        self.bid = {n: nm(B, n) for n in ("part", "tool", "hand", "left_finger", "right_finger", "housing")}
        self.sid = {n: nm(S, n) for n in ("tcp", "pocket", "holder", "fastener", "tool_tip", "tool_grip")}
        self.gid = {n: nm(G, n) for n in ("belt", "part", "tool_handle", "tool_tip", "fastener", "stopper")}
        self.stripes = [m.body_mocapid[nm(B, f"stripe{i}")] for i in range(10)]
        self.weld = nm(mujoco.mjtObj.mjOBJ_EQUALITY, "fasten")
        self.tc = nm(mujoco.mjtObj.mjOBJ_EQUALITY, "toolchanger")             # tool-changer lock (hand <-> driver)
        J = mujoco.mjtObj.mjOBJ_JOINT
        arm_j = [nm(J, f"joint{i}") for i in range(1, 8)]                       # the free joints of part/tool come FIRST in qpos
        self.arm_q = np.array([m.jnt_qposadr[j] for j in arm_j]); self.arm_dof = np.array([m.jnt_dofadr[j] for j in arm_j])
        self.arm_range = m.jnt_range[arm_j]
        self.finger_q = np.array([m.jnt_qposadr[nm(J, n)] for n in ("finger_joint1", "finger_joint2")])
        self.part_qadr = m.jnt_qposadr[nm(mujoco.mjtObj.mjOBJ_JOINT, "part_free")]
        self.part_dadr = m.jnt_dofadr[nm(mujoco.mjtObj.mjOBJ_JOINT, "part_free")]
        self.tool_qadr = m.jnt_qposadr[nm(mujoco.mjtObj.mjOBJ_JOINT, "tool_free")]
        self.finger_bodies = {self.bid["left_finger"], self.bid["right_finger"]}
        self.belt_y = (-0.62, -0.06)
        # gripper servo 10x stiffer: Menagerie's kp=100 N/m squeezes only ~3 N over a 1.5 cm closure (a real Panda grips
        # ~70 N) and the part slid out on every lift; 1000 N/m -> ~30 N, inside the +-100 N actuator force range
        m.actuator_gainprm[7, 0] *= 10.0; m.actuator_biasprm[7, 1:3] *= 10.0
        # grip physics: high friction + torsional friction (condim 4) on the finger pads and the held objects
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.finger_bodies and m.geom_contype[g]:
                m.geom_friction[g] = [2.0, 0.02, 0.001]; m.geom_condim[g] = 4
        for g in (self.gid["part"], self.gid["tool_handle"]):
            m.geom_friction[g] = [2.0, 0.02, 0.001]; m.geom_condim[g] = 4
        # driver bit spins freely in the fastener: frictionless tip contact (normal force only), otherwise the 90-degree
        # twist torques the square handle round inside the grip and the next move drops the driver
        m.geom_condim[self.gid["tool_tip"]] = 1
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self._renderer = mujoco.Renderer(m, self.cfg["height"], self.cfg["width"]) if self.cfg["render"] else None
        self.reset()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=self._obs().shape, dtype=np.float32)

    # ---- teacher-facing state -----------------------------------------------------------------------------------
    def get_state(self):
        return (snapshot(self.data), self.stage_i, self.steps, self.target.copy(), self.twist, self.weld_on,
                self.ok_steps, self.grip_cmd, self.tool_locked)

    def set_state(self, s) -> None:
        snap, self.stage_i, self.steps, tgt, self.twist, self.weld_on, self.ok_steps, self.grip_cmd, self.tool_locked = s
        self.target = tgt.copy(); restore(self.model, self.data, snap)
        self.data.eq_active[self.weld] = int(self.weld_on); self.data.eq_active[self.tc] = int(self.tool_locked)

    # ---- gym --------------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_q] = HOME_Q; d.qpos[self.finger_q] = 0.04
        d.ctrl[:7] = HOME_Q; d.ctrl[7] = 255.0
        lo, hi = self.cfg["part_yaw_range"]
        yaw = np.radians(self.rng.uniform(lo, hi) * self.rng.choice([-1, 1]))
        q = self.part_qadr
        d.qpos[q:q + 3] = [0.62 + self.rng.uniform(-0.01, 0.01), self.belt_y[0] + 0.08, 0.06 + 0.010 + 0.001]
        d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.eq_active[self.weld] = 0; d.eq_active[self.tc] = 0; self.tool_locked = False
        mujoco.mj_forward(m, d)
        self.stage_i, self.steps, self.twist, self.weld_on, self.ok_steps, self.grip_cmd = 0, 0, 0.0, False, 0, -1.0
        self.target = np.array([*d.site_xpos[self.sid["tcp"]], yaw_of(d.site_xmat[self.sid["tcp"]].reshape(3, 3))])
        self.prev_progress = self._stage_progress(self._readings())
        for _ in range(10):                                                   # settle the arm + belt for 0.5 s
            self._physics(self.target, -1.0)
        return self._obs(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if self._held("part") or self._held("tool"):
            # payload safety layer (like a cobot controller's velocity limit under load): any policy carrying an object
            # is capped at 40 % translation speed and 50 % wrist rate, so a slightly-off student cannot shake it loose
            n = np.linalg.norm(a[:3])
            if n > 0.4:
                a[:3] *= 0.4 / n
            a[3] = np.clip(a[3], -0.5, 0.5) if self.stage != "fasten" else a[3]
        self.target[:3] += a[:3] * self.cfg["max_dpos"]
        self.target[3] = wrap(self.target[3] + a[3] * self.cfg["max_dyaw"])
        self.target[:3] = np.clip(self.target[:3], [0.20, -0.55, 0.075], [0.80, 0.55, 0.55])
        self.grip_cmd = float(a[4])
        self._physics(self.target, self.grip_cmd)
        self._tool_changer()
        self.steps += 1
        r = self._readings()
        self._fasten_logic(r)
        r = self._readings()
        # recovery: losing the object of the current stage regresses the stage machine (the "recover" skill)
        regressed = False
        if self.stage in ("lift", "rotate") and not r["part_held"] and self.ok_steps == 0:
            self._lost = getattr(self, "_lost", 0) + 1
            if self._lost >= 6:
                self.stage_i = STAGES.index("reach"); regressed = True
        elif self.stage == "place" and not r["part_held"] and not r["part_in_pocket"] and r["finger_gap"] > 0.07 \
                and np.linalg.norm(r["part_pos"][:2] - r["pocket"][:2]) > 0.01:
            self._lost = getattr(self, "_lost", 0) + 1                    # released off the pocket: pick it up again
            if self._lost >= 10:
                self.stage_i = STAGES.index("reach"); regressed = True
        elif self.stage == "fasten" and not r["tool_held"] and not self.weld_on:
            self._lost = getattr(self, "_lost", 0) + 1
            if self._lost >= 6:
                self.stage_i = STAGES.index("get_tool"); regressed = True
        else:
            self._lost = 0
        if regressed:
            self._lost = 0; self.prev_progress = self._stage_progress(r)
        # stage machine with a 0.25 s hold requirement (dd: a win is checked after the plan)
        ok = self._stage_ok(r)
        self.ok_steps = self.ok_steps + 1 if ok else 0
        reward = 0.0
        prog = self._stage_progress(r)
        reward += 10.0 * (prog - self.prev_progress)                           # dense shaping toward the stage goal
        advanced = False
        if self.ok_steps >= 5 and self.stage_i < len(STAGES):
            reward += 10.0                                                     # stage completion bonus
            self.stage_i += 1; self.ok_steps = 0; advanced = True
            prog = self._stage_progress(r) if self.stage_i < len(STAGES) else 0.0
        self.prev_progress = prog
        dropped = bool(r["part_pos"][2] < 0.03 or r["tool_pos"][2] < -0.02 or r["tool_pos"][2] < 0.03 and not r["tool_in_holder"]
                       and np.linalg.norm(r["tool_pos"][:2] - r["holder"][:2]) > 0.03)
        reward -= self.cfg.get("time_penalty", 0.01)                           # time penalty (cycle-time pressure)
        if dropped:
            reward -= 10.0
        terminated = bool(dropped or self.stage_i >= len(STAGES))
        truncated = self.steps >= self.cfg["max_steps"]
        if regressed:
            reward -= 5.0
        info = self._info(r); info.update(advanced=advanced, regressed=regressed, dropped=dropped, reward=reward)
        return self._obs(r), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera=self.cfg["camera"]); return self._renderer.render()

    @property
    def stage(self) -> str:
        return STAGES[min(self.stage_i, len(STAGES) - 1)] if self.stage_i < len(STAGES) else "done"

    # ---- low level ---------------------------------------------------------------------------------------------
    def _physics(self, target: np.ndarray, grip: float) -> None:
        m, d = self.model, self.data
        d.ctrl[:7] = self._ik(target)
        # binary gripper (like an industrial gripper's close/open I/O): > 0 closes with full force, otherwise opens to the
        # ~5 cm pre-grasp width. A continuous mapping let a student's in-between output loosen a held part.
        d.ctrl[7] = 0.0 if grip > 0 else 160.0
        for _ in range(self.substeps):
            self._conveyor()
            mujoco.mj_step(m, d)
        t = d.time * self.cfg["belt_speed"]; L = self.belt_y[1] - self.belt_y[0]
        for i, mid in enumerate(self.stripes):                                 # animate the belt stripes
            y = self.belt_y[0] + ((i + 0.5) * L / 10 + t) % L
            d.mocap_pos[mid][1] = y

    def _ik(self, target: np.ndarray) -> np.ndarray:
        m, d = self.model, self.data
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.sid["tcp"])
        J = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
        pos = d.site_xpos[self.sid["tcp"]]; R = d.site_xmat[self.sid["tcp"]].reshape(3, 3)
        c, s = np.cos(target[3]), np.sin(target[3])
        Rt = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]])                     # TCP z down, yaw = target
        Re = Rt @ R.T
        qe = np.zeros(4); mujoco.mju_mat2Quat(qe, Re.flatten())
        rot_err = 2.0 * qe[1:] * np.sign(qe[0]) if abs(qe[0]) > 1e-9 else np.pi * qe[1:]
        err = np.concatenate([np.clip(target[:3] - pos, -0.05, 0.05), np.clip(rot_err, -0.5, 0.5)])
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        q = d.qpos[self.arm_q]
        null = (HOME_Q - q) * 0.05; null[6] = 0.0                              # posture bias (not on the wrist yaw)
        dq += (np.eye(7) - np.linalg.pinv(J) @ J) @ null
        return np.clip(q + dq, self.arm_range[:, 0], self.arm_range[:, 1])

    def _conveyor(self) -> None:
        """Belt drives the part toward the stopper while it rests on the belt (and is not held)."""
        d = self.data; pb = self.bid["part"]
        d.xfrc_applied[pb, :] = 0
        on_belt = any({d.contact[i].geom1, d.contact[i].geom2} >= {self.gid["belt"], self.gid["part"]} for i in range(d.ncon))
        if on_belt and not self._held("part"):
            # kinematic belt: the surface carries the part at belt speed (a static geom's friction would hold it);
            # the stopper's contact still stops it at the gate
            stopped = d.xpos[pb][1] > -0.12 - 0.031
            d.qvel[self.part_dadr + 1] = 0.0 if stopped else self.cfg["belt_speed"]
            d.qvel[self.part_dadr] *= 0.5

    def _tool_changer(self) -> None:
        """Tool-changer lock: fingers closed on the driver's coupling -> weld the driver to the hand at its current
        relative pose; opening the gripper releases it (real cells use a locking tool changer for exactly this)."""
        m, d = self.model, self.data
        if not self.tool_locked and self.grip_cmd > 0 and self.finger_bodies <= self._contacts("tool"):
            hp, hq = d.xpos[self.bid["hand"]], d.xquat[self.bid["hand"]]
            tp, tq = d.xpos[self.bid["tool"]], d.xquat[self.bid["tool"]]
            hqi = np.zeros(4); mujoco.mju_negQuat(hqi, hq)
            rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, tp - hp, hqi); rq = np.zeros(4); mujoco.mju_mulQuat(rq, hqi, tq)
            m.eq_data[self.tc, 3:6] = rp; m.eq_data[self.tc, 6:10] = rq
            d.eq_active[self.tc] = 1; self.tool_locked = True
        elif self.tool_locked and self.grip_cmd <= 0:
            d.eq_active[self.tc] = 0; self.tool_locked = False

    def _contacts(self, body: str) -> set:
        d, m = self.data, self.model; b = self.bid[body]; out = set()
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if b1 == b: out.add(b2)
            elif b2 == b: out.add(b1)
        return out

    def _held(self, body: str) -> bool:
        return self.finger_bodies <= self._contacts(body)

    def _fasten_logic(self, r: Dict[str, Any]) -> None:
        """Twist-lock: driver tip on the fastener + pressed (tool held, tip within tol) -> accumulate the twist;
        >= fasten_twist_deg -> activate the weld at the part's CURRENT pose relative to the housing."""
        if self.weld_on or self.stage != "fasten":
            return
        engaged = r["tool_held"] and r["tip_lateral"] < self.cfg["fasten_tol"] and r["tip_vertical"] < self.cfg["fasten_tol"] \
            and r["part_in_pocket"]
        dyaw = wrap(r["tcp_yaw"] - getattr(self, "_last_yaw", r["tcp_yaw"]))
        self._last_yaw = r["tcp_yaw"]
        if engaged:
            self.twist += abs(np.degrees(dyaw))
        if self.twist >= self.cfg["fasten_twist_deg"]:
            m, d = self.model, self.data
            hp, hq = d.xpos[self.bid["housing"]], d.xquat[self.bid["housing"]]
            pp, pq = d.xpos[self.bid["part"]], d.xquat[self.bid["part"]]
            hqi = np.zeros(4); mujoco.mju_negQuat(hqi, hq)
            rel_p = np.zeros(3); mujoco.mju_rotVecQuat(rel_p, pp - hp, hqi)
            rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, hqi, pq)
            m.eq_data[self.weld, 3:6] = rel_p; m.eq_data[self.weld, 6:10] = rel_q
            d.eq_active[self.weld] = 1; self.weld_on = True
            m.geom_rgba[self.gid["fastener"]] = [0.1, 0.8, 0.2, 1.0]                # fastener turns green when driven

    # ---- readers (privileged) -----------------------------------------------------------------------------------
    def _readings(self) -> Dict[str, Any]:
        d = self.data
        R_p = d.xmat[self.bid["part"]].reshape(3, 3)
        part_pos = d.xpos[self.bid["part"]].copy(); tcp = d.site_xpos[self.sid["tcp"]].copy()
        pocket = d.site_xpos[self.sid["pocket"]].copy(); holder = d.site_xpos[self.sid["holder"]].copy()
        tool_pos = d.xpos[self.bid["tool"]].copy(); tip = d.site_xpos[self.sid["tool_tip"]].copy()
        fast = d.site_xpos[self.sid["fastener"]].copy()
        part_yaw = yaw_of(R_p)
        key_err = abs(np.degrees(wrap(part_yaw - 0.0, np.pi)))                    # pocket key yaw = 0, part symmetric 180
        in_pocket = bool(np.linalg.norm(part_pos[:2] - pocket[:2]) < self.cfg["place_tol"]
                         and abs(part_pos[2] - pocket[2]) < self.cfg["place_tol"] and key_err < 8.0)
        tool_upright = float(d.xmat[self.bid["tool"]].reshape(3, 3)[2, 2])
        tool_in_holder = bool(np.linalg.norm(tool_pos[:2] - holder[:2]) < 0.025 and tool_pos[2] < 0.09 and tool_upright > 0.90)
        return {"tcp": tcp, "tcp_yaw": yaw_of(d.site_xmat[self.sid["tcp"]].reshape(3, 3)),
                "part_pos": part_pos, "part_yaw": part_yaw, "key_err_deg": key_err, "pocket": pocket,
                "part_held": self._held("part"), "part_in_pocket": in_pocket,
                "tool_pos": tool_pos, "tool_tip": tip, "tool_held": bool(self.tool_locked or self._held("tool")), "tool_in_holder": tool_in_holder,
                "holder": holder, "fastener": fast, "tip_lateral": float(np.linalg.norm(tip[:2] - fast[:2])),
                "tip_vertical": float(abs(tip[2] - fast[2])), "twist": self.twist, "weld_on": self.weld_on,
                "finger_gap": float(d.qpos[self.finger_q].sum()), "on_stop": bool(part_pos[1] > -0.12 - 0.03 - 0.016),
                "on_belt": bool(abs(part_pos[0] - 0.62) < 0.07 and -0.63 < part_pos[1] < -0.05 and part_pos[2] < 0.085)}

    def _stage_ok(self, r: Dict[str, Any]) -> bool:
        st = self.stage
        if st == "reach":
            g = r["part_pos"] + [0, 0, 0.0]
            return bool(np.linalg.norm(r["tcp"][:2] - g[:2]) < 0.015 and abs(r["tcp"][2] - g[2]) < 0.015
                        and abs(np.degrees(wrap(r["tcp_yaw"] - r["part_yaw"], np.pi))) < 10.0)
        if st == "grasp":
            return r["part_held"] and self.grip_cmd > 0
        if st == "lift":
            return r["part_held"] and r["part_pos"][2] > 0.07 + self.cfg["lift_height"]
        if st == "rotate":
            return r["part_held"] and r["key_err_deg"] < self.cfg["rotate_tol_deg"]
        if st == "place":
            return r["part_in_pocket"] and not r["part_held"]
        if st == "get_tool":
            return r["tool_held"] and r["tool_pos"][2] > 0.13
        if st == "fasten":
            return r["weld_on"]
        if st == "finish":
            return r["tool_in_holder"] and not r["tool_held"] and np.linalg.norm(r["tcp"] - HOME_TCP) < 0.04
        return False

    def _stage_progress(self, r: Dict[str, Any]) -> float:
        """Dense shaping in [-inf, 0]: minus the distance to this stage's goal."""
        st = self.stage
        if st == "reach":
            return -float(np.linalg.norm(r["tcp"] - r["part_pos"])) - 0.002 * abs(np.degrees(wrap(r["tcp_yaw"] - r["part_yaw"], np.pi)))
        if st == "grasp":
            return -0.1 * (0.0 if r["part_held"] else 1.0)
        if st == "lift":
            return -max(0.0, 0.07 + self.cfg["lift_height"] + 0.01 - r["part_pos"][2]) - (0.0 if r["part_held"] else 0.1)
        if st == "rotate":
            return -0.002 * r["key_err_deg"] - (0.0 if r["part_held"] else 0.1)
        if st == "place":
            return -float(np.linalg.norm(r["part_pos"] - r["pocket"])) - (0.05 if r["part_held"] else 0.0) * float(r["part_in_pocket"])
        if st == "get_tool":
            return -float(np.linalg.norm(r["tcp"] - (r["tool_pos"] + [0, 0, 0.03]))) * (0.0 if r["tool_held"] else 1.0) \
                - max(0.0, 0.14 - r["tool_pos"][2])
        if st == "fasten":
            return -r["tip_lateral"] - r["tip_vertical"] - 0.001 * max(0.0, self.cfg["fasten_twist_deg"] - r["twist"])
        if st == "finish":
            seat = np.array([r["holder"][0], r["holder"][1], 0.069])                # driver seated upright in its cup
            if not r["tool_in_holder"]:
                return -float(np.linalg.norm(r["tool_pos"] - seat)) - 0.5          # 3D: lowering the driver pays too
            if r["tool_held"]:
                return -0.25                                                       # seated: releasing pays next
            return -0.5 * float(np.linalg.norm(r["tcp"] - HOME_TCP))
        return 0.0

    def _info(self, r: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = r or self._readings()
        return {**r, "stage": self.stage, "stage_i": self.stage_i, "steps": self.steps}

    def _obs(self, r: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Student observation: proprioception + object poses relative to the TCP + task flags (no stage label)."""
        r = r or self._readings()
        tcp = r["tcp"]; c, s = np.cos(r["part_yaw"]), np.sin(r["part_yaw"])
        dy = wrap(r["tcp_yaw"] - r["part_yaw"], np.pi)
        return np.concatenate([
            tcp, [np.cos(r["tcp_yaw"]), np.sin(r["tcp_yaw"]), r["finger_gap"], self.grip_cmd],
            r["part_pos"] - tcp, [np.cos(2 * r["part_yaw"]), np.sin(2 * r["part_yaw"]), np.sin(2 * dy), np.cos(2 * dy)],
            r["pocket"] - r["part_pos"], r["pocket"] - tcp,
            r["tool_pos"] - tcp, r["tool_tip"] - r["fastener"], r["holder"] - r["tool_pos"], HOME_TCP - tcp,
            [float(r["part_held"]), float(r["part_in_pocket"]), float(r["tool_held"]), float(r["tool_in_holder"]),
             float(r["weld_on"]), min(r["twist"], 120.0) / 90.0, float(r["on_stop"])],
        ]).astype(np.float32)
