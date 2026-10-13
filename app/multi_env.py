"""Multi-part assembly cell as a Gymnasium env (v006) - five (phase 1: three) different parts, two tools, three
joining methods, parts arriving in random order.

Per part the plan is
    inspect -> reach -> grasp -> lift -> rotate -> place -> <join>
with <join> = get_tool(driver) -> fasten -> return_tool   (screw parts: bracket, plate)
              get_tool(torch)  -> weld   -> return_tool   (weld parts: lbracket, rail)
              press                                       (press-fit parts: cap)
and one final `finish` (arm home). Every stage has a success test, dense shaping and a +10 completion bonus; the plan
for an episode is the concatenation over the (random) part order - 26 stages for the 3-part cell, 44 for 5 parts.

Perception: the part is scanned as it passes the inspection gantry upstream of the stop. Two sensors see it:
  * the RGB camera `inspect` (render_inspect()) - what DiffusionGemma's vision reads to name the part;
  * a 16x16 height scanner (MuJoCo rays that only see part geoms) - the cheap sensor the student's part-ID head learns
    from, distilled from DiffusionGemma's vision labels.
The student never gets the part identity directly: `obs` holds geometry relative to every nest/tool plus the scan;
`info` carries the privileged readings the teacher uses.

Action (5-D, normalized) as in v003: [dx, dy, dz, dyaw, grip] Cartesian deltas -> damped-least-squares IK; binary grip.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import numpy as np
import gymnasium as gym
import mujoco

from build_multi_cell import PARTS, TOOLS, BELT, STOP_Y, SCAN_Y, FLOOR_TOP, half_height
from sim_state import snapshot, restore

HERE = os.path.dirname(os.path.abspath(__file__))
HOME_Q = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854])
HOME_TCP = np.array([0.45, 0.0, 0.32])
JOIN_TOOL = {"screw": "driver", "weld": "torch", "press": None}
SKILLS = ("inspect", "reach", "grasp", "lift", "rotate", "place", "get_tool", "fasten", "weld", "press", "return_tool", "finish")
SCAN_N, SCAN_W, SCAN_HI = 16, 0.12, 48          # student scan 16x16 (7.5 mm); station measures at 48x48 (2.5 mm)
DEFAULT_CONFIG: Dict[str, Any] = {
    "parts": ("bracket", "cap", "lbracket"), "order": "random",
    "control_hz": 20, "max_steps": 2600, "max_dpos": 0.015, "max_dyaw": np.radians(8.0), "belt_speed": 0.10,
    "part_yaw_range": (30.0, 90.0), "lift_z": 0.12, "rotate_tol_deg": 5.0, "place_tol": 0.004,
    "fasten_twist_deg": 90.0, "fasten_tol": 0.006, "weld_tol": 0.006, "weld_gap": 0.010, "weld_dwell": 3,
    "press_force": 10.0, "press_steps": 5, "time_penalty": 0.01,
    "render": False, "camera": "overview", "width": 480, "height": 360,
}


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a: float, period: float = 2 * np.pi) -> float:
    return float((a + period / 2) % period - period / 2)


def build_plan(order: List[str]) -> List[tuple]:
    plan = []
    for p in order:
        plan += [(p, s) for s in ("inspect", "reach", "grasp", "lift", "rotate", "place")]
        j = PARTS[p]["join"]
        plan += [(p, s) for s in (("get_tool", "fasten" if j == "screw" else "weld", "return_tool") if j != "press" else ("press",))]
    return plan + [("", "finish")]


class MultiAssemblyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.parts = list(self.cfg["parts"])
        xml = self.cfg.get("model_xml") or os.path.join(HERE, "ignored", "assets", f"multi_cell_{len(self.parts)}.xml")
        m = self.model = mujoco.MjModel.from_xml_path(xml); self.data = mujoco.MjData(m)
        self.substeps = max(1, int(round(1.0 / (m.opt.timestep * self.cfg["control_hz"]))))
        nm = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, S, G, J, E = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_GEOM,
                         mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_EQUALITY)
        self.bid = {n: nm(B, n) for n in self.parts + list(TOOLS) + ["hand", "left_finger", "right_finger", "chassis"]}
        self.sid = {"tcp": nm(S, "tcp")}
        for p in self.parts:
            self.sid[f"nest_{p}"] = nm(S, f"nest_{p}")
            for i in range(len(PARTS[p].get("fasteners", []))):
                self.sid[f"{p}_fast{i}"] = nm(S, f"{p}_fast{i}")
            for i in range(len(PARTS[p].get("seam", []))):
                self.sid[f"{p}_seam{i}"] = nm(S, f"{p}_seam{i}")
        for t in TOOLS:
            for s in ("tip", "grip"):
                self.sid[f"{t}_{s}"] = nm(S, f"{t}_{s}")
            self.sid[f"holder_{t}"] = nm(S, f"holder_{t}")
        self.belt_gid = nm(G, "belt")
        self.part_geoms = {p: {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.bid[p] and m.geom_contype[g]} for p in self.parts}
        self.deco = {p: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.bid[p] and not m.geom_contype[g]] for p in self.parts}
        self.deco_rgba0 = {g: m.geom_rgba[g].copy() for p in self.parts for g in self.deco[p]}
        self.stripes = [m.body_mocapid[nm(B, f"stripe{i}")] for i in range(10)]
        self.join_eq = {p: nm(E, f"join_{p}") for p in self.parts}
        self.tc_eq = {t: nm(E, f"tc_{t}") for t in TOOLS}
        arm_j = [nm(J, f"joint{i}") for i in range(1, 8)]
        self.arm_q = np.array([m.jnt_qposadr[j] for j in arm_j]); self.arm_dof = np.array([m.jnt_dofadr[j] for j in arm_j])
        self.arm_range = m.jnt_range[arm_j]
        self.finger_q = np.array([m.jnt_qposadr[nm(J, n)] for n in ("finger_joint1", "finger_joint2")])
        self.qadr = {o: m.jnt_qposadr[nm(J, f"{o}_free")] for o in self.parts + list(TOOLS)}
        self.dadr = {o: m.jnt_dofadr[nm(J, f"{o}_free")] for o in self.parts + list(TOOLS)}
        self.finger_bodies = {self.bid["left_finger"], self.bid["right_finger"]}
        self.belt_y = (BELT["y0"], BELT["y1"])
        m.actuator_gainprm[7, 0] *= 10.0; m.actuator_biasprm[7, 1:3] *= 10.0       # ~30 N grip (see assembly_env)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.finger_bodies and m.geom_contype[g]:
                m.geom_friction[g] = [2.0, 0.02, 0.001]; m.geom_condim[g] = 4
        self.park = {p: m.body_pos[self.bid[p]].copy() for p in self.parts}
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self._renderer = mujoco.Renderer(m, self.cfg["height"], self.cfg["width"]) if self.cfg["render"] else None
        self._insp_renderer = None
        self.reset()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=self._obs().shape, dtype=np.float32)

    # ---- state for the teacher / DAgger ------------------------------------------------------------------------
    def get_state(self):
        keys = ("order", "plan", "stage_i", "steps", "target", "ok_steps", "grip_cmd", "locked", "joined", "fed", "twist",
                "fast_done", "seam_done", "dwell", "press_n", "scans", "scans_hi", "scan_part", "_lost", "_last_yaw")
        return (snapshot(self.data), {k: (v.copy() if isinstance(v, (np.ndarray, dict, list, set)) else v)
                                       for k, v in ((k, getattr(self, k)) for k in keys)})

    def set_state(self, s) -> None:
        snap, st = s
        for k, v in st.items():
            setattr(self, k, v.copy() if isinstance(v, (np.ndarray, dict, list, set)) else v)
        restore(self.model, self.data, snap)
        for p in self.parts:
            self.data.eq_active[self.join_eq[p]] = int(p in self.joined)
        for t in TOOLS:
            self.data.eq_active[self.tc_eq[t]] = int(self.locked[t])

    # ---- gym ---------------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_q] = HOME_Q; d.qpos[self.finger_q] = 0.04; d.ctrl[:7] = HOME_Q; d.ctrl[7] = 255.0
        for g, c in self.deco_rgba0.items():
            m.geom_rgba[g] = c
        self.order = list(self.parts)
        if self.cfg["order"] == "random":
            self.rng.shuffle(self.order)
        self.plan = build_plan(self.order)
        self.stage_i, self.steps, self.ok_steps, self.grip_cmd = 0, 0, 0, -1.0
        self.locked = {t: False for t in TOOLS}; self.joined = set(); self.fed = []
        self.twist = 0.0; self.fast_done = {p: 0 for p in self.parts}; self.seam_done = {p: 0 for p in self.parts}
        self.dwell = 0; self.press_n = 0; self.scans = {}; self.scans_hi = {}; self.scan_part = None
        self._lost = 0; self._last_yaw = None
        for e in list(self.join_eq.values()) + list(self.tc_eq.values()):
            d.eq_active[e] = 0
        mujoco.mj_forward(m, d)
        self._feed(self.order[0])
        self.target = np.array([*d.site_xpos[self.sid["tcp"]], yaw_of(d.site_xmat[self.sid["tcp"]].reshape(3, 3))])
        for _ in range(10):
            self._physics(self.target, -1.0)
        self.prev_progress = self._stage_progress(self._readings())
        return self._obs(), self._info()

    def _feed(self, p: str) -> None:
        """Put part p on the belt start at a random off-key yaw."""
        d = self.data; lo, hi = self.cfg["part_yaw_range"]
        yaw = np.radians(self.rng.uniform(lo, hi) * self.rng.choice([-1, 1]))
        q, v = self.qadr[p], self.dadr[p]
        d.qpos[q:q + 3] = [BELT["x"] + self.rng.uniform(-0.01, 0.01), self.belt_y[0] + 0.07, BELT["top"] + half_height(PARTS[p]) + 0.001]
        d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]; d.qvel[v:v + 6] = 0
        mujoco.mj_forward(self.model, d); self.fed.append(p)

    @property
    def part(self) -> str:
        """The part the plan is currently working on ("" at finish)."""
        return self.plan[min(self.stage_i, len(self.plan) - 1)][0]

    @property
    def stage(self) -> str:
        return self.plan[self.stage_i][1] if self.stage_i < len(self.plan) else "done"

    @property
    def tool(self) -> Optional[str]:
        return JOIN_TOOL[PARTS[self.part]["join"]] if self.part else None

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if self._holding_anything():
            n = np.linalg.norm(a[:3])
            if n > 0.4:
                a[:3] *= 0.4 / n                                               # payload speed cap (see assembly_env)
            if self.stage != "fasten":
                a[3] = np.clip(a[3], -0.5, 0.5)
        self.target[:3] += a[:3] * self.cfg["max_dpos"]
        self.target[3] = wrap(self.target[3] + a[3] * self.cfg["max_dyaw"])
        self.target[:3] = np.clip(self.target[:3], [0.15, -0.55, 0.075], [0.80, 0.55, 0.55])
        self.grip_cmd = float(a[4])
        self._physics(self.target, self.grip_cmd)
        self._tool_changer()
        self.steps += 1
        self._scanner()
        r = self._readings()
        self._join_logic(r)
        r = self._readings()
        regressed = self._recovery(r)
        if regressed:
            self.prev_progress = self._stage_progress(r)
        ok = self._stage_ok(r)
        self.ok_steps = self.ok_steps + 1 if ok else 0
        prog = self._stage_progress(r)
        reward = 10.0 * (prog - self.prev_progress)
        advanced = False
        if self.ok_steps >= 5 and self.stage_i < len(self.plan):
            reward += 10.0; advanced = True
            if self.stage == "place":                                          # next part onto the belt while we join
                nxt = [p for p in self.order if p not in self.fed]
                if nxt:
                    self._feed(nxt[0])
            self.stage_i += 1; self.ok_steps = 0; self.twist = 0.0; self.dwell = 0; self.press_n = 0
            r = self._readings(); prog = self._stage_progress(r) if self.stage_i < len(self.plan) else 0.0
        self.prev_progress = prog
        dropped = self._dropped(r)
        reward -= self.cfg["time_penalty"]
        if dropped:
            reward -= 10.0
        if regressed:
            reward -= 5.0
        terminated = bool(dropped or self.stage_i >= len(self.plan))
        truncated = self.steps >= self.cfg["max_steps"]
        info = self._info(r); info.update(advanced=advanced, regressed=regressed, dropped=dropped, reward=reward)
        return self._obs(r), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera=self.cfg["camera"]); return self._renderer.render()

    def render_inspect(self, size: int = 336) -> np.ndarray:
        """RGB frame from the inspection camera (DiffusionGemma's eye)."""
        if self._insp_renderer is None:
            self._insp_renderer = mujoco.Renderer(self.model, size, size)
        self._insp_renderer.update_scene(self.data, camera="inspect"); return self._insp_renderer.render()

    # ---- low level ---------------------------------------------------------------------------------------------
    def _physics(self, target, grip) -> None:
        m, d = self.model, self.data
        d.ctrl[:7] = self._ik(target); d.ctrl[7] = 0.0 if grip > 0 else 160.0
        for _ in range(self.substeps):
            self._conveyor(); mujoco.mj_step(m, d)
        t = d.time * self.cfg["belt_speed"]; L = self.belt_y[1] - self.belt_y[0]
        for i, mid in enumerate(self.stripes):
            d.mocap_pos[mid][1] = self.belt_y[0] + ((i + 0.5) * L / 10 + t) % L

    def _ik(self, target) -> np.ndarray:
        m, d = self.model, self.data
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.sid["tcp"])
        J = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
        pos = d.site_xpos[self.sid["tcp"]]; R = d.site_xmat[self.sid["tcp"]].reshape(3, 3)
        c, s = np.cos(target[3]), np.sin(target[3])
        Re = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]]) @ R.T
        qe = np.zeros(4); mujoco.mju_mat2Quat(qe, Re.flatten())
        rot_err = 2.0 * qe[1:] * np.sign(qe[0]) if abs(qe[0]) > 1e-9 else np.pi * qe[1:]
        err = np.concatenate([np.clip(target[:3] - pos, -0.05, 0.05), np.clip(rot_err, -0.5, 0.5)])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), err)
        q = d.qpos[self.arm_q]; null = (HOME_Q - q) * 0.05; null[6] = 0.0
        dq += (np.eye(7) - np.linalg.pinv(J) @ J) @ null
        return np.clip(q + dq, self.arm_range[:, 0], self.arm_range[:, 1])

    def _extent_y(self, p: str, yaw: float) -> float:
        spec = PARTS[p]
        if "oct" in spec:
            return spec["oct"][0] / np.cos(np.pi / 8)
        c, s = np.cos(yaw), np.sin(yaw); ys = []
        for (cx, cy, _), (hx, hy, _) in spec["boxes"]:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = cx + sx * hx, cy + sy * hy; ys.append(s * x + c * y)
        return max(ys)

    def _on_belt(self, p: str) -> bool:
        x, y, z = self.data.xpos[self.bid[p]]
        return bool(abs(x - BELT["x"]) < BELT["half_w"] and self.belt_y[0] - 0.01 < y < self.belt_y[1] and z < BELT["top"] + 0.03)

    def _on_stop(self, p: str) -> bool:
        y = self.data.xpos[self.bid[p]][1]
        return self._on_belt(p) and y + self._extent_y(p, self._yaw(p)) > STOP_Y - 0.003

    def _yaw(self, p: str) -> float:
        return yaw_of(self.data.xmat[self.bid[p]].reshape(3, 3))

    def _conveyor(self) -> None:
        d = self.data
        for p in self.fed:
            if p in self.joined or not self._on_belt(p) or self._held(p):
                continue
            v = self.dadr[p]
            d.qvel[v + 1] = 0.0 if self._on_stop(p) else self.cfg["belt_speed"]
            d.qvel[v] *= 0.5

    def _scanner(self) -> None:
        """Height scan as the newest fed part passes under the gantry (line-scan station)."""
        p = self.fed[-1]
        if self.scan_part == p:
            return
        y = self.data.xpos[self.bid[p]][1]
        if abs(y - SCAN_Y) < 0.006 and self._on_belt(p):
            hi = self.height_scan(SCAN_HI); self.scans_hi[p] = hi
            self.scans[p] = hi.reshape(SCAN_N, 3, SCAN_N, 3).max(axis=(1, 3)).reshape(-1); self.scan_part = p

    def height_scan(self, n: int = SCAN_N) -> np.ndarray:
        m, d = self.model, self.data
        grp = np.array([0, 1, 0, 0, 0, 0], np.uint8); gid = np.array([-1], np.int32)
        out = np.zeros((n, n), np.float32); h = SCAN_W / n
        for i in range(n):
            for j in range(n):
                pnt = np.array([BELT["x"] - SCAN_W / 2 + (j + 0.5) * h, SCAN_Y - SCAN_W / 2 + (i + 0.5) * h, 0.25])
                dist = mujoco.mj_ray(m, d, pnt, np.array([0, 0, -1.0]), grp, 1, -1, gid)
                if dist >= 0:
                    out[i, j] = np.clip((0.25 - dist - BELT["top"]) / 0.025, 0, 1)
        return out.reshape(-1)

    def _tool_changer(self) -> None:
        m, d = self.model, self.data
        for t in TOOLS:
            if not self.locked[t] and self.grip_cmd > 0 and self.finger_bodies <= self._contacts(self.bid[t]):
                # self-aligning tool changer (locating pins): lock the tool UPRIGHT (yaw kept) and centred under the hand
                hp, hq = d.xpos[self.bid["hand"]], d.xquat[self.bid["hand"]]; tp = d.xpos[self.bid[t]].copy()
                ty = self._yaw(t); tq = np.array([np.cos(ty / 2), 0, 0, np.sin(ty / 2)])
                tcp = d.site_xpos[self.sid["tcp"]]; tp[:2] = tcp[:2]
                hqi = np.zeros(4); mujoco.mju_negQuat(hqi, hq)
                rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, tp - hp, hqi); rq = np.zeros(4); mujoco.mju_mulQuat(rq, hqi, tq)
                e = self.tc_eq[t]; m.eq_data[e, 3:6] = rp; m.eq_data[e, 6:10] = rq; d.eq_active[e] = 1; self.locked[t] = True
            elif self.locked[t] and self.grip_cmd <= 0:
                d.eq_active[self.tc_eq[t]] = 0; self.locked[t] = False

    def _contacts(self, body: int) -> set:
        d, m = self.data, self.model; out = set()
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if b1 == body: out.add(b2)
            elif b2 == body: out.add(b1)
        return out

    def _held(self, p: str) -> bool:
        return self.finger_bodies <= self._contacts(self.bid[p])

    def _holding_anything(self) -> bool:
        return any(self.locked.values()) or (self.part != "" and self._held(self.part))

    def _press_force(self, p: str) -> float:
        d, m = self.data, self.model; f6 = np.zeros(6); tot = 0.0
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if {b1, b2} & self.finger_bodies and self.bid[p] in (b1, b2):
                mujoco.mj_contactForce(m, d, i, f6)
                if abs(c.frame[2]) > 0.7:                                       # roughly vertical contact normal = pushing down
                    tot += abs(f6[0])
        return tot

    def _weld_part(self, p: str) -> None:
        m, d = self.model, self.data
        hp, hq = d.xpos[self.bid["chassis"]], d.xquat[self.bid["chassis"]]; pp, pq = d.xpos[self.bid[p]], d.xquat[self.bid[p]]
        hqi = np.zeros(4); mujoco.mju_negQuat(hqi, hq)
        rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, pp - hp, hqi); rq = np.zeros(4); mujoco.mju_mulQuat(rq, hqi, pq)
        e = self.join_eq[p]; m.eq_data[e, 3:6] = rp; m.eq_data[e, 6:10] = rq; d.eq_active[e] = 1; self.joined.add(p)

    def _deco(self, p: str, prefix: str, i: int, rgba) -> None:
        g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{p}_{prefix}{i}")
        if g >= 0:
            self.model.geom_rgba[g] = rgba

    def _join_logic(self, r) -> None:
        p, st = self.part, self.stage
        if not p or p in self.joined:
            return
        # pressing compresses the (soft, mass-scaled) contact under the light cap: allow 12 mm of sink while pushing
        seated = r["in_nest"] or (st == "press" and r["nest_dist_xy"] < self.cfg["place_tol"] and abs(r["part_pos"][2] - r["nest"][2]) < 0.012)
        if not seated:
            return
        if st == "fasten" and r["tool_held"] == "driver":
            k = self.fast_done[p]
            yaw = r["tcp_yaw"]; dyaw = wrap(yaw - (self._last_yaw if self._last_yaw is not None else yaw)); self._last_yaw = yaw
            if r["tip_lateral"] < self.cfg["fasten_tol"] and r["tip_vertical"] < self.cfg["fasten_tol"]:
                self.twist += abs(np.degrees(dyaw))
            if self.twist >= self.cfg["fasten_twist_deg"]:
                self._deco(p, "fast", k, [0.1, 0.8, 0.2, 1]); self.fast_done[p] = k + 1; self.twist = 0.0
                if self.fast_done[p] >= len(PARTS[p]["fasteners"]):
                    self._weld_part(p)
        elif st == "weld" and r["tool_held"] == "torch":
            k = self.seam_done[p]
            if r["tip_lateral"] < self.cfg["weld_tol"] and -0.002 < r["tip_dz"] < self.cfg["weld_gap"]:
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg["weld_dwell"]:
                self._deco(p, "bead", k, [1.0, 0.55, 0.1, 1]); self.seam_done[p] = k + 1; self.dwell = 0
                if self.seam_done[p] >= len(PARTS[p]["seam"]):
                    self._weld_part(p)
        elif st == "press":
            if self.grip_cmd > 0 and r["press_force"] > self.cfg["press_force"]:
                self.press_n += 1
            else:
                self.press_n = max(0, self.press_n - 1)
            if self.press_n >= self.cfg["press_steps"]:
                self._deco(p, "ring", "", [0.1, 0.8, 0.2, 1]); self._weld_part(p)

    def _recovery(self, r) -> bool:
        st = self.stage; back = None
        if st in ("lift", "rotate") and not r["held"] and self.ok_steps == 0:
            self._lost += 1; back = "reach" if self._lost >= 6 else None
        elif st == "place" and not r["held"] and not r["in_nest"] and self.grip_cmd <= 0 and r["finger_gap"] > 0.04 \
                and (r["nest_dist_xy"] > 0.01 or r["part_pos"][2] - r["nest"][2] > 0.005):
            self._lost += 1; back = "reach" if self._lost >= 10 else None
        elif st in ("fasten", "weld") and r["tool_held"] != self.tool and self.part not in self.joined:
            self._lost += 1; back = "get_tool" if self._lost >= 6 else None
        else:
            self._lost = 0
        if back is None:
            return False
        p = self.part
        self.stage_i = next(i for i, (q, s) in enumerate(self.plan) if q == p and s == back)
        self._lost = 0; self.ok_steps = 0; return True

    def _dropped(self, r) -> bool:
        for p in self.fed:
            if p not in self.joined and self.data.xpos[self.bid[p]][2] < 0.03:
                return True
            if p not in self.joined and self.data.xmat[self.bid[p]][8] < 0.6 and not self._held(p):
                return True                                                    # knocked over (on its edge): unrecoverable
        for t in TOOLS:
            tp = self.data.xpos[self.bid[t]]; h = self.data.site_xpos[self.sid[f"holder_{t}"]]
            if tp[2] < 0.03 and np.linalg.norm(tp[:2] - h[:2]) > 0.03:
                return True
        return False

    # ---- readers (privileged) ----------------------------------------------------------------------------------
    def _tool_in_holder(self, t: str) -> bool:
        d = self.data; tp = d.xpos[self.bid[t]]; h = d.site_xpos[self.sid[f"holder_{t}"]]
        up = float(d.xmat[self.bid[t]].reshape(3, 3)[2, 2])
        return bool(np.linalg.norm(tp[:2] - h[:2]) < 0.025 and tp[2] < 0.10 and up > 0.90)

    @staticmethod
    def grasp_period(p: str) -> float:
        """Jaw-yaw period of a part: fingers across the short axis (pi), across any pair of flats for the cap."""
        return PARTS[p].get("grasp_sym", min(np.pi, PARTS[p]["sym"]))

    @staticmethod
    def seat_z(t: str) -> float:
        """Tool body z when standing in its holder (tip on the table)."""
        from build_multi_cell import TOOL_HALF
        return 2 * TOOLS[t]["tip_half"] + TOOL_HALF

    def key_err(self, p: str) -> float:
        sym = PARTS[p]["sym"]
        return 0.0 if sym == 0 else abs(np.degrees(wrap(self._yaw(p), sym)))

    def join_target(self, p: str) -> np.ndarray:
        d = self.data; spec = PARTS[p]
        if spec["join"] == "screw":
            k = min(self.fast_done[p], len(spec["fasteners"]) - 1); return d.site_xpos[self.sid[f"{p}_fast{k}"]].copy()
        if spec["join"] == "weld":
            k = min(self.seam_done[p], len(spec["seam"]) - 1); return d.site_xpos[self.sid[f"{p}_seam{k}"]].copy()
        nest = d.site_xpos[self.sid[f"nest_{p}"]]                               # press-fit: the seated cap's top face
        return np.array([d.xpos[self.bid[p]][0], d.xpos[self.bid[p]][1], nest[2] + half_height(spec)])

    def _readings(self) -> Dict[str, Any]:
        d = self.data; p = self.part or self.order[-1]
        tcp = d.site_xpos[self.sid["tcp"]].copy(); tcp_yaw = yaw_of(d.site_xmat[self.sid["tcp"]].reshape(3, 3))
        pp = d.xpos[self.bid[p]].copy(); nest = d.site_xpos[self.sid[f"nest_{p}"]].copy(); ke = self.key_err(p)
        in_nest = bool(np.linalg.norm(pp[:2] - nest[:2]) < self.cfg["place_tol"] and abs(pp[2] - nest[2]) < self.cfg["place_tol"]
                       and ke < 8.0) or p in self.joined
        held_tool = next((t for t in TOOLS if self.locked[t] or self._held_tool(t)), None)
        jt = self.join_target(p); t_need = JOIN_TOOL[PARTS[p]["join"]]
        tip = d.site_xpos[self.sid[f"{held_tool or t_need or 'driver'}_tip"]].copy()
        return {"part": p, "stage": self.stage, "tcp": tcp, "tcp_yaw": tcp_yaw, "part_pos": pp, "part_yaw": self._yaw(p),
                "key_err_deg": ke, "nest": nest, "nest_dist_xy": float(np.linalg.norm(pp[:2] - nest[:2])),
                "held": self._held(p), "in_nest": in_nest, "joined": p in self.joined,
                "on_belt": self._on_belt(p), "on_stop": self._on_stop(p), "scanned": p in self.scans,
                "tool_need": t_need, "tool_held": held_tool, "tool_tip": tip, "join_target": jt,
                "tip_lateral": float(np.linalg.norm(tip[:2] - jt[:2])), "tip_vertical": float(abs(tip[2] - jt[2])),
                "tip_dz": float(tip[2] - jt[2]),
                "tools": {t: {"pos": d.xpos[self.bid[t]].copy(), "grip": d.site_xpos[self.sid[f"{t}_grip"]].copy(),
                              "holder": d.site_xpos[self.sid[f"holder_{t}"]].copy(), "in_holder": self._tool_in_holder(t),
                              "yaw": self._yaw(t)} for t in TOOLS},
                "finger_gap": float(d.qpos[self.finger_q].sum()), "twist": self.twist,
                "fast_done": self.fast_done[p], "seam_done": self.seam_done[p], "press_n": self.press_n,
                "press_force": self._press_force(p) if self.stage == "press" else 0.0,
                "n_joined": len(self.joined), "n_parts": len(self.parts)}

    def _held_tool(self, t: str) -> bool:
        return self.finger_bodies <= self._contacts(self.bid[t])

    def _stage_ok(self, r) -> bool:
        st, cfg = self.stage, self.cfg
        if st == "inspect":
            return r["scanned"]
        if st == "reach":
            yaw_ok = abs(np.degrees(wrap(r["tcp_yaw"] - r["part_yaw"], self.grasp_period(r["part"])))) < 10.0
            return bool(np.linalg.norm(r["tcp"][:2] - r["part_pos"][:2]) < 0.012 and abs(r["tcp"][2] - r["part_pos"][2]) < 0.005 and yaw_ok and (r["on_stop"] or not r["on_belt"]))
        if st == "grasp":
            return r["held"] and self.grip_cmd > 0
        if st == "lift":
            return r["held"] and r["part_pos"][2] > cfg["lift_z"]
        if st == "rotate":
            return r["held"] and r["key_err_deg"] < cfg["rotate_tol_deg"]
        if st == "place":
            return r["in_nest"] and not r["held"]
        if st == "get_tool":
            return r["tool_held"] == r["tool_need"] and r["tools"][r["tool_need"]]["pos"][2] > 0.13
        if st in ("fasten", "weld", "press"):
            return r["joined"]
        if st == "return_tool":
            t = r["tools"][r["tool_need"]]
            return t["in_holder"] and r["tool_held"] is None and r["tcp"][2] > t["pos"][2] + 0.12
        if st == "finish":
            return np.linalg.norm(r["tcp"] - HOME_TCP) < 0.04 and r["n_joined"] == r["n_parts"]
        return False

    def _stage_progress(self, r) -> float:
        st = self.stage
        if st == "inspect":
            return -0.1 * (0.0 if r["scanned"] else 1.0)
        if st == "reach":
            ye = abs(np.degrees(wrap(r["tcp_yaw"] - r["part_yaw"], self.grasp_period(r["part"]))))
            return -float(np.linalg.norm(r["tcp"] - r["part_pos"])) - 0.002 * ye
        if st == "grasp":
            return -0.1 * (0.0 if r["held"] else 1.0)
        if st == "lift":
            return -max(0.0, self.cfg["lift_z"] + 0.01 - r["part_pos"][2]) - (0.0 if r["held"] else 0.1)
        if st == "rotate":
            return -0.002 * r["key_err_deg"] - (0.0 if r["held"] else 0.1)
        if st == "place":
            return -float(np.linalg.norm(r["part_pos"] - r["nest"])) - (0.05 if r["held"] and r["in_nest"] else 0.0)
        if st == "get_tool":
            t = r["tools"][r["tool_need"]]
            return -float(np.linalg.norm(r["tcp"] - t["grip"])) * (0.0 if r["tool_held"] == r["tool_need"] else 1.0) \
                - max(0.0, 0.14 - t["pos"][2])
        if st == "fasten":
            return -r["tip_lateral"] - r["tip_vertical"] - 0.001 * max(0.0, 90.0 - r["twist"]) + 0.1 * r["fast_done"]
        if st == "weld":
            return -r["tip_lateral"] - max(0.0, r["tip_dz"] - 0.004) + 0.1 * r["seam_done"]
        if st == "press":
            top = r["join_target"]
            return -float(np.linalg.norm(r["tcp"] - top)) + 0.02 * r["press_n"]
        if st == "return_tool":
            t = r["tools"][r["tool_need"]]; seat = np.array([t["holder"][0], t["holder"][1], self.seat_z(r["tool_need"])])
            if not t["in_holder"]:
                return -float(np.linalg.norm(t["pos"] - seat)) - 0.5
            if r["tool_held"]:
                return -0.25
            return -0.25 * max(0.0, t["pos"][2] + 0.12 - r["tcp"][2])
        if st == "finish":
            return -0.5 * float(np.linalg.norm(r["tcp"] - HOME_TCP))
        return 0.0

    def _info(self, r=None) -> Dict[str, Any]:
        r = r or self._readings()
        return {**r, "stage": self.stage, "stage_i": self.stage_i, "n_stages": len(self.plan), "steps": self.steps,
                "order": list(self.order), "dropped": False}

    def _obs(self, r=None) -> np.ndarray:
        """Student observation (no part identity, no stage label): proprioception, the part on the line relative to
        the TCP, every nest relative to that part, every tool and holder, the next join feature (fastener head / seam
        mark / part top - visible features) relative to the held tool tip, flags, and the height scan."""
        r = r or self._readings(); tcp = r["tcp"]; y = r["part_yaw"]; dy = wrap(r["tcp_yaw"] - y, np.pi)
        nests = [self.data.site_xpos[self.sid[f"nest_{p}"]] - r["part_pos"] for p in self.parts]
        tools = []
        for t in TOOLS:
            T = r["tools"][t]
            tools += [T["pos"] - tcp, T["holder"] - T["pos"], [float(r["tool_held"] == t), float(T["in_holder"])]]
        prog = max(r["fast_done"], r["seam_done"]) / 4.0
        return np.concatenate([
            tcp, [np.cos(r["tcp_yaw"]), np.sin(r["tcp_yaw"]), r["finger_gap"], self.grip_cmd],
            r["part_pos"] - tcp, [np.cos(y), np.sin(y), np.cos(2 * y), np.sin(2 * y), np.sin(2 * dy), np.cos(2 * dy)],
            *nests, *tools,
            r["join_target"] - r["tool_tip"], r["join_target"] - tcp, HOME_TCP - tcp,
            [float(r["held"]), float(r["in_nest"]), float(r["joined"]), float(r["on_stop"]), float(r["on_belt"]),
             min(r["twist"], 120.0) / 90.0, prog, r["press_n"] / 5.0, min(r["press_force"], 60.0) / 30.0,
             r["n_joined"] / r["n_parts"]],
            self.scans.get(self.part, np.zeros(SCAN_N * SCAN_N, np.float32)),   # scan of the part being handled
        ]).astype(np.float32)
