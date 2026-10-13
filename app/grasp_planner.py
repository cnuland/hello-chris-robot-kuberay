"""The teacher: sampling-based planning in the simulator with replay verification.

Same contract as fight_planner.FightPlanner.plan(): search from the current state, return an action sequence that
reaches the stage goal without violating the constraints, restore the simulator afterwards, None if nothing is found
within the budget. CEM over open-loop joint-target sequences is the simplest teacher; MPPI is the upgrade.

Lessons carried over from Double Dragon that apply verbatim:
  * a win must be checked AFTER the sequence (dd: `_grounded_after`, `_safe_after`) - here: hold for `settle_s`
    seconds and require the success test to still pass and the part not to drop;
  * prune on any constraint violation inside the rollout, not just at the end;
  * budget by wall time, and log the time and the result on one line.
"""
from __future__ import annotations
import time
from typing import Callable, List, Optional
import numpy as np

from sim_state import replay_prefix


class GraspPlanner:
    def __init__(self, env, horizon: int = 20, samples: int = 64, elites: int = 8, iters: int = 6, settle_s: float = 1.0):
        self.env, self.horizon, self.samples, self.elites, self.iters, self.settle_s = env, horizon, samples, elites, iters, settle_s
        self.plans = 0

    def violated(self, info) -> bool:
        return bool(info["dropped"] or info["force_violation"])

    def rollout(self, seq: np.ndarray) -> float:
        """Score one open-loop sequence from the current state; -inf on violation. Restores the state."""
        snap = self.env.get_state(); score = -np.inf
        try:
            total = 0.0
            for a in seq:
                _, r, term, _, info = self.env.step(a)
                if self.violated(info):
                    return -np.inf
                total += r
            hold = self.env.hold_action(); ok = True
            for _ in range(int(self.settle_s * self.env.cfg["control_hz"])):
                _, _, _, _, info = self.env.step(hold)
                if self.violated(info) or not info["stage_ok"]:
                    ok = False; break
            score = total + (100.0 if ok else 0.0)
        finally:
            self.env.set_state(snap)
        return score

    def plan(self, time_s: float = 60.0, seed: Optional[int] = None) -> Optional[List[np.ndarray]]:
        rng = np.random.default_rng(seed); t0 = time.time(); self.plans += 1
        n = self.env.action_space.shape[0]
        mean = np.tile(self.env.hold_action(), (self.horizon, 1)); std = np.full((self.horizon, n), 0.3)
        best, best_score = None, -np.inf
        for it in range(self.iters):
            if time.time() - t0 > time_s:
                break
            cands = np.clip(mean[None] + std[None] * rng.standard_normal((self.samples, self.horizon, n)), -1, 1)
            scores = np.array([self.rollout(c) for c in cands])
            order = np.argsort(-scores); elite = cands[order[: self.elites]]
            mean, std = elite.mean(0), elite.std(0) + 1e-3
            if scores[order[0]] > best_score:
                best, best_score = cands[order[0]], scores[order[0]]
        found = None
        if best is not None and best_score >= 100.0:              # the settle test passed inside rollout()
            found = replay_prefix(self.env, list(best), self.violated, tail=int(self.settle_s * self.env.cfg["control_hz"]))
        print("grasp-planner stage=%s -> %s in %.0fs (plans so far %d)" % (
            self.env.stage, ("plan %d steps" % len(found)) if found else "no plan", time.time() - t0, self.plans), flush=True)
        return found


class MPCPlanner:
    """v002: receding-horizon predictive sampling (the MuJoCo-MPC approach that solves Allegro in-hand reorientation).

    Every control step: perturb a spline over normalized joint targets (`knots` knots across `horizon` steps), roll each
    candidate forward exactly from the current state, keep the cheapest, execute its first action, shift the spline.
    The nominal (unperturbed, shifted) plan is always candidate 0, so the cost never gets worse by sampling noise.
    Rollouts end early on a drop / force violation with a large penalty - every executed action has been simulated
    from the true state first (dd: verify by replay)."""

    TIPS = ("ff_tip", "mf_tip", "rf_tip", "th_tip")

    W = {"grasp": 4.0, "height": 60.0, "drift": 5.0, "still": 2.0, "angle": 1.0 / 30.0}   # sweep v002: still 0.5 -> 2.0 solved lift (6.4 cm held)

    def __init__(self, env, horizon: int = 12, samples: int = 48, knots: int = 3, sigma: float = 0.2, seed: int = 0,
                 w: dict = None):
        import mujoco
        self.env, self.H, self.N, self.K, self.sigma = env, horizon, samples, knots, sigma
        self.w = {**self.W, **(w or {})}
        self.rng = np.random.default_rng(seed); self.plans = 0
        self.tip_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in self.TIPS]
        self.home_xy = env._part_pose()[0][:2].copy()
        self.nominal = np.tile(env.hold_action(), (self.K, 1))
        self.last_cost = np.inf

    # ---- spline ----------------------------------------------------------------------------------------------
    def _actions(self, knots: np.ndarray) -> np.ndarray:
        t_k = np.linspace(0, self.H - 1, self.K); t = np.arange(self.H)
        return np.stack([np.interp(t, t_k, knots[:, j]) for j in range(knots.shape[1])], axis=1)

    def _shift(self):
        t_k = np.linspace(0, self.H - 1, self.K); t_new = t_k + 1.0
        self.nominal = np.stack([np.interp(t_new, t_k, self.nominal[:, j]) for j in range(self.nominal.shape[1])], axis=1)

    # ---- cost ------------------------------------------------------------------------------------------------
    def _cost(self, info) -> float:
        env, d = self.env, self.env.data
        pos = info["part_pos"]
        tips = d.site_xpos[self.tip_ids]
        dist = np.linalg.norm(tips - pos, axis=1)                         # tip -> part centre
        grasp = 2.0 * dist[3] + float(np.sort(dist[:3])[:2].sum())        # thumb + the two closest fingers
        drift = float(np.linalg.norm(pos[:2] - self.home_xy))
        st = info["stage"]
        w = self.w
        still = float(np.linalg.norm(d.qvel[:3]))                          # part linear speed (free joint first)
        if st == "lift":
            target = env.cfg["lift_height"] + 0.005
            return w["grasp"] * grasp + w["height"] * max(0.0, target - info["lift_dz"]) + w["drift"] * drift + w["still"] * still
        if st == "reorient":
            return w["angle"] * info["angle_err_deg"] + 2.0 * grasp + 40.0 * max(0.0, 0.01 - info["lift_dz"]) + 8.0 * drift + w["still"] * still
        # insert
        return 60.0 * max(0.0, env.cfg["insert_depth"] + 0.003 - info["insert_depth"]) + 300.0 * info["insert_lateral"] \
            + (info["angle_err_deg"] / 30.0) + 1.0 * grasp

    def _rollout(self, acts: np.ndarray) -> float:
        total = 0.0
        for k, a in enumerate(acts):
            _, _, _, _, info = self.env.step(a)
            if info["dropped"] or info["force_violation"]:
                return total + 1e4 * (self.H - k)
            total += self._cost(info) * (1.0 + k / self.H)                   # weight the later steps a little more
        return total

    def act(self) -> np.ndarray:
        snap = self.env.get_state()
        cands = [self.nominal] + [np.clip(self.nominal + self.sigma * self.rng.standard_normal(self.nominal.shape), -1, 1)
                                  for _ in range(self.N - 1)]
        costs = []
        for c in cands:
            costs.append(self._rollout(self._actions(c))); self.env.set_state(snap)
        best = int(np.argmin(costs)); self.nominal = cands[best]; self.last_cost = costs[best]
        a = self._actions(self.nominal)[0].astype(np.float32)
        self._shift(); self.plans += 1
        return a
