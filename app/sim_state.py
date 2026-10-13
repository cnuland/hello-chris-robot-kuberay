"""Snapshot / restore / replay helpers - the Double Dragon save-state primitive, for MuJoCo.

The teacher saves and restores thousands of times per decision, and every plan is verified by replay before it is
executed (dd: maze_plan_survives / plan_hp_prefix). Keep these three functions tiny and exact."""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

try:
    import mujoco  # noqa: F401
except Exception:  # the template must import without the simulator installed
    mujoco = None


def snapshot(data) -> Tuple:
    """Full physics state: qpos, qvel, act, time, plus ctrl and the solver warm-start (v001: the MPC teacher restores
    thousands of times per step; without ctrl the 'hold' action leaked a sampled rollout's targets)."""
    return (data.qpos.copy(), data.qvel.copy(), data.act.copy(), float(data.time), data.ctrl.copy(), data.qacc_warmstart.copy())


def restore(model, data, snap) -> None:
    qpos, qvel, act, t = snap[:4]
    data.qpos[:] = qpos; data.qvel[:] = qvel; data.act[:] = act; data.time = t
    if len(snap) >= 6:
        data.ctrl[:] = snap[4]; data.qacc_warmstart[:] = snap[5]
    mujoco.mj_forward(model, data)


def replay_prefix(env, actions: Sequence[np.ndarray], violated: Callable[[dict], bool], tail: int = 0,
                  min_len: int = 4) -> Optional[List[np.ndarray]]:
    """Replay `actions` from the current state; return the prefix before the first step where `violated(info)` is
    True (None if shorter than min_len). `tail` idle steps are appended to catch violations that lag the action
    (dd: the life counter lagged a fall by ~15 steps; here a drop shows up a few steps after the release).
    Restores the state afterwards."""
    snap = env.get_state(); cut = len(actions)
    try:
        for i, a in enumerate(actions):
            _, _, _, _, info = env.step(a)
            if violated(info):
                cut = max(0, i - 2); break
        else:
            hold = env.hold_action()
            for _ in range(tail):
                _, _, _, _, info = env.step(hold)
                if violated(info):
                    cut = max(0, len(actions) - 4); break
    finally:
        env.set_state(snap)
    return list(actions[:cut]) if cut >= min_len else None
