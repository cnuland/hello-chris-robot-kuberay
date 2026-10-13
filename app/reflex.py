"""Safety reflexes that outrank the student but not a verified plan (dd: ledge veto, stalactite dodge, walk-in veto).

Keep each one a pure function of the observation and the proposed action, returning a replacement action or None."""
from __future__ import annotations
from typing import Optional
import numpy as np


def joint_limit_guard(env, action: np.ndarray, margin: float = 0.05) -> Optional[np.ndarray]:
    """Clip targets that would drive a joint into its limit."""
    t = env._action_to_targets(action)
    lo, hi = env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1]
    clipped = np.clip(t, lo + margin * (hi - lo), hi - margin * (hi - lo))
    return env._targets_to_action(clipped) if np.any(clipped != t) else None


def force_guard(info, action: np.ndarray, hold: np.ndarray, limit: float) -> Optional[np.ndarray]:
    """Contact force near the limit: hold instead of pressing on (dd: the pit-back veto turned a step into a punch)."""
    return hold if info["contact_force_max"] > 0.8 * limit else None
