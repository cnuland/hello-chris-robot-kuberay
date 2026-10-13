"""Student policy client - same role as the Double Dragon decision_client: observation in, action out, with a
fallback when the server is unreachable (the dd runs crawled when the vLLM port-forward died; keep the timeout
short and the fallback explicit).

TODO: pick the student. Options: a vision-language-action model served by vLLM (dd's DiffusionGemma path), or a
small torch policy loaded in-process (fastest for a 20 Hz control loop)."""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import requests


@dataclass
class Decision:
    action: np.ndarray
    confidence: float
    latency_s: float
    source: str          # "model" | "fallback"


class DecisionClient:
    def __init__(self, endpoints: List[str], timeout_s: Optional[float] = None, action_dim: int = 16):
        self.endpoints = endpoints; self.i = 0; self.action_dim = action_dim
        self.timeout_s = float(timeout_s if timeout_s is not None else os.environ.get("STUDENT_TIMEOUT_S", "2"))

    def choose_action(self, obs: np.ndarray, info: Dict[str, Any], image=None) -> Decision:
        body = {"obs": obs.tolist(), "stage": info["stage"]}
        t0 = time.time()
        for _ in range(len(self.endpoints)):
            ep = self.endpoints[self.i % len(self.endpoints)]; self.i += 1
            try:
                r = requests.post(f"{ep}/v1/act", json=body, timeout=self.timeout_s); r.raise_for_status()
                js = r.json()
                return Decision(np.asarray(js["action"], dtype=np.float32), float(js.get("confidence", 0.0)), time.time() - t0, "model")
            except Exception:
                continue
        return Decision(np.zeros(self.action_dim, dtype=np.float32), 0.0, time.time() - t0, "fallback")
