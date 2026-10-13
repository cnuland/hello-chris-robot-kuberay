"""DAgger: the planner labels the states the student visits; aggregate and fine-tune (dd: dagger-after-teacher-clear).

The teacher must clear each stage first; only then does the student's data get labelled. Records are
(obs, image?, stage, teacher_action, student_action, step, run_tag)."""
from __future__ import annotations
import json, os, time
from typing import Any, Dict, List
import numpy as np


class DaggerLog:
    def __init__(self, out_dir: str, run_tag: str):
        self.path = os.path.join(out_dir, f"dagger-{run_tag}.jsonl"); os.makedirs(out_dir, exist_ok=True)
        self.n = 0

    def add(self, obs: np.ndarray, stage: str, teacher_action: np.ndarray, student_action: np.ndarray, step: int) -> None:
        rec = {"t": time.time(), "step": step, "stage": stage, "obs": obs.tolist(),
               "teacher": np.asarray(teacher_action).tolist(), "student": np.asarray(student_action).tolist()}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.n += 1


def aggregate(paths: List[str]) -> Dict[str, Any]:
    """Merge jsonl logs into arrays for a fine-tune step. TODO: upload to MinIO/S3 like the dd traces."""
    X, Y, S = [], [], []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line); X.append(r["obs"]); Y.append(r["teacher"]); S.append(r["stage"])
    return {"obs": np.asarray(X, dtype=np.float32), "act": np.asarray(Y, dtype=np.float32), "stage": S}
