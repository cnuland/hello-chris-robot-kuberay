"""DiffusionGemma skill supervisor (v004) - the decision model of this project.

In Double Dragon the vLLM-served DiffusionGemma chose the next button from a structured radar; here it chooses the next
assembly SKILL from a structured description of the cell (the same /v1/systemone "choice" contract). It is queried
only when the task flags change (event-driven, ~1 Hz) - a supervisor over the 20 Hz controller - and its choices are
the skill labels DAgger distills into the lightweight student. If the service is unreachable, the privileged stage
label is used and the record says so (`source`), so agreement can be measured honestly.
"""
from __future__ import annotations
import os, time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import requests

SKILLS: List[Tuple[str, str]] = [
    ("reach", "move the open gripper around the bracket"),
    ("grasp", "close the gripper on the bracket"),
    ("lift", "lift the held bracket clear of the conveyor"),
    ("rotate", "turn the held bracket to match the housing pocket key"),
    ("place", "lower the bracket into the housing pocket and let go"),
    ("get_tool", "pick the driver tool up out of its holder"),
    ("fasten", "press the driver onto the fastener and twist it 90 degrees to lock the bracket"),
    ("finish", "put the driver back in its holder and return the arm home"),
]
SKILL_NAMES = [n for n, _ in SKILLS]
INSTRUCTIONS = (
    "You supervise a robot assembly cell. The arm assembles a keyed bracket onto a housing with this procedure, in "
    "order: reach, grasp, lift, rotate, place, get_tool, fasten, finish. Read the facts about the cell, decide which "
    "steps are already accomplished, and name the next step the arm must perform. If the bracket was dropped or let go "
    "outside the pocket, the procedure restarts at reach."
)


def lifted(r: Dict[str, Any]) -> bool:
    """'Lifted clear of the conveyor': above belt height, or no longer over the conveyor at all (near the housing the
    part is legitimately low again during the place descent)."""
    p = np.asarray(r["part_pos"]); over_belt = abs(p[0] - 0.62) < 0.08 and -0.64 < p[1] < -0.04
    return bool(p[2] > 0.115 or not over_belt)


def describe(r: Dict[str, Any]) -> Dict[str, Any]:
    """Semantic scene description for the decision model (the dd 'radar advice' for this task): plain statements of
    what has already happened, not raw numbers - a language model reasons about procedure, not thresholds."""
    in_pocket, held, tool_held = r["part_in_pocket"], r["part_held"], r["tool_held"]
    facts = []
    if in_pocket:
        facts.append("The bracket is seated in the housing pocket.")
        facts.append("The gripper has let go of the bracket." if not held else "The gripper is still holding the bracket in the pocket.")
    elif held:
        facts.append("The gripper is holding the bracket.")
        facts.append("The bracket has been lifted clear of the conveyor." if lifted(r) else "The bracket has NOT been lifted yet.")
        facts.append("The bracket is turned to match the pocket key." if r["key_err_deg"] < 5.0 else
                     f"The bracket is still {r['key_err_deg']:.0f} degrees off the pocket key.")
    else:
        where = ("waiting at the conveyor stop" if r.get("on_stop") else "moving along the conveyor" if r.get("on_belt") else "lying loose in the cell")
        facts.append(f"The bracket is {where}; the gripper is not holding it.")
        near = float(np.linalg.norm(np.asarray(r["tcp"]) - np.asarray(r["part_pos"]))) < 0.02
        facts.append("The open gripper is positioned around the bracket." if near else "The gripper is away from the bracket.")
    if r["weld_on"]:
        facts.append("The fastener has been driven: the bracket is locked to the housing.")
    else:
        facts.append(f"The fastener has not been driven yet (twisted {r['twist']:.0f} of 90 degrees).")
    facts.append("The driver tool is in the gripper." if tool_held else
                 "The driver tool is standing in its holder." if r["tool_in_holder"] else "The driver tool is lying loose.")
    at_home = float(np.linalg.norm(np.asarray(r["tcp"]) - np.array([0.45, 0.0, 0.32]))) < 0.04
    facts.append("The arm is at its home position." if at_home else "The arm is not at its home position.")
    return {"facts": facts}


def flags(r: Dict[str, Any]) -> Tuple:
    """Coarse task flags; a change means the supervisor should be asked again."""
    near = float(np.linalg.norm(np.asarray(r["tcp"]) - np.asarray(r["part_pos"]))) < 0.02
    home = float(np.linalg.norm(np.asarray(r["tcp"]) - np.array([0.45, 0.0, 0.32]))) < 0.04
    return (r["part_held"], r["part_in_pocket"], r.get("on_stop", False), r.get("on_belt", False), r["tool_held"],
            r["tool_in_holder"], r["weld_on"], r["key_err_deg"] < 5.0, lifted(r), near, home)


class SkillClient:
    """mode "chat" (default): vLLM's OpenAI chat API on :8000 - plain generation, parsed against the skill list.
    vLLM 0.31's constrained-choice path ("diffusion_constrained", used by the /v1/systemone sidecar) requires
    tensor-parallel 1, and on 16 GB T4s the 27 GB checkpoint must be split over 4 GPUs. mode "systemone" keeps the
    sidecar contract for single-GPU deployments."""
    def __init__(self, url: Optional[str] = None, timeout_s: float = None, mode: Optional[str] = None):
        self.mode = mode or os.environ.get("DGEMMA_MODE", "chat")
        self.url = (url or os.environ.get("DGEMMA_URL", "http://127.0.0.1:18000")).rstrip("/")
        self.timeout_s = float(timeout_s if timeout_s is not None else os.environ.get("DGEMMA_TIMEOUT_S", "15"))
        self.session = requests.Session(); self.last_flags = None; self.last = None
        self.stats = {"queries": 0, "model": 0, "fallback": 0, "latency_sum": 0.0}

    def ask(self, r: Dict[str, Any], fallback_stage: str, force: bool = False) -> Dict[str, Any]:
        """Return {"skill", "source", "probs", "latency"}; re-queries only when the task flags change."""
        f = flags(r)
        if not force and self.last is not None and f == self.last_flags:
            return {**self.last, "cached": True}
        self.last_flags = f; self.stats["queries"] += 1
        t0 = time.time()
        try:
            if self.mode == "chat":
                import json as _json
                opts = "\n".join(f"- {n}: {d}" for n, d in SKILLS)
                user = ("Facts about the cell:\n" + "\n".join("- " + f for f in describe(r)["facts"]) + f"\n\nSteps:\n{opts}\n\n"
                        "Briefly state which steps are already accomplished, then give the next step on a final line "
                        "'ANSWER: <step>'.")
                body = {"model": "dgemma", "max_tokens": int(os.environ.get("DGEMMA_MAX_TOKENS", "160")),   # diffusion: no temperature/seed
                        "messages": [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]}
                resp = self.session.post(f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout_s); resp.raise_for_status()
                text = (resp.json()["choices"][0]["message"]["content"] or "").strip().lower()
                tail = text.rsplit("answer:", 1)[-1] if "answer:" in text else text
                hits = sorted((tail.find(n), n) for n in SKILL_NAMES if n in tail)
                skill = hits[0][1] if hits else None; probs = {}
                self.last_text = text
                if skill is None:
                    raise ValueError(f"unparsable answer {text!r}")
            else:
                body = {"model": "dgemma", "instructions": INSTRUCTIONS, "state": describe(r), "images": [],
                        "questions": {"skill": {"type": "choice", "instructions": "Which skill should the arm execute now?",
                                                "criteria": {n: d for n, d in SKILLS}}}}
                resp = self.session.post(f"{self.url}/v1/systemone", json=body, timeout=self.timeout_s); resp.raise_for_status()
                ans = (resp.json().get("answers") or {}).get("skill") or {}
                probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
                skill = ans.get("choice") if ans.get("choice") in SKILL_NAMES else (max(probs, key=probs.get) if probs else None)
            if skill not in SKILL_NAMES:
                raise ValueError(f"unknown skill {ans}")
            dt = time.time() - t0; self.stats["model"] += 1; self.stats["latency_sum"] += dt
            self.last = {"skill": skill, "source": "dgemma", "probs": probs, "latency": dt}
        except Exception as exc:
            self.stats["fallback"] += 1
            self.last = {"skill": fallback_stage, "source": "privileged", "probs": {}, "latency": time.time() - t0, "error": str(exc)[:120]}
        return {**self.last, "cached": False}
