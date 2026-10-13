"""DiffusionGemma cell supervisor for the multi-part cell (v006): names the next skill from plain-language facts.

The part identity in the facts is DiffusionGemma's OWN vision answer for that part (vision_labeler), not the
simulator's ground truth - the supervisor reasons from what it saw. Queried once per task-state class (flags) through
the cluster-wide cache, like v004.
"""
from __future__ import annotations
import os, time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import requests

from multi_env import SKILLS, HOME_TCP

NAMES = {"bracket": "the keyed bracket", "cap": "the octagonal cap", "lbracket": "the L-bracket", "plate": "the square plate",
         "rail": "the rail"}
METHOD = {"bracket": "screwed (driver)", "cap": "press-fitted (push down with the closed gripper)", "lbracket": "welded (torch)",
          "plate": "screwed (driver)", "rail": "welded (torch)"}
SKILL_TEXT = {
    "inspect": "wait for the incoming part to pass the inspection camera",
    "reach": "move the open gripper around the part", "grasp": "close the gripper on the part",
    "lift": "lift the held part clear of the conveyor", "rotate": "turn the held part to match its fixture nest",
    "place": "lower the part into its nest and let go", "get_tool": "pick the needed tool out of its holder",
    "fasten": "drive the fastener with the screwdriver", "weld": "weld the seam with the torch",
    "press": "press the part home with the closed gripper", "return_tool": "put the tool back in its holder",
    "finish": "return the arm home (every part is attached)"}
INSTRUCTIONS = (
    "You supervise a robot assembly cell. Parts arrive one at a time on a conveyor. For each part the arm must: "
    "inspect, reach, grasp, lift, rotate, place; then attach it - screwed parts: get_tool, fasten, return_tool; welded "
    "parts: get_tool, weld, return_tool; press-fitted parts: press. When every part is attached: finish. Read the facts, "
    "decide what is already done for the current part, and name the next step. If a part was dropped or let go outside "
    "its nest, that part restarts at reach.")


def _lifted(r) -> bool:
    return bool(r["part_pos"][2] > 0.115 or not r["on_belt"])


def describe(r: Dict[str, Any], seen_as: Optional[str]) -> Dict[str, Any]:
    p = seen_as or "unknown"; f = []
    if not r["scanned"]:
        f.append("A new part is on the conveyor and has NOT reached the inspection camera yet.")
    else:
        f.append(f"The part being handled was identified at inspection as {NAMES.get(p, 'an unknown part')}; it is to be "
                 f"{METHOD.get(p, 'attached')}.")
    if r["joined"]:
        f.append("This part is attached to the main body.")
    elif r["in_nest"] and not r["held"]:
        f.append("The part is seated in its nest and the gripper has let go; it is NOT attached yet.")
    elif r["held"]:
        f.append("The gripper is holding the part.")
        f.append("It has been lifted clear of the conveyor." if _lifted(r) else "It has NOT been lifted yet.")
        f.append("It is turned to match its nest." if r["key_err_deg"] < 5 else f"It is still {r['key_err_deg']:.0f} degrees off its nest key.")
    else:
        where = "waiting at the conveyor stop" if r["on_stop"] else "moving along the conveyor" if r["on_belt"] else "lying loose in the cell"
        f.append(f"The part is {where}; the gripper is not holding it.")
        near = float(np.linalg.norm(r["tcp"] - r["part_pos"])) < 0.02
        f.append("The open gripper is positioned around it." if near else "The gripper is away from it.")
    t = r["tool_held"]
    f.append(f"The gripper is holding the {'screwdriver' if t == 'driver' else 'welding torch'}." if t else "No tool is in the gripper.")
    need = r["tool_need"]
    if need and not t:
        f.append(f"The {'screwdriver' if need == 'driver' else 'welding torch'} is {'in its holder' if r['tools'][need]['in_holder'] else 'out of its holder'}.")
    if r["stage"] == "weld" or r["seam_done"]:
        f.append(f"{r['seam_done']} weld points done.")
    f.append(f"{r['n_joined']} of {r['n_parts']} parts are attached to the main body.")
    f.append("The arm is at its home position." if float(np.linalg.norm(r["tcp"] - HOME_TCP)) < 0.04 else "The arm is not at home.")
    return {"facts": f}


def flags(r: Dict[str, Any], seen_as: Optional[str]) -> Tuple:
    near = float(np.linalg.norm(r["tcp"] - r["part_pos"])) < 0.02
    home = float(np.linalg.norm(r["tcp"] - HOME_TCP)) < 0.04
    need = r["tool_need"]
    return (seen_as, r["scanned"], r["held"], r["in_nest"], r["joined"], r["on_stop"], r["on_belt"], r["tool_held"],
            bool(need and r["tools"][need]["in_holder"]), r["key_err_deg"] < 5.0, _lifted(r), near, home,
            r["n_joined"] == r["n_parts"], r["stage"] == "weld" and r["seam_done"] > 0)


class MultiSkillClient:
    def __init__(self, url: Optional[str] = None, timeout_s: float = 60.0):
        self.url = (url or os.environ.get("DGEMMA_URL", "http://127.0.0.1:18000")).rstrip("/")
        self.timeout_s = timeout_s; self.session = requests.Session()

    def ask(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        opts = "\n".join(f"- {n}: {SKILL_TEXT[n]}" for n in SKILLS)
        user = ("Facts about the cell:\n" + "\n".join("- " + x for x in facts["facts"]) + f"\n\nSteps:\n{opts}\n\n"
                "In at most three short sentences state what is already done for the current part, then give the next step on a "
                "final line 'ANSWER: <step>'.")
        body = {"model": "dgemma", "max_tokens": int(os.environ.get("DGEMMA_MAX_TOKENS", "400")),
                "messages": [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]}
        t0 = time.time()
        try:
            resp = self.session.post(f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout_s); resp.raise_for_status()
            text = (resp.json()["choices"][0]["message"]["content"] or "").strip().lower()
            if "answer:" not in text:
                raise ValueError(f"no ANSWER line: {text[-120:]!r}")
            tail = text.rsplit("answer:", 1)[-1]
            tail = tail.replace("get tool", "get_tool").replace("return tool", "return_tool")
            hits = sorted((tail.find(n), n) for n in SKILLS if n in tail)
            if not hits:
                raise ValueError(f"unparsable {text[-120:]!r}")
            return {"skill": hits[0][1], "source": "dgemma", "latency": time.time() - t0, "text": text[-300:]}
        except Exception as exc:
            return {"skill": None, "source": "error", "latency": time.time() - t0, "error": str(exc)[:160]}
