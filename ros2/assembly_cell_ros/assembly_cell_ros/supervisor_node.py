"""supervisor: DiffusionGemma cell supervisor (v006) - names the next skill from plain-language facts.

Subscribes /cell/task_state and /cell/part_id; whenever the coarse task state changes it asks DiffusionGemma for the
next skill (asynchronously, ~1 Hz at most) and publishes /cell/skill_advice (std_msgs/String JSON). Like the part
inspector it is advisory: the student's own skill head (distilled from these answers) drives the cell; the advice and
the student's choice are logged side by side.
"""
from __future__ import annotations
import json, os, sys, threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, os.environ.get("CELL_APP_DIR", "/opt/cell/app"))


class Supervisor(Node):
    def __init__(self):
        super().__init__("supervisor")
        self.declare_parameter("dgemma_url", os.environ.get("DGEMMA_URL", ""))
        self.url = str(self.get_parameter("dgemma_url").value)
        self.pub = self.create_publisher(String, "/cell/skill_advice", 5)
        self.create_subscription(String, "/cell/task_state", self.on_state, 10)
        self.create_subscription(String, "/cell/part_id", self.on_part, 5)
        self.seen = None; self.last_key = None; self.busy = False

    def on_part(self, msg):
        self.seen = json.loads(msg.data).get("part")

    def on_state(self, msg):
        st = json.loads(msg.data)
        key = (st["part"], st["stage"], tuple(st["attached"]), st["tool_held"])
        if key == self.last_key or self.busy or not self.url:
            return
        self.last_key = key; self.busy = True
        threading.Thread(target=self.advise, args=(st,), daemon=True).start()

    def advise(self, st):
        from multi_skill_client import MultiSkillClient, NAMES, METHOD
        facts = [f"The part being handled was identified as {NAMES.get(self.seen or '', 'an unknown part')}; it is to be "
                 f"{METHOD.get(self.seen or '', 'attached')}.",
                 f"Current activity: {st['stage'].replace('_', ' ')}.",
                 f"{len(st['attached'])} parts are attached to the main body.",
                 f"The gripper holds {'the ' + st['tool_held'] if st['tool_held'] else 'no tool'}."]
        ans = MultiSkillClient(self.url).ask({"facts": facts})
        self.pub.publish(String(data=json.dumps({"for_state": st["stage"], "part": st["part"], "advice": ans.get("skill"),
                                                 "latency_s": round(ans["latency"], 2)})))
        self.busy = False


def main():
    rclpy.init(); n = Supervisor()
    try:
        rclpy.spin(n)
    except Exception:
        pass
