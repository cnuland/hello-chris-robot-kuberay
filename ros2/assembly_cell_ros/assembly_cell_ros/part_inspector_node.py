"""part_inspector: DiffusionGemma part identification at the inspection station (v006).

Subscribes /cell/inspection/scan (48x48 height scan), turns it into plain-language measurements (inspection_facts)
and asks the vLLM-served DiffusionGemma which part it is - the same prompt vision_labeler used to produce the
student's part-ID labels. Publishes /cell/part_id (std_msgs/String, JSON with the
answer and latency). The controller does NOT wait for it - the distilled student identifies the part itself from the
height scan in ~0.1 ms; this node is the 26 B "eye" the student was distilled from, kept in the graph for audit.
"""
from __future__ import annotations
import json, os, sys, threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

sys.path.insert(0, os.environ.get("CELL_APP_DIR", "/opt/cell/app"))


class PartInspector(Node):
    def __init__(self):
        super().__init__("part_inspector")
        self.declare_parameter("dgemma_url", os.environ.get("DGEMMA_URL", ""))
        self.declare_parameter("parts", "bracket,cap,lbracket")
        self.url = str(self.get_parameter("dgemma_url").value); self.parts = str(self.get_parameter("parts").value).split(",")
        self.pub = self.create_publisher(String, "/cell/part_id", 5)
        self.create_subscription(Float32MultiArray, "/cell/inspection/scan", self.on_scan, 2)
        self.get_logger().info(f"part inspector: DiffusionGemma at {self.url or '(disabled)'}")

    def on_scan(self, msg: Float32MultiArray):
        threading.Thread(target=self.classify, args=(np.asarray(msg.data, np.float32),), daemon=True).start()

    def classify(self, scan):
        from inspection_facts import facts
        f = facts(scan)
        if not self.url:
            return
        from vision_labeler import ask_facts
        part, text, lat = ask_facts(self.url, f, self.parts)
        out = {"part": part, "latency_s": round(lat, 2), "facts": f, "model": "DiffusionGemma-26B-A4B", "reply": (text or "")[-160:]}
        self.get_logger().info(f"DiffusionGemma sees: {part} ({lat:.1f}s)")
        self.pub.publish(String(data=json.dumps(out)))


def main():
    rclpy.init(); n = PartInspector()
    try:
        rclpy.spin(n)
    except Exception:
        pass
