"""student_controller: the distilled lightweight student as the cell's 20 Hz controller (v006).

Subscribes /cell/observation ([seq, obs...]); runs MultiStudent (part-ID head on the height scan, skill head, action
head - ~0.4 M parameters, sub-millisecond on one CPU core) and publishes /cell/command ([seq, dx, dy, dz, dyaw, grip])
plus /cell/student_decision (String JSON: the skill and part the student believes, for the bag / dashboards).
Checkpoint: local path or s3://bucket/key (MinIO, S3_* env).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

sys.path.insert(0, os.environ.get("CELL_APP_DIR", "/opt/cell/app"))


def _load(src: str) -> bytes:
    if src.startswith("s3://"):
        import boto3
        b, k = src[5:].split("/", 1)
        s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"), region_name="us-east-1")
        return s3.get_object(Bucket=b, Key=k)["Body"].read()
    return open(src, "rb").read()


class StudentController(Node):
    def __init__(self):
        super().__init__("student_controller")
        self.declare_parameter("checkpoint", os.environ.get("STUDENT_CKPT", ""))
        self.declare_parameter("parts", "bracket,cap,lbracket")
        from student_multi import MultiStudent
        from multi_env import SKILLS
        self.skills = SKILLS; self.parts = str(self.get_parameter("parts").value).split(",")
        self.pol = MultiStudent.load(_load(str(self.get_parameter("checkpoint").value)))
        n = sum(p.numel() for p in self.pol.parameters())
        self.pub = self.create_publisher(Float32MultiArray, "/cell/command", 10)
        self.pub_dec = self.create_publisher(String, "/cell/student_decision", 10)
        self.create_subscription(Float32MultiArray, "/cell/observation", self.on_obs, 10)
        self.lat = []; self.last = None
        self.get_logger().info(f"student loaded: {n} parameters, parts {self.parts}")

    def on_obs(self, msg):
        seq = msg.data[0]; obs = np.asarray(msg.data[1:], np.float32)
        t0 = time.perf_counter(); a, sk, pid = self.pol.act(obs); self.lat.append(time.perf_counter() - t0)
        self.pub.publish(Float32MultiArray(data=[float(seq)] + [float(x) for x in a]))
        dec = (self.skills[sk], self.parts[pid])
        if dec != self.last:
            self.last = dec
            self.pub_dec.publish(String(data=json.dumps({"seq": int(seq), "skill": dec[0], "part": dec[1],
                                                         "inference_ms_p50": round(1e3 * float(np.median(self.lat)), 3)})))


def main():
    rclpy.init(); n = StudentController()
    try:
        rclpy.spin(n)
    except Exception:
        pass
