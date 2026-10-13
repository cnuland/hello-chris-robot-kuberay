"""cell_sim: the MuJoCo multi-part assembly cell as a ROS 2 node (v006).

Lockstep with the controller: publish the observation for step k, wait for the command, advance one 50 ms control
step. (A free-running timer would add a one-step actuation delay the student never trained with; sim-in-the-loop
rigs use the same lockstep handshake.) If no command arrives within `cmd_timeout_s` the cell holds position.

Publishes  /joint_states                (sensor_msgs/JointState)   Panda joints + fingers, every step
           /cell/observation            (std_msgs/Float32MultiArray) [seq, obs...] for the student
           /cell/inspection/image       (sensor_msgs/Image, rgb8)   once per part, when it passes the gantry
           /cell/inspection/scan        (std_msgs/Float32MultiArray) the 48x48 station height scan, same moment
           /cell/task_state             (std_msgs/String, JSON)     stage / part / attached parts, every step
           /cell/done                   (std_msgs/String, JSON)     episode summary at the end
Subscribes /cell/command                (std_msgs/Float32MultiArray) [seq, dx, dy, dz, dyaw, grip]
Records    <record_dir>/episode_<seed>.mp4 (overview camera) when record_dir is set.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, String

sys.path.insert(0, os.environ.get("CELL_APP_DIR", "/opt/cell/app"))


class CellSim(Node):
    def __init__(self):
        super().__init__("cell_sim")
        self.declare_parameter("seed", 1); self.declare_parameter("parts", "bracket,cap,lbracket")
        self.declare_parameter("record_dir", ""); self.declare_parameter("cmd_timeout_s", 2.0)
        self.declare_parameter("max_steps", 2600)
        from multi_env import MultiAssemblyEnv
        p = self.get_parameter
        self.seed = int(p("seed").value); self.record_dir = str(p("record_dir").value)
        self.env = MultiAssemblyEnv({"parts": tuple(str(p("parts").value).split(",")), "render": bool(self.record_dir),
                                     "camera": "demo", "width": 640, "height": 480, "max_steps": int(p("max_steps").value)})
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_obs = self.create_publisher(Float32MultiArray, "/cell/observation", 10)
        self.pub_img = self.create_publisher(Image, "/cell/inspection/image", 2)
        self.pub_scan = self.create_publisher(Float32MultiArray, "/cell/inspection/scan", 2)
        self.pub_state = self.create_publisher(String, "/cell/task_state", 10)
        self.pub_done = self.create_publisher(String, "/cell/done", 2)
        self.create_subscription(Float32MultiArray, "/cell/command", self.on_command, 10)
        self.seq = 0; self.sent_at = 0.0; self.frames = []; self.inspected = set(); self.holds = 0; self.t0 = time.time()
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
        self.timer = self.create_timer(0.05, self.watchdog)
        self.get_logger().info(f"cell ready: seed {self.seed}, part order {self.env.order}, {len(self.env.plan)} stages")
        self.publish()

    def publish(self):
        d = self.env.data
        js = JointState(); js.header.stamp = self.get_clock().now().to_msg(); js.name = self.joint_names
        js.position = [float(x) for x in np.concatenate([d.qpos[self.env.arm_q], d.qpos[self.env.finger_q]])]
        self.pub_js.publish(js)
        r = self.info
        st = {"seq": self.seq, "step": self.env.steps, "stage": self.env.stage, "part": self.env.part,
              "stage_i": self.env.stage_i, "n_stages": len(self.env.plan), "attached": sorted(self.env.joined),
              "tool_held": r.get("tool_held"), "order": self.env.order}
        self.pub_state.publish(String(data=json.dumps(st)))
        for part, _ in self.env.scans.items():
            if part not in self.inspected:                                     # the part just passed the gantry
                self.inspected.add(part); img = self.env.render_inspect(336)
                m = Image(); m.header.stamp = js.header.stamp; m.header.frame_id = "inspect_camera"
                m.height, m.width = img.shape[:2]; m.encoding = "rgb8"; m.step = img.shape[1] * 3; m.data = img.tobytes()
                self.pub_img.publish(m)
                self.pub_scan.publish(Float32MultiArray(data=[float(x) for x in self.env.scans_hi[part]]))
        self.pub_obs.publish(Float32MultiArray(data=[float(self.seq)] + [float(x) for x in self.obs]))
        self.sent_at = time.time()

    def advance(self, action):
        self.obs, _, term, trunc, self.info = self.env.step(np.asarray(action, np.float32)); self.seq += 1
        if self.record_dir and self.env.steps % 2 == 0:
            self.frames.append(self.env.render())
        if term or trunc:
            self.finish(); return
        self.publish()

    def on_command(self, msg: Float32MultiArray):
        if int(msg.data[0]) != self.seq:
            return                                                             # stale command
        self.advance(list(msg.data[1:6]))

    def watchdog(self):
        if self.seq == 0 and time.time() - self.sent_at > 1.0:
            self.publish(); return                                             # controller not up yet: re-send, don't advance
        if self.seq > 0 and time.time() - self.sent_at > float(self.get_parameter("cmd_timeout_s").value):
            self.holds += 1; self.get_logger().warn(f"no command for seq {self.seq}: holding position")
            self.advance([0, 0, 0, 0, 1.0 if self.env.grip_cmd > 0 else -1.0])

    def finish(self):
        i = self.info
        summary = {"seed": self.seed, "order": self.env.order, "stages": int(i["stage_i"]), "n_stages": int(i["n_stages"]),
                   "success": bool(i["stage_i"] >= i["n_stages"]), "steps": self.env.steps, "attached": sorted(self.env.joined),
                   "dropped": bool(i["dropped"]), "watchdog_holds": self.holds, "wall_s": round(time.time() - self.t0, 1)}
        self.get_logger().info(f"episode done: {summary}")
        self.pub_done.publish(String(data=json.dumps(summary)))
        if self.record_dir and self.frames:
            import imageio.v2 as imageio
            os.makedirs(self.record_dir, exist_ok=True)
            path = os.path.join(self.record_dir, f"ros_episode_seed{self.seed}.mp4")
            imageio.mimwrite(path, self.frames, fps=10, codec="libx264", quality=8)
            json.dump(summary, open(os.path.join(self.record_dir, f"ros_episode_seed{self.seed}.json"), "w"), indent=2)
            self.get_logger().info(f"recorded {path}")
        self.seq = -1; self.timer.cancel()
        self.create_timer(2.0, lambda: rclpy.shutdown())


def main():
    rclpy.init(); node = CellSim()
    try:
        rclpy.spin(node)
    except Exception:
        pass
