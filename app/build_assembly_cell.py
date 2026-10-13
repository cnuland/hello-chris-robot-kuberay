"""Generate the arm assembly cell app/ignored/assets/assembly_cell.xml (MuJoCo Menagerie franka_emika_panda + props).

Task (v003): a keyed bracket ("part") arrives on a conveyor at a random yaw; the Panda picks it, rotates it to the
housing pocket's key, places it in the pocket, fetches a twist-lock driver ("tool") from its holder, drives the
fastener (press + 90-degree twist -> a weld constraint attaches the part to the housing), returns the tool and goes
home. Everything static is in the robot base frame; table top is z = 0.

Usage: app/venv/bin/python app/build_assembly_cell.py
"""
from __future__ import annotations
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.join(HERE, "ignored", "menagerie", "franka_emika_panda")
OUT_DIR = os.path.join(HERE, "ignored", "assets")
OUT = os.path.join(OUT_DIR, "assembly_cell.xml")

# ---- layout (metres) ----------------------------------------------------------------------------------------------
BELT = dict(x=0.62, y0=-0.62, y1=-0.06, half_w=0.06, top=0.06)       # conveyor runs toward +y
STOP_Y = -0.12                                                        # stopper gate face
PART = (0.030, 0.015, 0.010)                                          # bracket half-sizes 6 x 3 x 2 cm
HOUSING = dict(x=0.47, y=0.22, half=(0.09, 0.07), top=0.08)           # main body, keyed pocket on top (yaw 0)
POCKET = dict(half=(PART[0] + 0.002, PART[1] + 0.002), depth=0.012)   # 2 mm clearance per side
HOLDER = dict(x=0.32, y=0.40)                                         # tool cup
TOOL = dict(r=0.012, half=0.045, tip_r=0.004, tip_half=0.012)         # driver: square 2.4 cm handle (flat faces for the pads) + tip
N_STRIPES = 10


def housing_xml() -> str:
    x, y, (hx, hy), top = HOUSING["x"], HOUSING["y"], HOUSING["half"], HOUSING["top"]
    px, py, depth = POCKET["half"][0], POCKET["half"][1], POCKET["depth"]
    floor_top = top - depth
    g = [  # base slab under the pocket + four rim blocks around it
        ((x, y, floor_top / 2), (hx, hy, floor_top / 2)),
        ((x + (px + hx) / 2, y, floor_top + depth / 2), ((hx - px) / 2, hy, depth / 2)),
        ((x - (px + hx) / 2, y, floor_top + depth / 2), ((hx - px) / 2, hy, depth / 2)),
        ((x, y + (py + hy) / 2, floor_top + depth / 2), (px, (hy - py) / 2, depth / 2)),
        ((x, y - (py + hy) / 2, floor_top + depth / 2), (px, (hy - py) / 2, depth / 2)),
    ]
    geoms = "\n".join(f'      <geom type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" '
                      f'material="housing"/>' for p, s in g)
    return (f'    <body name="housing">\n{geoms}\n'
            f'      <site name="pocket" pos="{x:.4f} {y:.4f} {floor_top + PART[2]:.4f}" size="0.004" rgba="1 0.2 0.2 1"/>\n'
            f'      <geom type="box" pos="{x:.4f} {y:.4f} 0.002" size="{hx + 0.02:.4f} {hy + 0.02:.4f} 0.002" material="jig" '
            f'contype="0" conaffinity="0"/>\n    </body>')


def holder_xml() -> str:
    x, y, w, h = HOLDER["x"], HOLDER["y"], 0.0155, 0.025          # snug 3.1 cm cup, 5 cm tall: captures the handle upright
    walls = [((x + w, y, h), (0.003, w + 0.003, h)), ((x - w, y, h), (0.003, w + 0.003, h)),
             ((x, y + w, h), (w, 0.003, h)), ((x, y - w, h), (w, 0.003, h))]
    geoms = "\n".join(f'      <geom type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" '
                      f'material="holder"/>' for p, s in walls)
    return f'    <body name="holder">\n{geoms}\n      <site name="holder" pos="{x} {y} 0.0" size="0.004"/>\n    </body>'


def belt_xml() -> str:
    b = BELT; cy = (b["y0"] + b["y1"]) / 2; hy = (b["y1"] - b["y0"]) / 2
    s = (f'    <geom name="belt" type="box" pos="{b["x"]} {cy:.4f} {b["top"] / 2}" size="{b["half_w"]} {hy:.4f} {b["top"] / 2}" '
         f'material="belt" friction="1.0 0.005 0.0001"/>\n'
         f'    <geom name="belt_frame_l" type="box" pos="{b["x"] - b["half_w"] - 0.006} {cy:.4f} {b["top"] / 2 + 0.006}" '
         f'size="0.006 {hy:.4f} {b["top"] / 2 + 0.006}" material="frame"/>\n'
         f'    <geom name="belt_frame_r" type="box" pos="{b["x"] + b["half_w"] + 0.006} {cy:.4f} {b["top"] / 2 + 0.006}" '
         f'size="0.006 {hy:.4f} {b["top"] / 2 + 0.006}" material="frame"/>\n'
         f'    <geom name="stopper" type="box" pos="{b["x"]} {STOP_Y + 0.004} {b["top"] + 0.012}" size="{b["half_w"]} 0.004 0.012" '
         f'material="frame"/>\n')
    for i in range(N_STRIPES):                                       # moving stripes (mocap) animate the belt surface
        y = b["y0"] + (i + 0.5) * (b["y1"] - b["y0"]) / N_STRIPES
        s += (f'    <body name="stripe{i}" mocap="true" pos="{b["x"]} {y:.4f} {b["top"] + 0.0005}">'
              f'<geom type="box" size="{b["half_w"] - 0.004} 0.004 0.0005" material="stripe" contype="0" conaffinity="0"/></body>\n')
    return s


def build() -> str:
    src = open(os.path.join(PANDA, "panda.xml")).read()
    # self-contained assets dir (the cluster submit ships app/ignored/assets, not the Menagerie checkout)
    import shutil
    mesh_dst = os.path.join(OUT_DIR, "panda_meshes")
    if not os.path.isdir(mesh_dst):
        shutil.copytree(os.path.join(PANDA, "assets"), mesh_dst)
    src = src.replace('meshdir="assets"', 'meshdir="panda_meshes"', 1)
    # tool-centre-point site between the fingertips
    src = re.sub(r'(<body name="hand"[^>]*>)', r'\1\n          <site name="tcp" pos="0 0 0.1034" size="0.005" rgba="1 1 0 1"/>', src, count=1)
    b = BELT
    part_y = b["y0"] + 0.08
    world = f"""
    <light pos="0.5 -0.3 1.6" dir="-0.2 0.2 -1" directional="true" castshadow="true"/>
    <light pos="-0.4 0.5 1.4" dir="0.3 -0.3 -1"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="0 0 0.05" material="floor"/>
    <geom name="table" type="box" pos="0.45 0 -0.02" size="0.55 0.75 0.02" material="table"/>
{belt_xml()}
{housing_xml()}
{holder_xml()}
    <body name="part" pos="{b["x"]} {part_y:.4f} {b["top"] + PART[2] + 0.001}">
      <freejoint name="part_free"/>
      <geom name="part" type="box" size="{PART[0]} {PART[1]} {PART[2]}" mass="0.08" material="part" condim="4"
        friction="1.2 0.005 0.0001"/>
      <geom name="fastener" type="cylinder" pos="0 0 {PART[2] + 0.002}" size="0.006 0.002" material="fastener"
        contype="0" conaffinity="0"/>
      <site name="fastener" pos="0 0 {PART[2] + 0.004}" size="0.003"/>
    </body>
    <body name="tool" pos="{HOLDER["x"]} {HOLDER["y"]} {2 * TOOL["tip_half"] + TOOL["half"] + 0.001}">
      <freejoint name="tool_free"/>
      <geom name="tool_handle" type="box" size="{TOOL["r"]} {TOOL["r"]} {TOOL["half"]}" mass="0.06" material="tool" condim="4"
        friction="1.2 0.005 0.0001"/>
      <geom name="tool_tip" type="cylinder" pos="0 0 {-TOOL["half"] - TOOL["tip_half"]}" size="{TOOL["tip_r"]} {TOOL["tip_half"]}"
        mass="0.01" material="tooltip"/>
      <site name="tool_tip" pos="0 0 {-TOOL["half"] - 2 * TOOL["tip_half"]}" size="0.003"/>
      <site name="tool_grip" pos="0 0 {TOOL["half"] - 0.030}" size="0.003"/>
    </body>
    <camera name="overview" pos="1.35 -0.95 0.95" xyaxes="0.58 0.81 0 -0.43 0.31 0.85" fovy="45"/>
    <camera name="cell" pos="1.05 0.0 0.75" xyaxes="0 1 0 -0.6 0 0.8" fovy="55"/>
"""
    src = src.replace("<worldbody>", "<worldbody>" + world, 1)
    extra = """
  <equality>
    <weld name="fasten" body1="housing" body2="part" active="false" solref="0.005 1"/>
    <weld name="toolchanger" body1="hand" body2="tool" active="false" solref="0.005 1"/>
  </equality>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.55" rgb2="0.05 0.05 0.08" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.25 0.27 0.3" rgb2="0.2 0.22 0.25" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
    <material name="table" rgba="0.55 0.57 0.6 1"/>
    <material name="frame" rgba="0.3 0.32 0.36 1"/>
    <material name="belt" rgba="0.12 0.12 0.13 1"/>
    <material name="stripe" rgba="0.35 0.35 0.37 1"/>
    <material name="housing" rgba="0.75 0.62 0.2 1"/>
    <material name="jig" rgba="0.2 0.25 0.5 1"/>
    <material name="holder" rgba="0.25 0.25 0.28 1"/>
    <material name="part" rgba="0.1 0.45 0.85 1"/>
    <material name="fastener" rgba="0.85 0.15 0.15 1"/>
    <material name="tool" rgba="0.9 0.45 0.05 1"/>
    <material name="tooltip" rgba="0.7 0.7 0.72 1"/>
  </asset>
"""
    src = src.replace("</mujoco>", extra + "</mujoco>", 1)
    return src.replace('<mujoco model="panda">', '<mujoco model="panda_assembly_cell">', 1)


def main() -> None:
    if not os.path.exists(os.path.join(PANDA, "panda.xml")):
        sys.exit(f"missing {PANDA}/panda.xml - `make assets` (sparse-checkout franka_emika_panda)")
    os.makedirs(OUT_DIR, exist_ok=True); open(OUT, "w").write(build())
    import mujoco
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f"wrote {OUT}: {m.nu} actuators, {m.nbody} bodies, {m.neq} equality, nq={m.nq}")


if __name__ == "__main__":
    main()
