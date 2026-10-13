"""Generate the assembly-cell scene app/ignored/assets/allegro_cell.xml from MuJoCo Menagerie `wonik_allegro`.

The Menagerie hand is already palm-UP in its own frame (fingers along +x, flexion lifts the tips toward +z), but it
sits at floor level with nothing around it. The cell mounts it on a fixed pedestal above a table and adds the three
stage props:

  * part        - 5 x 3 x 3 cm connector-like block (50 g, free joint) presented on a pick fixture above the palm
  * fixture     - thin plate under the part; collides ONLY with the part (contype/conaffinity 2), so it never snags
                  the fingers (the fingers keep the Menagerie 1/1 collision bits; the part is 3/3)
  * receptacle  - overhead socket, mouth facing down, rotated 90 deg about z relative to the part's start pose and
                  with 2.5 mm clearance per side, so the stages chain: lift 5 cm -> reorient 90 deg -> insert 1 cm up

Usage: app/venv/bin/python app/build_cell.py   (run from the repo root or app/; idempotent)
"""
from __future__ import annotations
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(HERE, "ignored", "menagerie", "wonik_allegro")
OUT_DIR = os.path.join(HERE, "ignored", "assets")
OUT = os.path.join(OUT_DIR, "allegro_cell.xml")

PALM_Z = 0.20                       # palm origin height above the table top (table top is z = 0)
PALM_TOP = 0.0111                   # palm collision box top, palm frame (measured from right_hand.xml)
PALM_X = -0.0475                    # palm box centre in x (box spans x -0.095 .. 0.0)
PART_X, PART_Y = -0.022, -0.018      # thumb / middle-finger pinch point (FK sweep: tips meet ~4.5 cm apart here, fully curled)
PART = (0.025, 0.015, 0.015)        # part half-sizes: 5 x 3 x 3 cm
PART_MASS = 0.05
FIXTURE_T = 0.002                   # fixture plate half-thickness
LIFT = 0.05                         # spec: lift 5 cm
CLEAR = 0.0025                      # socket clearance per side (spec lateral tolerance is 2 mm)
WALL = 0.003                        # socket wall half-thickness

fixture_z = PALM_Z + PALM_TOP + 0.002 + FIXTURE_T            # plate centre, 2 mm above the palm surface
part_z = fixture_z + FIXTURE_T + PART[2] + 0.0005             # part rests on the plate
mouth_z = part_z + PART[2] + LIFT + 0.004                     # socket mouth: just above the part top after the lift
sock_depth = 0.035                                            # socket inner height (part is 3 cm tall)
# socket is rotated 90 deg about z: its inner half-sizes are (PART y, PART x) + clearance
inner = (PART[1] + CLEAR, PART[0] + CLEAR)


def socket_xml() -> str:
    cx, cy = PART_X, PART_Y
    zc = mouth_z + sock_depth / 2
    ix, iy = inner
    walls = [  # (pos, size) of the four side walls and the ceiling, in world coordinates
        ((cx + ix + WALL, cy, zc), (WALL, iy + 2 * WALL, sock_depth / 2)),
        ((cx - ix - WALL, cy, zc), (WALL, iy + 2 * WALL, sock_depth / 2)),
        ((cx, cy + iy + WALL, zc), (ix, WALL, sock_depth / 2)),
        ((cx, cy - iy - WALL, zc), (ix, WALL, sock_depth / 2)),
        ((cx, cy, mouth_z + sock_depth + WALL), (ix + 2 * WALL, iy + 2 * WALL, WALL)),
    ]
    g = "\n".join(
        f'      <geom type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" '
        f'material="socket" contype="2" conaffinity="2"/>' for p, s in walls)
    gantry = (f'      <geom type="box" pos="{cx:.4f} {cy + 0.10:.4f} {mouth_z + sock_depth + 0.02:.4f}" size="0.006 0.10 0.006" '
              f'material="frame" contype="0" conaffinity="0"/>\n'
              f'      <geom type="box" pos="{cx:.4f} {cy + 0.20:.4f} {(mouth_z + sock_depth + 0.02) / 2:.4f}" '
              f'size="0.008 0.008 {(mouth_z + sock_depth + 0.02) / 2:.4f}" material="frame" contype="0" conaffinity="0"/>')
    return (f'    <body name="receptacle">\n{g}\n{gantry}\n'
            f'      <site name="socket_mouth" pos="{cx:.4f} {cy:.4f} {mouth_z:.4f}" size="0.003" rgba="1 0 0 1"/>\n'
            f'    </body>')


def build() -> str:
    src = open(os.path.join(MENAGERIE, "right_hand.xml")).read()
    meshdir = os.path.relpath(os.path.join(MENAGERIE, "assets"), OUT_DIR)
    src = src.replace('meshdir="assets"', f'meshdir="{meshdir}"', 1)
    src = src.replace('<body name="palm" quat="0 1 0 1"', f'<body name="palm" pos="0 0 {PALM_Z}" quat="0 1 0 1"', 1)
    for f in ("ff", "mf", "rf", "th"):                            # fingertip sites for perception.observe()
        src = re.sub(rf'(<body name="{f}_tip"[^>]*>)', rf'\1\n                <site name="{f}_tip" pos="0 0 0.0267" size="0.004"/>', src, count=1)
    assets = """
  <visual>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.55" rgb2="0.05 0.05 0.08" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.25 0.27 0.3" rgb2="0.2 0.22 0.25" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
    <material name="table" rgba="0.55 0.57 0.6 1"/>
    <material name="frame" rgba="0.3 0.32 0.36 1"/>
    <material name="pedestal" rgba="0.15 0.15 0.17 1"/>
    <material name="fixture" rgba="0.95 0.55 0.1 1"/>
    <material name="part" rgba="0.1 0.45 0.85 1"/>
    <material name="socket" rgba="0.2 0.75 0.35 0.55"/>
  </asset>
"""
    world = f"""
    <light pos="0.3 -0.3 1.2" dir="-0.3 0.3 -1" directional="true" castshadow="true"/>
    <light pos="-0.4 0.4 1.0" dir="0.4 -0.4 -1"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="0 0 0.05" material="floor"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.35 0.30 0.02" material="table"/>
    <geom name="table_leg" type="box" pos="0 0 -0.39" size="0.05 0.05 0.35" material="frame" contype="0" conaffinity="0"/>
    <geom name="pedestal" type="box" pos="{PALM_X - 0.01:.4f} 0 {(PALM_Z - 0.03) / 2:.4f}" size="0.03 0.03 {(PALM_Z - 0.03) / 2:.4f}"
      material="pedestal" contype="0" conaffinity="0"/>
    <geom name="fixture" type="box" pos="{PART_X:.4f} {PART_Y:.4f} {fixture_z:.4f}" size="0.02 0.012 {FIXTURE_T}" material="fixture"
      contype="2" conaffinity="2"/>
    <body name="part" pos="{PART_X:.4f} {PART_Y:.4f} {part_z:.4f}">
      <freejoint name="part_free"/>
      <geom name="part" type="box" size="{PART[0]} {PART[1]} {PART[2]}" mass="{PART_MASS}" material="part"
        contype="3" conaffinity="3" condim="6" priority="1" friction="1.0 0.005 0.0005"/>
      <site name="part_center" size="0.003"/>
    </body>
{socket_xml()}
    <camera name="wrist" pos="0.20 -0.16 {PALM_Z + 0.20:.3f}" xyaxes="0.63 0.78 0 -0.40 0.32 0.86" fovy="55"/>
    <camera name="overview" pos="0.45 -0.45 {PALM_Z + 0.35:.3f}" xyaxes="0.71 0.71 0 -0.36 0.36 0.86" fovy="45"/>
"""
    src = src.replace("<worldbody>", "<worldbody>" + world, 1)
    src = src.replace("</mujoco>", assets + "</mujoco>", 1)
    src = src.replace('<mujoco model="allegro_right">', '<mujoco model="allegro_assembly_cell">', 1)
    return src


def main() -> None:
    if not os.path.exists(os.path.join(MENAGERIE, "right_hand.xml")):
        sys.exit(f"missing {MENAGERIE}/right_hand.xml - run `make assets` (sparse clone of mujoco_menagerie)")
    os.makedirs(OUT_DIR, exist_ok=True)
    open(OUT, "w").write(build())
    import mujoco
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f"wrote {OUT}: {m.nu} actuators, {m.nbody} bodies, {m.ngeom} geoms; part z0={part_z:.4f}, "
          f"socket mouth z={mouth_z:.4f}, palm z={PALM_Z}")


if __name__ == "__main__":
    main()
