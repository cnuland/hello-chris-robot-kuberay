"""Generate the multi-part assembly cell (v006): app/ignored/assets/multi_cell.xml.

Phase 1 (default --parts bracket,cap,lbracket): a chassis plate with one fixture nest per part; parts arrive one at a
time on the conveyor in random order (all the same anodised grey - only their SHAPE tells them apart); an inspection
camera + height scanner over the conveyor stop; a tool rack with a screwdriver and a welding torch.

  part      symmetry   joining                                   nest
  bracket   180 deg    screw: 1 twist-lock fastener (driver)     keyed rectangle
  cap       round      press-fit: push down >= 10 N (gripper)    round socket
  lbracket  none       weld: 4-point seam (torch)                L-shaped nest
  plate     90 deg     screw: 2 fasteners (phase 2)              square
  rail      180 deg    weld: 2 tack welds (phase 2)              long slot

Parts that are not on the line yet are parked on the floor under the table (the env teleports the next one onto the
belt). Each part has a weld equality to the chassis ("join_<part>"), each tool a tool-changer weld to the hand.

Usage: app/venv/bin/python app/build_multi_cell.py [--parts bracket,cap,lbracket,plate,rail]
"""
from __future__ import annotations
import argparse, os, re, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.join(HERE, "ignored", "menagerie", "franka_emika_panda")
OUT_DIR = os.path.join(HERE, "ignored", "assets")

BELT = dict(x=0.62, y0=-0.62, y1=-0.06, half_w=0.07, top=0.06)
STOP_Y = -0.12
SCAN_Y = -0.42                         # inspection gantry upstream of the stop: parts are scanned as they pass
FLOOR_TOP = 0.068                      # chassis plate top = nest floor (same height as the v003 pocket floor)
WALL_H, WALL_T, CLEAR = 0.012, 0.004, 0.002
RELIEF = 0.020                         # half-width of the finger-relief notch (jaw pads are ~2 cm wide)
PART_GROUP = 1                         # parts in geom group 1: the height scanner's rays only see this group

# part specs: boxes (centre xyz, half sizes) or a cylinder, in the part frame (origin = grasp point, z = mid-height)
PARTS = {
    "bracket": dict(boxes=[((0, 0, 0), (0.030, 0.015, 0.010))], sym=np.pi, join="screw", nest=(0.40, 0.21), relief="x",
                    fasteners=[(0.0, 0.0)]),
    "cap": dict(oct=(0.017, 0.010), sym=0.0, grasp_sym=np.pi / 4, join="press", nest=(0.50, 0.21),
                relief="xy"),   # octagonal cap: flats for the jaws
    # grasp point 8 mm from the long leg's centre, away from the corner (finger pads clear the short leg by ~9 mm);
    # 2 cm thick like the others, so closed fingertips stay above the nest walls when it is seated
    "lbracket": dict(boxes=[((0.008, 0, 0), (0.030, 0.010, 0.010)), ((0.028, 0.025, 0), (0.010, 0.015, 0.010))],
                     sym=2 * np.pi, join="weld", nest=(0.61, 0.20), relief="x", relief_w=0.020,
                     seam=[(-0.016, -0.0105), (0.000, -0.0105), (0.016, -0.0105), (0.032, -0.0105)]),
    "plate": dict(boxes=[((0, 0, 0), (0.022, 0.022, 0.010))], sym=np.pi / 2, join="screw", nest=(0.43, 0.32), relief="xy",
                  fasteners=[(-0.012, -0.012), (0.012, 0.012)]),
    "rail": dict(boxes=[((0, 0, 0), (0.045, 0.010, 0.010))], sym=np.pi, join="weld", nest=(0.58, 0.32), relief="x",
                 seam=[(-0.035, -0.0105), (0.035, -0.0105)]),
}
TOOLS = {"driver": dict(holder=(0.30, 0.42), tip_half=0.012, rgba="0.9 0.45 0.05 1", tip_rgba="0.7 0.7 0.72 1"),
         "torch": dict(holder=(0.24, 0.30), tip_half=0.022, rgba="0.15 0.35 0.75 1", tip_rgba="0.72 0.45 0.2 1")}
TOOL_R, TOOL_HALF = 0.012, 0.045


def half_height(spec) -> float:
    return spec["oct"][1] if "oct" in spec else max(h[2] for _, h in spec["boxes"])


def circumradius(spec) -> float:
    return spec["oct"][0] / np.cos(np.pi / 8)


def oct_vertices(spec) -> str:
    a, hh = spec["oct"]; R = circumradius(spec)
    return " ".join(f"{R * np.cos(np.pi / 8 + k * np.pi / 4):.5f} {R * np.sin(np.pi / 8 + k * np.pi / 4):.5f} {z:.4f}"
                    for z in (-hh, hh) for k in range(8))


def inside(spec, x, y, grow: float) -> bool:
    """Is (x, y) (part frame) inside the part's footprint grown by `grow`?"""
    if "oct" in spec:                                          # round socket: accepts the octagon at any yaw
        return x * x + y * y <= (circumradius(spec) + grow) ** 2
    return any(abs(x - c[0]) <= h[0] + grow and abs(y - c[1]) <= h[1] + grow for c, h in spec["boxes"])


def nest_walls(spec, res: float = 0.001):
    """Fixture nest walls = (footprint grown by clearance + wall) minus (footprint grown by clearance), rasterised at
    1 mm and merged into boxes (row runs, then identical runs stacked vertically)."""
    ext = 0.06
    xs = np.arange(-ext, ext, res) + res / 2; ys = np.arange(-ext, ext, res) + res / 2
    rows = []
    for y in ys:
        runs, start = [], None
        for i, x in enumerate(xs):
            wall = inside(spec, x, y, CLEAR + WALL_T) and not inside(spec, x, y, CLEAR)
            rel = spec.get("relief", "")                         # finger-relief notches where the jaws go down
            rw = spec.get("relief_w", RELIEF)
            if ("x" in rel and abs(x) < rw) or ("y" in rel and abs(y) < rw):
                wall = False
            if wall and start is None:
                start = i
            if not wall and start is not None:
                runs.append((start, i)); start = None
        if start is not None:
            runs.append((start, len(xs)))
        rows.append(runs)
    boxes, open_ = [], {}                                        # run -> first row index
    for j, runs in enumerate(rows + [[]]):
        cur = set(runs)
        for run in list(open_):
            if run not in cur:
                j0 = open_.pop(run); i0, i1 = run
                boxes.append(((xs[i0] + xs[i1 - 1]) / 2, (ys[j0] + ys[j - 1]) / 2, (i1 - i0) * res / 2, (j - j0) * res / 2))
        for run in cur:
            open_.setdefault(run, j)
    return boxes


def part_geoms(name, spec) -> str:
    out = []
    if "oct" in spec:
        out.append(f'<geom name="{name}_g0" type="mesh" mesh="{name}_oct" mass="0.06" material="part" group="{PART_GROUP}" '
                   f'condim="4" friction="2 0.02 0.001"/>')
    else:
        for i, (c, h) in enumerate(spec["boxes"]):
            out.append(f'<geom name="{name}_g{i}" type="box" pos="{c[0]} {c[1]} {c[2]}" size="{h[0]} {h[1]} {h[2]}" mass="0.05" '
                       f'material="part" group="{PART_GROUP}" condim="4" friction="2 0.02 0.001"/>')
    hh = half_height(spec)
    for i, (fx, fy) in enumerate(spec.get("fasteners", [])):
        out.append(f'<geom name="{name}_fast{i}" type="cylinder" pos="{fx} {fy} {hh + 0.002}" size="0.006 0.002" '
                   f'material="fastener" contype="0" conaffinity="0"/>')
        out.append(f'<site name="{name}_fast{i}" pos="{fx} {fy} {hh + 0.004}" size="0.003"/>')
    for i, (sx, sy) in enumerate(spec.get("seam", [])):
        out.append(f'<geom name="{name}_bead{i}" type="sphere" pos="{sx} {sy} {hh}" size="0.0035" material="bead_cold" '
                   f'contype="0" conaffinity="0"/>')
        out.append(f'<site name="{name}_seam{i}" pos="{sx} {sy} {hh}" size="0.002"/>')
    if spec["join"] == "press":
        out.append(f'<geom name="{name}_ring" type="cylinder" pos="0 0 {hh + 0.0005}" size="{spec["oct"][0] * 0.6} 0.0005" '
                   f'material="bead_cold" contype="0" conaffinity="0"/>')
    return "\n        ".join(out)


def build(parts) -> str:
    src = open(os.path.join(PANDA, "panda.xml")).read()
    mesh_dst = os.path.join(OUT_DIR, "panda_meshes")
    if not os.path.isdir(mesh_dst):
        shutil.copytree(os.path.join(PANDA, "assets"), mesh_dst)
    src = src.replace('meshdir="assets"', 'meshdir="panda_meshes"', 1)
    src = re.sub(r'(<body name="hand"[^>]*>)', r'\1\n          <site name="tcp" pos="0 0 0.1034" size="0.005" rgba="1 1 0 1"/>', src, count=1)
    b = BELT; cy = (b["y0"] + b["y1"]) / 2; hy = (b["y1"] - b["y0"]) / 2
    w = [f'<geom name="belt" type="box" pos="{b["x"]} {cy:.4f} {b["top"] / 2}" size="{b["half_w"]} {hy:.4f} {b["top"] / 2}" material="belt"/>']
    for s in (-1, 1):
        w.append(f'<geom type="box" pos="{b["x"] + s * (b["half_w"] + 0.006)} {cy:.4f} {b["top"] / 2 + 0.006}" '
                 f'size="0.006 {hy:.4f} {b["top"] / 2 + 0.006}" material="frame"/>')
    w.append(f'<geom name="stopper" type="box" pos="{b["x"]} {STOP_Y + 0.004} {b["top"] + 0.012}" size="{b["half_w"]} 0.004 0.012" material="frame"/>')
    for i in range(10):
        y = b["y0"] + (i + 0.5) * (b["y1"] - b["y0"]) / 10
        w.append(f'<body name="stripe{i}" mocap="true" pos="{b["x"]} {y:.4f} {b["top"] + 0.0005}"><geom type="box" '
                 f'size="{b["half_w"] - 0.004} 0.004 0.0005" material="stripe" contype="0" conaffinity="0"/></body>')
    # inspection station: a gantry over the belt upstream of the stop with a downward camera (the DiffusionGemma eye) + light
    w.append(f'<geom type="box" pos="{b["x"]} {SCAN_Y} 0.33" size="0.10 0.012 0.008" material="frame" contype="0" conaffinity="0"/>')
    for s in (-1, 1):
        w.append(f'<geom type="box" pos="{b["x"] + s * 0.10} {SCAN_Y} 0.165" size="0.008 0.008 0.165" material="frame"/>')
    w.append(f'<camera name="inspect" pos="{b["x"]} {SCAN_Y} 0.32" xyaxes="1 0 0 0 1 0" fovy="32"/>')
    w.append(f'<light pos="{b["x"]} {SCAN_Y} 0.31" dir="0 0 -1" diffuse="0.18 0.18 0.18" specular="0 0 0" castshadow="false"/>')
    # chassis plate + nests
    xs = [PARTS[p]["nest"][0] for p in parts]; ys = [PARTS[p]["nest"][1] for p in parts]
    cx, cyy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    hx, hyy = (max(xs) - min(xs)) / 2 + 0.07, (max(ys) - min(ys)) / 2 + 0.065
    body = [f'<geom name="chassis" type="box" pos="{cx:.4f} {cyy:.4f} {FLOOR_TOP / 2}" size="{hx:.4f} {hyy:.4f} {FLOOR_TOP / 2}" material="housing"/>']
    for p in parts:
        nx, ny = PARTS[p]["nest"]
        for (x, y, sx, sy) in nest_walls(PARTS[p]):
            body.append(f'<geom type="box" pos="{nx + x:.4f} {ny + y:.4f} {FLOOR_TOP + WALL_H / 2}" size="{sx:.4f} {sy:.4f} {WALL_H / 2}" material="nest"/>')
        body.append(f'<site name="nest_{p}" pos="{nx} {ny} {FLOOR_TOP + half_height(PARTS[p])}" size="0.003" rgba="1 0.2 0.2 1"/>')
    w.append('<body name="chassis">\n      ' + "\n      ".join(body) + "\n    </body>")
    # tool rack
    for t, ts in TOOLS.items():
        x, y = ts["holder"]; wh, h = 0.0155, 0.025
        for (px, py, sx, sy) in ((x + wh, y, 0.003, wh + 0.003), (x - wh, y, 0.003, wh + 0.003), (x, y + wh, wh, 0.003), (x, y - wh, wh, 0.003)):
            w.append(f'<geom type="box" pos="{px:.4f} {py:.4f} {h}" size="{sx:.4f} {sy:.4f} {h}" material="holder"/>')
        w.append(f'<site name="holder_{t}" pos="{x} {y} 0.0" size="0.004"/>')
        th = ts["tip_half"]
        coll = 'condim="1"'                       # frictionless tip contact: the bit spins in the fastener, the nozzle rests in the cup
        w.append(f"""<body name="{t}" pos="{x} {y} {2 * th + TOOL_HALF + 0.001}">
      <freejoint name="{t}_free"/>
      <geom name="{t}_handle" type="box" size="{TOOL_R} {TOOL_R} {TOOL_HALF}" mass="0.06" rgba="{ts['rgba']}" condim="4" friction="2 0.02 0.001"/>
      <geom name="{t}_tip" type="cylinder" pos="0 0 {-TOOL_HALF - th}" size="0.004 {th}" mass="0.01" rgba="{ts['tip_rgba']}" {coll}/>
      <site name="{t}_tip" pos="0 0 {-TOOL_HALF - 2 * th}" size="0.003"/>
      <site name="{t}_grip" pos="0 0 {TOOL_HALF - 0.030}" size="0.003"/>
    </body>""")
    # parts, parked under the table until the env feeds them onto the belt
    for i, p in enumerate(parts):
        w.append(f"""<body name="{p}" pos="{0.2 + 0.12 * i} 0.6 {-0.74 + half_height(PARTS[p])}">
      <freejoint name="{p}_free"/>
        {part_geoms(p, PARTS[p])}
    </body>""")
    world = f"""
    <light pos="0.5 -0.3 1.6" dir="-0.2 0.2 -1" directional="true" castshadow="true"/>
    <light pos="-0.4 0.5 1.4" dir="0.3 -0.3 -1"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="0 0 0.05" material="floor"/>
    <geom name="table" type="box" pos="0.45 0 -0.02" size="0.55 0.75 0.02" material="table"/>
    {chr(10).join('    ' + x for x in w)}
    <camera name="overview" pos="1.35 -0.95 0.95" xyaxes="0.58 0.81 0 -0.43 0.31 0.85" fovy="45"/>
    <camera name="cell" pos="1.10 0.05 0.80" xyaxes="0 1 0 -0.6 0 0.8" fovy="55"/>
    <camera name="demo" pos="1.20 -0.50 0.85" xyaxes="0.581 0.814 -0.000 -0.554 0.396 0.732" fovy="52"/>
"""
    src = src.replace("<worldbody>", "<worldbody>" + world, 1)
    eq = "\n".join(f'    <weld name="join_{p}" body1="chassis" body2="{p}" active="false" solref="0.005 1"/>' for p in parts)
    eq += "\n" + "\n".join(f'    <weld name="tc_{t}" body1="hand" body2="{t}" active="false" solref="0.005 1"/>' for t in TOOLS)
    meshes = "\n".join(f'    <mesh name="{p}_oct" vertex="{oct_vertices(PARTS[p])}"/>' for p in parts if "oct" in PARTS[p])
    extra = f"""
  <equality>
{eq}
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
    <material name="nest" rgba="0.2 0.25 0.5 1"/>
    <material name="holder" rgba="0.25 0.25 0.28 1"/>
    <material name="part" rgba="0.55 0.58 0.63 1" specular="0.3" shininess="0.4"/>
    <material name="fastener" rgba="0.85 0.15 0.15 1"/>
    <material name="bead_cold" rgba="0.35 0.35 0.38 1"/>
{meshes}
  </asset>
"""
    src = src.replace("</mujoco>", extra + "</mujoco>", 1)
    return src.replace('<mujoco model="panda">', '<mujoco model="panda_multi_cell">', 1)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="bracket,cap,lbracket"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); parts = a.parts.split(",")
    if not os.path.exists(os.path.join(PANDA, "panda.xml")):
        sys.exit(f"missing {PANDA}/panda.xml")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = a.out or os.path.join(OUT_DIR, f"multi_cell_{len(parts)}.xml")
    open(out, "w").write(build(parts))
    import mujoco
    m = mujoco.MjModel.from_xml_path(out)
    print(f"wrote {out}: parts={parts} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} neq={m.neq} nq={m.nq}")


if __name__ == "__main__":
    main()
