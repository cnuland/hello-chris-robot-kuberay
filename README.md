# hello-chris-robot-kuberay

The follow-on to `hello-chris-dd-kuberay`: the same **decision-model -> distilled-student** recipe that beat Double
Dragon, applied to an industrial assembly cell. A Franka Panda (MuJoCo) assembles three different parts onto a main
body with two tools - a screwdriver (bracket), a welding torch (L-bracket) and a press-fit (cap) - 26 rewarded stages
per episode, parts arriving in random order on a conveyor.

## Results & videos

**[Progression reel (98 s)](media/progression-reel-decision-model-to-finished.mp4)** - labelled: decision model
(DiffusionGemma + teacher) -> DAgger Round 0 ... Round 25 -> RL fine-tuning (PPO) -> Finished (trained student driving
the robot through ROS 2). **[Full assembly through ROS 2](media/ros2-trained-student-full-assembly.mp4).**

| phase | full assemblies (unseen seeds) | stages done | parts attached |
|---|---|---|---|
| DAgger round 9 | 14% | 50% | 1.1/3 |
| DAgger, checkpoint average | 56% | 79% | 2.2/3 |
| DAgger + PPO (final, `models/multi_final_student.pt`) | **80%** (200 seeds) | 92% | 2.7/3 |

Through ROS 2 (Jazzy, on OpenShift): 9 of 12 episodes complete the full assembly.

* **Decision model:** DiffusionGemma-26B (vLLM on 4x T4, ARO) names each part from inspection-scanner facts (300/300)
  and supervises the next skill. Its labels train the student's part-ID and skill heads.
* **Distillation:** DAgger on KubeRay (Ray on OpenShift), then locally; teacher actions + DiffusionGemma labels.
* **RL:** PPO on the per-stage rewards, from the distilled student.
* **Runtime:** ROS 2 graph - `cell_sim` (MuJoCo), `part_inspector` + `supervisor` (DiffusionGemma), `student_controller`.

Credentials are never committed: the cluster scripts read MinIO credentials from environment variables / a
Kubernetes Secret, and `k8s/minio/minio.yaml` ships with `CHANGE_ME` placeholders. The changelog below has the full
history (v000 -> v006d).

## The use case

Connector and small-part insertion at an electronics assembly station, APAC assembly-plant audience. A dexterous
hand on a **fixed base** (Allegro Hand, 16 DoF, Wonik Robotics) picks a part from a fixture, reorients it in hand to
a target pose, and inserts it into a receptacle with a tolerance. Three stages, each with a success test readable
from the simulator - the robot equivalent of a room exit.

Precedents to cite: Siemens + UC Berkeley industrial insertion with RL (2019-2020), NVIDIA DeXtreme (Allegro Hand
in-hand reorientation, sim-to-real, 2022), Fanuc + Preferred Networks bin picking, ETH's privileged-teacher ->
student distillation (ANYmal, 2020), Intrinsic (owner of ROS) learned assembly skills.

## Concept mapping (Double Dragon -> robot cell)

| Double Dragon                               | Robot cell                                                     |
|---------------------------------------------|----------------------------------------------------------------|
| PyBoy emulator, save/load state             | MuJoCo `MjData` snapshot/restore (`app/sim_state.py`)          |
| RAM readers (HP 0xC498, enemy slots)        | `MjData` object pose, joint state, contacts (`perception.py`)  |
| Sprite radar -> structured observation      | `perception.observe()` -> joints, object pose, tip positions   |
| Fight / hop / escape planners (beam search) | `grasp_planner.py` sampling MPC (CEM/MPPI) over joint targets  |
| Reflexes (ledge veto, stalactite dodge)     | `reflex.py` joint-limit / torque / drop guards                 |
| Decision model (DiffusionGemma via vLLM)    | Student policy served by vLLM (VLA) or a local torch policy    |
| DAgger: planner labels the model's states   | `dagger.py` - identical loop                                   |
| Fixtures every 100 steps, replay probes     | Checkpointed scenes every N steps, replay verifier             |
| `hp-drop` per-step trace                    | `trace_line()` per-step contact / pose / violation trace       |
| KubeRay workers + submit script             | `k8s/ray/submit-decision-run.sh` (same shape)                  |
| ROM + .state files in `app/ignored/`        | MJCF assets + `.npz` scene snapshots in `app/ignored/`         |

## Architecture

```
ROS 2 (interface layer)                      Planning / learning layer (this repo)
  allegro_hand_bringup (ros2_control)   <->   app/hand_env.py   Gymnasium env, MuJoCo, get_state/set_state
  mujoco_ros2_control plugin                   app/grasp_planner.py   teacher: sampled rollouts + replay verify
  Gazebo only for integration tests            app/run-decision-hand.py   runner (Ray workers), DAgger logging
                                               app/decision_client.py   student policy client (vLLM)
```

ROS is the interface, not the planning simulator: the teacher needs microsecond snapshot/restore and bit-exact
replay, which MuJoCo gives and Gazebo does not. The same robot description serves both through ros2_control.

## Stages (the "rooms")

1. `lift`     - grasp the part from the fixture and lift it 5 cm; success = object z > threshold for 1 s, no drop.
2. `reorient` - rotate the part in hand to a target quaternion; success = angle error < 10 deg for 1 s.
3. `insert`   - insert into the receptacle; success = tip depth > d_min with lateral error < tolerance, force < F_max.

Each stage has a fixture scene (`app/ignored/scenes/*.npz`), a success test (`hand_env.stage_success`), a planner
budget, and a per-step trace. Debug a failure the Double Dragon way: load the fixture, replay, probe, patch, replay.

## Conventions carried over

* Verify every plan by replay in the simulator before executing it; cut plans at the first violation.
* Save a scene snapshot every 100 steps (local runs) and at every failure.
* One per-step trace line whenever an invariant breaks (drop, contact force spike, joint limit) with the full context.
* Planner-first for every stage from first contact; the student acts only where the planner has no opinion.
* Append every finding to this README's changelog with the run tag and the fixture that reproduces it.

## Running

```
make venv            # app/venv with mujoco + gymnasium + ray
make assets          # sparse-clone Menagerie wonik_allegro + generate app/ignored/assets/allegro_cell.xml
make test            # pytest app/tests
make local           # one local episode: python app/run-decision-hand.py --local --stage lift
RUN_TAG=v001 k8s/ray/submit-decision-run.sh --num-envs 20 --stage lift   # cluster
```

## Video

`app/render_demo.py` renders the Allegro Hand offscreen with MuJoCo and writes `app/ignored/videos/allegro_demo.mp4`
plus stills (scripted finger wave, 8 s, 640x480). The runner records the same way with `--video`: `HandEnv.render()`
frames per step -> mp4 per episode (dd: the mediapy rollout videos). On the cluster, videos go to object storage.

## The assembly cell (v003+) - conveyor -> rotate -> place -> tool -> fasten

`app/build_assembly_cell.py` generates `app/ignored/assets/assembly_cell.xml` from MuJoCo Menagerie `franka_emika_panda`:
a conveyor (kinematic belt, animated stripes, stopper gate) brings a keyed 6x3x2 cm bracket at a random 30-90 deg yaw; the
Panda (7 position servos + gripper, Cartesian [dx, dy, dz, dyaw, grip] actions through damped least-squares IK) must:

| # | stage     | success test (held 0.25 s)                                         | shaped reward                                  |
|---|-----------|---------------------------------------------------------------------|------------------------------------------------|
| 0 | reach     | TCP within 1.5 cm of the part, yaw within 10 deg of its short axis   | -distance, -yaw error                          |
| 1 | grasp     | both fingers in contact, gripper closed                              | held / not held                                |
| 2 | lift      | held >= 5 cm above the belt                                          | -height shortfall                              |
| 3 | rotate    | part yaw on the pocket key (mod 180 deg) within 5 deg                | -key error                                     |
| 4 | place     | seated in the housing pocket (4 mm), released                        | -distance to the pocket                        |
| 5 | get_tool  | twist-lock driver locked in the tool changer and lifted              | -distance to the driver, -height shortfall     |
| 6 | fasten    | driver tip on the fastener, pressed, twisted 90 deg -> weld attaches the part to the housing | -tip offsets, -remaining twist |
| 7 | finish    | driver back in its holder, released, arm home                        | -3D distance to the seat, then -distance home  |

+10 per completed stage, -10 for a drop, -5 for a regression (dropping a part regresses the stage machine - the "recover"
skill), -0.01 per step. Cell engineering that mattered (each found by tracing failures, Double Dragon style): a 10x stiffer
gripper servo (Menagerie's ~3 N squeeze dropped every part), a binary gripper command, a payload speed limit in the
controller, a square driver handle, and a tool-changer lock (weld) instead of friction-holding the driver.

### Learning stack - the Double Dragon recipe on a robot

```
privileged teacher (assembly_teacher.py)  ---- action labels -----\
                                                                   +--> DAgger on KubeRay --> lightweight student --> PPO
DiffusionGemma 26B on vLLM (skill_client)  ---- skill labels ------/       (dagger_assembly.py)  (student_policy.py)   (ppo_assembly.py)
```

* **Teacher** - a closed-loop controller per stage reading exact poses and contacts; no hidden memory, so it can label
  any state a student visits. Full assembly on 12/12 seeds (~460 steps, 23 s of robot time).
* **DiffusionGemma** (`aidendle94/diffusiongemma-26B-A4B-it-INT8-dynamic`, vLLM 0.31 on 4x T4, ARO) - the cell supervisor:
  it reads a plain-language fact sheet of the cell ("the bracket is seated in the pocket", "the driver is in the
  gripper", ...) and names the next procedure step. Queried once per task-state class through a cluster-wide Ray actor
  cache. Measured agreement with the procedure: 88.6% of time steps (common states dominate); per distinct task-state
  class it is 34% (148 classes, v5 audit): ~70% of the misses name the NEXT step early (place->get_tool,
  get_tool->fasten, fasten->finish), and ~20% read "part seated but not yet fastened" as "start over" (reach) - the
  fact sheet's weakest spot. Its decisions are the skill labels distilled into the student.
* **Student** - 215 k-parameter MLP: skill head (8-way, DiffusionGemma's labels) + skill-conditioned action head (teacher's
  labels) + value head for PPO (on DETACHED trunk features - see v005). Sub-millisecond per step on one CPU core - the edge controller the 26 B model is not.
* **KubeRay** - RayCluster `hand-decision` on ARO: 4x Standard_D32s_v3, 56 x 2-CPU workers (112 CPUs); 112 parallel
  episodes per DAgger round in 25-110 s; the image is built in-cluster from `image/Containerfile`.
* **Progression reel** - every DAgger round and PPO checkpoint renders a sampled evaluation episode;
  `app/make_progression.py` captions and concatenates them (`app/ignored/videos/progression/`).

Run it: `RUN_TAG=dagger-v1 k8s/ray/submit-dagger.sh --rounds 8 --episodes 112`, then
`RUN_TAG=ppo-v1 k8s/ray/submit-dagger.sh ppo_assembly.py --init <student.pt>`.

## Changelog

* v000 (template, 2026-10-05): skeleton only, no simulator wired. Decisions recorded above.
* v000a (2026-10-05): MuJoCo 3.14 in `app/venv`, Menagerie `wonik_allegro` fetched (sparse clone), first rendered clip
  `allegro_demo.mp4` (16 actuators, 23 bodies). The default Menagerie scene lays the hand palm-down on the floor;
  the cell scene must mount it upright on the fixed base.
* v001 (2026-10-06): **simulator brought up.** `app/build_cell.py` (`make assets`) generates the assembly cell from
  Menagerie `wonik_allegro`. Correction to v000a: the Menagerie hand is already palm-UP (fingers +x, flexion lifts the
  tips toward +z); it just sat at floor level. Cell = palm on a fixed pedestal 20 cm above a table, a 5x3x3 cm 50 g
  part on a pick fixture plate (collides only with the part, contype 2), an overhead socket (mouth down, rotated
  90 deg, 2.5 mm clearance/side) so the stages chain lift 5 cm -> reorient 90 deg -> insert 1 cm up, fingertip sites,
  `wrist` (student) and `overview` cameras. The part first sat over the palm box (x -0.0475), OUTSIDE the finger
  workspace (no grasp synergy touched it); an FK sweep put the thumb/middle pinch point at (-0.022, -0.018), where
  scripted wraps now reach 7-12 hand-part contacts. `hand_env` success tests wired to the scene (lift relative to the
  fixture rest height, reorient vs the socket yaw modulo the box's 180-deg symmetries, insert = depth past
  `socket_mouth` > 1 cm and lateral < 2 mm, drop = 5 cm below rest). Scene paths anchored to the source files so
  `make test` runs from the repo root instead of silently skipping. Runner fallback: with no student the client
  returned zeros = every joint to MID-range; it now holds the current targets (the Double Dragon random-fallback
  lesson). `make test` passes (bit-exact snapshot/replay); `make local` episode (lift, 400 steps): 0 violations,
  20 teacher plans, fixtures every 100 steps, video - but **no lift plan yet**: blind CEM around a still hand does
  not discover a palm-up 5 cm pinch-and-raise in a 1 s horizon (a scripted wrap lifts ~1 mm). That is the next
  work item: a grasp-primitive / longer-horizon teacher. `render_demo.py --cell` -> `allegro_cell_demo.mp4`.
  ROS 2 layer and the cluster RayCluster are not wired yet (ARO has no KubeRay operator installed).
* v002 (2026-10-06): receding-horizon MPC teacher for the Allegro cell: the 5 cm palm-up lift is feasible after all
  (still-weighted predictive sampling held 6.4 cm and even pushed the part into the socket).
* v003 (2026-10-06): the user goal moved to an arm assembly cell: Panda + conveyor + keyed bracket + housing + twist-lock
  driver; 8 stages with per-stage rewards; privileged teacher completes 12/12.
* v004 (2026-10-06): distillation stack on KubeRay (ARO): DAgger with DiffusionGemma skill labels + teacher action labels
  into a 215 k-param student; PPO fine-tuning with a finish-stage curriculum. DiffusionGemma v2 DAgger round 2 student:
  7/8 stages (through fastening) on 28/28 evaluation episodes. See "The assembly cell" above.
* v005 (2026-10-07): **full autonomous assembly by the distilled student.** DAgger `dagger-dgemma-v5` (8 rounds, beta
  1.0 -> 0.008, 635 k DiffusionGemma-labelled samples, ~67 s collection per round on 112 CPUs), student-only evaluation
  on 28 unseen seeds per round:

  | round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | mean stages /8 | 2.43 | 4.00 | 4.50 | 5.46 | 7.21 | **8.00** | 6.11 | 6.75 |
  | full assemblies | 0% | 0% | 0% | 25% | 54% | **100%** | 39% | 57% |

  Late rounds (beta ~ 0, almost all on-policy recovery data) oscillate, so the run keeps `student_best.pt` (round 5).
  PPO fix: the value head shared the trunk with the distilled heads, and the critic warmup (return-scale loss) rewrote
  the trunk features - full-episode success fell 92% -> 1% with the policy loss OFF (`ppo-v3`). Now the value head reads
  detached features with its own optimizer and clip, and PPO adds a skill-head anchor to DiffusionGemma's decisions;
  `ppo-v4` holds 100% evaluation success (751 -> 753 steps: the 0.01/step time penalty is too weak to shorten cycles).
  `--time-penalty` exposes the cycle-time pressure for PPO (DAgger default unchanged). `ppo-v5` (24 iterations x 112
  episodes, time penalty 0.05, 30% finish-stage curriculum): evaluation success 100/100/100/93/96/100/100% at iterations
  0/4/8/12/16/20/24, successful-cycle length 751 -> 730-772 steps. RL fine-tuning on the stage rewards keeps the
  distilled behaviour; it does not yet shorten the cycle (the student is ~60% slower than the 460-step teacher, mostly
  in the payload-capped carry moves). Progression reel: teacher -> DAgger rounds 0-5 -> PPO iteration 24
  (`app/ignored/videos/progression/progression_reel.mp4`).
* v006 (2026-10-08): **multi-part, multi-tool cell + ROS 2.** Three parts arrive in random order on the conveyor
  (all the same grey - only shape tells them apart): keyed bracket (screwed with the driver), octagonal cap (press-fit,
  force-limited), L-bracket (4-point seam welded with the torch); 26 rewarded stages per episode
  (`build_multi_cell.py`, `multi_env.py`). The five-part version (+ square plate, rail; 44 stages) builds from the same
  specs and the teacher completes 7/8 seeds. Teacher (`multi_teacher.py`) completes ~92% of seeds.
  * **Decision model.** Parts are scanned as they pass an inspection gantry; the station's 48x48 height scan becomes
    plain-language measurements (`inspection_facts.py`) and DiffusionGemma names the part over the chat API:
    **300/300 correct** (`vision_labeler.py`; the RGB-image path returned garbage on the TP=4 T4 deployment and
    desynchronised the ranks, so it is not used). DiffusionGemma also supervises the next skill from cell facts
    (`multi_skill_client.py`), 186 task-state classes.
  * **Distillation (DAgger on KubeRay, 112 CPUs, 1.5 M DiffusionGemma-labelled samples).** Student (`student_multi.py`,
    0.9 M params): part-ID head on the cheap 16x16 scan (labels: DiffusionGemma), skill head (labels: DiffusionGemma),
    gripper as a classifier, and the identified part GATHERS its nest and tool features. Student-only evaluation, 28
    unseen seeds per round: round 0: 4% of stages -> round 6: first full assembly -> **round 9: 14% full assemblies, 50% of
    stages, 1.14/3 parts attached** (best). Late rounds oscillate (round 11: 43% / 0.89).
  * **What made it learnable** (each found with `stage_probe.py`, per-stage competence): (1) every teacher controller
    rewritten as ONE continuous funnel (hover height = f(lateral error)) - threshold switches (hover vs descend, carry
    vs approach) were averaged by the student into early descents and dithering; (2) close/release/twist decisions
    inside usable windows and on the final pose (a 5-step dwell labelled the same pose open then close); (3) the gripper
    as a binary classifier (a regressed grip flickered across its threshold and dropped parts); (4) DiffusionGemma's
    step labels run a phase ahead/behind in this cell (weld states read "return_tool" 73%), so they train the student's
    explainable skill output but no longer condition its motion.
  * **ROS 2** (`ros2/assembly_cell_ros`, Jazzy, built in-cluster, `k8s/ros2/run-ros-episode.sh`): cell_sim (MuJoCo,
    lockstep), part_inspector (DiffusionGemma, named cap -> L-bracket -> bracket correctly), supervisor (DiffusionGemma
    advice), student_controller (the distilled student). Round-9 student through ROS 2 on OpenShift: **seed 777004
    26/26 stages in 1203 steps, seed 777011 26/26 in 1225 steps**, 0 watchdog holds; rosbag (MCAP) recorded.
  * Videos: `~/Documents/robot-multipart-progression-reel.mp4` (labelled: decision model -> DAgger Round 0..11 ->
    Finished, 108 s), `robot-multipart-ROS2-trained-student-full-assembly.mp4`, `...-round6.mp4` (first full assembly),
    `...-round9.mp4`.
* v006b (2026-10-11): **training finished.** Continued DAgger locally (16-core Mac, DiffusionGemma labels through a
  port-forward; KubeRay workers scaled to 0 at the user's request), rounds 12-31. Two fixes moved the student:
  (1) **checkpoint averaging** - the average of good rounds generalises better than any single round (round 16 31% ->
  avg(14,16,18) 49% on 100 unseen seeds); (2) **fine-tuning learning rate 2e-4** for warm-started rounds (1e-3
  re-trained the policy each round and caused the round-to-round swings: the same round went 0% at 1e-3 vs 46% at
  2e-4). Final student `app/ignored/checkpoints/multi_final_student.pt` = average of v10b rounds 20,21,23,24,25:
  **56% full assemblies on 200 unseen seeds (54%/58% on two 100-seed sets), 79% of stages, 2.2/3 parts attached**;
  a further cycle (v11, lr 1e-4) plateaued at 45-50%. Through ROS 2 on OpenShift (worker nodes only): **4 of 6 episodes complete the full assembly**
  (seeds 777001/777003/777004/777005, 1127-1768 steps; 777000 and 777002 stall at the first reach); the launch file now shuts the graph down when the cell
  finishes so Jobs upload their results (rosbag still flushes empty - open item). Reel rebuilt: decision model ->
  Round 0, 3, 6, 9, 12, 14, 16, 21, 25 -> Finished (ROS 2), 91 s, `~/Documents/robot-multipart-progression-reel.mp4`.
* v006c (2026-10-12): **RL fine-tuning - final student.** The DAgger student plateaued at ~56% (a v12 cycle with a
  focus curriculum on the hard stages did not beat it). Two further steps: (1) teacher fix - when the jaws are commanded
  open but pinned against a nest wall the teacher backs straight up to free them (teacher success 90% -> 97% on 100
  seeds); (2) **PPO on the stage rewards** (`ppo_multi.py`, local 14 workers): exploration on the 4-D motion only, the
  gripper stays the deterministic classifier, detached critic + 2-iteration warmup, anchor 0.5 to the DAgger motion,
  lr 1e-5, start-stage + focus curriculum. Final student = PPO iteration 20
  (`app/ignored/checkpoints/multi_final_student.pt`, s3://robot/multi-final/multi_final_student_ppo.pt):
  **72% full assemblies on 200 unseen seeds (76% / 68%), 87% of stages, 2.5/3 parts attached** - up from 56% (DAgger)
  and 14% (round 9). Through ROS 2 on OpenShift worker nodes: **3 of 6 episodes complete the full assembly** (777001, 777003,
  777005; 1126-1159 steps); 777000/777002 drop a part, 777004 attaches 2 of 3 then times out. Reel rebuilt with the RL stage (98 s): decision model -> DAgger Round 0..25 -> PPO -> Finished (ROS 2).
* v006d (2026-10-12): **final student = PPO v2 iteration 24** (a second PPO pass from the v006c student, anchor 0.3):
  **80% full assemblies on 200 unseen seeds (81% / 78%), 92% of stages, 2.67/3 parts attached.** A third pass plateaued
  (79%). Through ROS 2 on OpenShift worker nodes: **9 of 12 episodes complete the full assembly** (1084-1152 steps;
  the misses reach 14, 21 and 23 of 26 stages). rosbag fixed: the recorder was backgrounded in a non-interactive shell,
  where SIGINT is ignored, so the MCAP never closed - the Job now stops it with SIGTERM and waits (bags 0.8-1.8 MB with
  metadata.yaml). Progression (unseen seeds): DAgger round 9 14% -> averaged DAgger 56% -> PPO 72% -> PPO 80%.
  Reel: decision model -> DAgger Round 0..25 -> PPO -> Finished (ROS 2), 98 s, `~/Documents/robot-multipart-progression-reel.mp4`.
