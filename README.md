
# ics_rl_lab

<a href="https://www.tum.de/en/news-and-events/all-news/press-releases/details/tum-roboter-gewinnt-best-international-team-award"><img src="docs/ultra_marathon.avif" align="right" width="320" alt="Tienkung Ultra at the Beijing Humanoid Half Marathon (click for the video)"></a>

[![mjlab](https://img.shields.io/badge/mjlab-1.2.0-silver)](https://github.com/mujocolab/mjlab)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.6.0-silver)](https://mujoco.org)
[![ics_rl](https://img.shields.io/badge/ics__rl-1.0.2-silver)](vendor/ics_rl)
[![Python](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://ubuntu.com/download)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE)

Reinforcement-learning environments for humanoid robots, built on
[mjlab](https://github.com/mujocolab/mjlab) and trained with [ics_rl](vendor/ics_rl), a fork of
Instinct RL.

**Beijing humanoid robot half marathon 2026.** The TUM team ran Tienkung Ultra (right) with the
AMP policy trained with this code: 21 km, 39th of 124 robots, and the
[Best International Team Award](https://www.tum.de/en/news-and-events/all-news/press-releases/details/tum-roboter-gewinnt-best-international-team-award) for the best team from outside China.

**Features**

- AMP walking and running, learned from retargeted human motion
- Velocity tracking and hoverboard riding
- A retargeting pipeline from MimicKit human clips to any robot
- ONNX export with deployment metadata, plus a sim2sim check in plain MuJoCo

**Robots**

- **EngineAI PM01**: all tasks; tested on the real robot
- **Tienkung Ultra**: AMP; tested on the real robot, outdoors and in the half marathon
- **EngineAI S2**: AMP, simulation only
- **Unitree G1**: AMP, simulation only

**Built with**

- [mjlab](https://github.com/mujocolab/mjlab): GPU-parallel MuJoCo (Warp) environments
- [ics_rl](vendor/ics_rl): PPO and AMP, a fork of Instinct RL
- [MimicKit](https://github.com/xbpeng/MimicKit): human motion clips for AMP

<br clear="right">

## Simulation and real robot

<table>
  <tr><th></th><th>Simulation</th><th>Real robot</th></tr>
  <tr>
    <td>PM01</td>
    <td><video src="https://github.com/user-attachments/assets/a2fe6406-d0fd-4ab3-b620-f38c43a71d83" autoplay loop muted playsinline width="360"></video></td>
    <td><video src="https://github.com/user-attachments/assets/45834470-b802-4df9-8653-14abc35d0582" autoplay loop muted playsinline width="360"></video></td>
  </tr>
  <tr>
    <td>Tienkung Ultra</td>
    <td><video src="https://github.com/user-attachments/assets/61368439-c167-41ce-8c1d-399174973ce7" autoplay loop muted playsinline width="360"></video></td>
    <td><video src="https://github.com/user-attachments/assets/2571d312-3468-4c7d-8a77-77bdf74c920f" autoplay loop muted playsinline width="360"></video></td>
  </tr>
</table>

## Tasks

| Task id | What | Motion data |
|---|---|---|
| `Ics-AMP-Walk-PM01` / `Ics-AMP-Run-PM01` | AMP walking / running on PM01 | `pm01/amp` |
| `Ics-AMP-Walk-S2` / `Ics-AMP-Run-S2` | AMP walking / running on S2 | `s2/amp` |
| `Ics-AMP-Walk-Ultra` / `Ics-AMP-Run-Ultra` | AMP walking / running on Ultra | `ultra/amp` |
| `Ics-AMP-Walk-G1` / `Ics-AMP-Run-G1` | AMP walking / running on G1 | `g1/amp` |
| `Ics-Hoverboard-Driving-PM01` | riding a self-balancing hoverboard | – |
| `Ics-Locomotion-Flat-PM01` | velocity tracking on rough terrain | – |

## Install

```bash
git clone --recurse-submodules https://github.com/TUM-ICS/ics_rl_lab.git
cd ics_rl_lab
uv sync
```

* **Linux**, Python **3.11–3.13** (3.13 is pinned in `.python-version`)
* [**uv**](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
* An **NVIDIA GPU** (4096–8192 envs needs about 8–16 GB)

## Motion data for AMP

Motion data is not part of the repository; it lives under `data/motions/` (or `$ICS_MOTION_DATA`).
The AMP clips are [MimicKit](https://github.com/xbpeng/MimicKit)'s human motions retargeted onto
each robot.

1. Download [MimicKit_Data.zip](https://1sfu-my.sharepoint.com/personal/xbpeng_sfu_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxbpeng%5Fsfu%5Fca%2FDocuments%2FMimicKit%2FMimicKit%5FData%2Ezip&parent=%2Fpersonal%2Fxbpeng%5Fsfu%5Fca%2FDocuments%2FMimicKit&ga=1) and unzip it into the data root:

   ```bash
   unzip MimicKit_Data.zip -d data/motions/
   ```

2. Retarget and build the AMP clips for a robot:

   ```bash
   uv run scripts/motion/retarget.py --robot pm01
   ```

   This writes the retargeted joint trajectories to `<root>/<robot>/retargeted/*.pkl` and the clips
   the tasks load to `<root>/<robot>/amp/*.npz`. An interrupted run resumes where it stopped.

| Option | Default | |
|---|---|---|
| `--robot` | `pm01` | `pm01`, `s2`, `ultra` or `g1` |
| `--input` | the 116 walking and running clips in `MIMICKIT_CLIPS` | a MimicKit `.pkl` clip or folder |
| `--output-dir` | `<root>/<robot>/retargeted` | the `.npz` clips go to the sibling `amp/` |
| `--workers` | `10` | clips retargeted in parallel |
| `--overwrite` | off | redo clips that already have a `.pkl` |
| `--no-replay` | | only retarget, don't write the `.npz` clips |
| `--view` | off | show the robot over the scaled human while retargeting |

Check one clip visually:

```bash
uv run scripts/motion/retarget.py --robot pm01 --view --no-replay --output-dir /tmp/check \
  --input data/motions/MimicKit_Data/motions/humanoid/parc/humanoid_run00.pkl
```

Under the hood, `retarget.py` fits MimicKit's humanoid to the robot and solves per-frame IK (CPU),
then calls `replay_motions.py`, which plays the clips on the robot in mjlab and records the states
the policy observes at 50 Hz (GPU). Run `replay_motions.py` on its own to rebuild the `.npz` clips
from existing `.pkl` files. To add a robot, add it to `TARGETS` in `retarget.py` and `ROBOTS` in
`replay_motions.py`.

## Train

```bash
uv run scripts/train.py <task-id> [--num-envs N] [--agent.max-iterations N] [--agent.logger wandb]
```

Run in the background:
```bash
nohup uv run scripts/train.py Ics-AMP-Run-PM01 --num-envs 8192 --agent.logger wandb > train_amp.log 2>&1 &
```

## Play / record a video

```bash
uv run scripts/play.py <task-id> [--checkpoint path/to/model_N.pt] [--num-envs 4]
uv run scripts/play.py Ics-AMP-Walk-PM01 --video True --video-length 2000
```

 * Without `--checkpoint`, the newest `model_*.pt` under `logs/ics_rl/<experiment_name>/` is used.
The loaded policy is exported to `<checkpoint_dir>/exported/model_<iter>/<run_name>.onnx` (plus `policy_normalizer.npz`
if the run uses one). The export carries deployment metadata: joint names, PD gains, default pose,
action scale, observation layout and command ranges. `MjlabVecEnv.get_policy_metadata()` collects it.
Command and action terms add their own entries: our terms through a `get_policy_metadata()` method
(see `HullVelocityCommand`), and mjlab's stock terms through the `TERM_METADATA` table in

## Sim2sim (plain MuJoCo)

```bash
uv run scripts/sim2sim.py path/to/exported/model_N/<run>.onnx              # viewer
uv run scripts/sim2sim.py path/to/<run>.onnx --vx 1.5 --headless --duration 10
```

A quick check of an exported policy in plain MuJoCo, on the CPU, without mjlab or torch.
Everything comes from the ONNX metadata (joint order, PD gains, armature, effort limits, default
pose, action scale, observation layout and history, command ranges), and the robot XML is matched
from `assets/` by joint names (`--xml` to override). Viewer keys: UP/DOWN forward speed,
PAGE_UP/PAGE_DOWN sideways, LEFT/RIGHT yaw rate, END zero command, HOME reset.


## Project layout

```
src/ics_rl_lab/
  assets/pm01/         robot MJCF + meshes, mjlab EntityCfg, foot skin patches + sensor cfg
  assets/s2/, ultra/   robot MJCF + meshes (converted from the robots' URDFs), mjlab EntityCfg
  assets/g1/           Unitree G1 MJCF + meshes (copied from mjlab's asset zoo), mjlab EntityCfg
  assets/robot_utils.py  helpers shared by the S2/Ultra entity cfgs
  assets/hoverboard/   hoverboard MJCF + its LQR balancer (action term)
  assets/humanoid/     MimicKit humanoid MJCF (source model for motion retargeting)
  motion_lib/          motion data: tracking MotionLibrary (resident/windowed), AMP loader, data paths
  sensors/             custom sensors (skin)
  rl/config.py         runner/policy/algorithm config dataclasses for ics_rl
  tasks/
    registry.py        register_task(...)
    <task>/            <task>_env_cfg.py (robot-agnostic skeleton), mdp/ (custom terms),
                       config/<robot>/ (robot wiring + agents/ PPO cfg + registration)
vendor/ics_rl/         training library (ics_rl.env.mjlab.MjlabVecEnv = mjlab adapter)
scripts/               train.py, play.py, sim2sim.py, motion/ (retargeting + AMP clip replay)
```

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
`vendor/ics_rl` is a fork of Instinct RL and keeps its own license (CC BY-NC 4.0, see its
`LICENSE`). Robot models in `src/ics_rl_lab/assets/` are subject to their manufacturers' terms.

## Contact

[Simon Armleder](simon.armleder@tum.de)
