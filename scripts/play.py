"""Roll out a trained ics_rl checkpoint on a mjlab pm01 locomotion task.

With ``--video`` the rollout of env 0 is rendered offscreen (EGL, no display
needed) and written as an mp4 next to the checkpoint, e.g.
``uv run scripts/play.py Ics-AMP-Walk-PM01 --video True --video-length 500``.
"""

from __future__ import annotations

import os

# Must be set before mujoco is imported (via mjlab) so offscreen rendering works
# headless. setdefault so e.g. MUJOCO_GL=osmesa can still be forced from outside.
os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass
from pathlib import Path

import mjlab
import torch
import tyro
from ics_rl.env.mjlab import MjlabVecEnv
from ics_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import ics_rl_lab.tasks  # noqa: F401  (import registers every task)
from ics_rl_lab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


@dataclass(frozen=True)
class PlayConfig:
  num_envs: int = 4
  device: str | None = None
  checkpoint: str | None = None
  """Explicit checkpoint path. If None, the newest run/checkpoint under
  logs/ics_rl/<experiment_name>/ is used."""
  num_steps: int = 1000
  video: bool = False
  """Record an mp4 of env 0 (offscreen, works without a display)."""
  video_length: int = 500
  """Frames per video (one frame per env step). Capped by num_steps."""
  video_height: int = 720
  video_width: int = 1280
  video_folder: str | None = None
  """Output directory. If None, <checkpoint_dir>/videos/ is used."""
  export_onnx: bool = True
  """Export the loaded policy (+ deployment metadata) to
  <checkpoint_dir>/exported/<checkpoint_name>/<run_name>.onnx (same layout as train.py)."""


def main(task_id: str | None = None) -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(PlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  configure_torch_backends()

  env_cfg = load_env_cfg(chosen_task, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  rl_cfg = load_rl_cfg(chosen_task)

  if cfg.checkpoint is not None:
    checkpoint_path = Path(cfg.checkpoint)
  else:
    log_root = Path("logs") / "ics_rl" / rl_cfg.experiment_name
    checkpoint_path = get_checkpoint_path(log_root, checkpoint=r"model_\d+\.pt")
  print(f"[INFO] Loading checkpoint: {checkpoint_path}")

  if cfg.video:
    env_cfg.viewer.height = cfg.video_height
    env_cfg.viewer.width = cfg.video_width
  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if cfg.video:
    video_folder = (
      Path(cfg.video_folder)
      if cfg.video_folder is not None
      else checkpoint_path.parent / "videos"
    )
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      name_prefix=f"{chosen_task}-{checkpoint_path.stem}",
    )
  vec_env = MjlabVecEnv(
    env,
    policy_group=rl_cfg.policy_observation_group,
    critic_group=rl_cfg.critic_observation_group,
  )

  runner = OnPolicyRunner(vec_env, rl_cfg.to_dict(), device=device)
  runner.load(str(checkpoint_path))
  policy = runner.get_inference_policy(device=device)

  obs, _ = vec_env.get_observations()
  if cfg.export_onnx:
    export_dir = checkpoint_path.parent / "exported" / checkpoint_path.stem
    export_dir.mkdir(parents=True, exist_ok=True)
    runner.export_as_onnx(obs[0:1], str(export_dir), name=checkpoint_path.parent.name)
    print(f"[INFO] Exported policy to: {export_dir}")
  with torch.inference_mode():
    for _ in range(cfg.num_steps):
      actions = policy(obs)
      obs, _, _, _ = vec_env.step(actions)

  vec_env.close()
  print("[INFO] Rollout finished without crashing.")


if __name__ == "__main__":
  main()
