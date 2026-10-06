"""ics_rl AMPAlgoPPO configuration for Ultra AMP: the PM01 runner config, only the
experiment name differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ics_rl_lab.rl.config import MjlabRunnerCfg
from ics_rl_lab.tasks.amp.config.pm01.agents.rl_amp_cfg import Pm01AmpFlatWalkRunnerCfg


@dataclass(kw_only=True)
class UltraAmpFlatWalkRunnerCfg(Pm01AmpFlatWalkRunnerCfg):
  experiment_name: str = "ultra_amp_walk"


@dataclass(kw_only=True)
class UltraAmpFlatRunRunnerCfg(Pm01AmpFlatWalkRunnerCfg):
  experiment_name: str = "ultra_amp_run"


def ultra_amp_rl_cfg() -> MjlabRunnerCfg:
  return UltraAmpFlatWalkRunnerCfg()


def ultra_amp_run_rl_cfg() -> MjlabRunnerCfg:
  return UltraAmpFlatRunRunnerCfg()
