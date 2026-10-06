"""ics_rl AMPAlgoPPO configuration for S2 AMP: the PM01 runner config, only the
experiment name differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ics_rl_lab.rl.config import MjlabRunnerCfg
from ics_rl_lab.tasks.amp.config.pm01.agents.rl_amp_cfg import Pm01AmpFlatWalkRunnerCfg


@dataclass(kw_only=True)
class S2AmpFlatWalkRunnerCfg(Pm01AmpFlatWalkRunnerCfg):
  experiment_name: str = "s2_amp_walk"


@dataclass(kw_only=True)
class S2AmpFlatRunRunnerCfg(Pm01AmpFlatWalkRunnerCfg):
  experiment_name: str = "s2_amp_run"


def s2_amp_rl_cfg() -> MjlabRunnerCfg:
  return S2AmpFlatWalkRunnerCfg()


def s2_amp_run_rl_cfg() -> MjlabRunnerCfg:
  return S2AmpFlatRunRunnerCfg()
