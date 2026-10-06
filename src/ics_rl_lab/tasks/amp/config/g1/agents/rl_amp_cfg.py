"""ics_rl AMPAlgoPPO configuration for G1 AMP: the PM01 runner config (as for S2 and
Ultra), only the experiment name differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ics_rl_lab.rl.config import MjlabRunnerCfg
from ics_rl_lab.tasks.amp.config.pm01.agents.rl_amp_cfg import Pm01AmpFlatWalkRunnerCfg


@dataclass(kw_only=True)
class G1AmpFlatWalkRunnerCfg(Pm01AmpFlatWalkRunnerCfg):
  experiment_name: str = "g1_amp_walk"


@dataclass(kw_only=True)
class G1AmpFlatRunRunnerCfg(Pm01AmpFlatWalkRunnerCfg):
  experiment_name: str = "g1_amp_run"


def g1_amp_rl_cfg() -> MjlabRunnerCfg:
  return G1AmpFlatWalkRunnerCfg()


def g1_amp_run_rl_cfg() -> MjlabRunnerCfg:
  return G1AmpFlatRunRunnerCfg()
