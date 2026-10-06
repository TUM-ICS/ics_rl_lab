"""Hoverboard MDP terms: mjlab stock + locomotion's custom terms (reused, not duplicated)
+ hoverboard-only ones. The board's balancer action term lives with the asset
(`ics_rl_lab.assets.hoverboard.hoverboard_driver`), not here."""

from mjlab.envs.mdp import *  # noqa: F401, F403

from ics_rl_lab.tasks.amp.mdp.rewards import ang_vel_xy_scaled_l2  # noqa: F401
from ics_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

from .events import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
