"""Tracking MDP terms: mjlab stock + locomotion ports (reused) + reference-motion terms."""

from ics_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
