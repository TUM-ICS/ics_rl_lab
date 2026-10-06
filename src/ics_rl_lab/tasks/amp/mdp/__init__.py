"""AMP MDP terms: mjlab stock + locomotion's custom terms (reused, not duplicated) + AMP-only ones."""

from mjlab.envs.mdp import *  # noqa: F401, F403

from ics_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F403
from .events import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
