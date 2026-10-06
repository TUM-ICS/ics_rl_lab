"""PM01 flat-ground velocity-tracking locomotion task."""

from . import config  # noqa: F401
from .locomotion_env_cfg import make_locomotion_env_cfg

__all__ = ["make_locomotion_env_cfg"]
