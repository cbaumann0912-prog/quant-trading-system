"""
Cross-cutting utilities shared by every layer of the research framework.

Contains infrastructure concerns that are deliberately kept out of the
research modules themselves: logging configuration and centralized
pseudo-random number generation. Nothing in this package may import from
``src.signals``, ``src.stats``, ``src.evaluation``, ``src.analysis``,
``src.features``, ``src.data``, or ``src.framework`` -- the dependency
arrow points one way only, so that importing a utility can never pull in
a research module or trigger a circular import.
"""

from src.utils.logging_config import configure_logging, get_logger
from src.utils.random_state import (
    DEFAULT_SEED,
    get_rng,
    resolve_seed,
    set_global_seed,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "DEFAULT_SEED",
    "get_rng",
    "resolve_seed",
    "set_global_seed",
]
