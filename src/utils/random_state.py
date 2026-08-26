"""
Single source of truth for pseudo-random number generation.

Why this module exists
----------------------
Before centralization, seeds were hardcoded as function defaults scattered
across at least six modules, and they were not even consistent with each
other: ``42`` in ``evaluation.bootstrap`` and ``evaluation.significance``,
``28`` in ``stats.stochastic`` and parts of ``stats.distributions``. Two
consequences follow, and both are research-integrity problems rather than
style problems.

First, reproducibility was per-function rather than per-run. There was no
single value a reader could set to reproduce a full research run, so
"reproducible" could only be verified one function at a time.

Second -- and worse -- a seed buried in a default argument is a silent
researcher degree of freedom. Nothing stops a seed from being nudged until
a bootstrap p-value crosses a threshold, and nothing in the output records
that it happened. Routing every generator through :func:`get_rng` and one
module-level :data:`DEFAULT_SEED` makes the choice explicit, global, and
logged, which is what turns it back into a fixed constant.

Usage contract
--------------
Library functions take ``seed: int | None = None`` and immediately call
:func:`get_rng`. They do **not** hardcode a numeric default, and they do
**not** touch the legacy global ``np.random.seed``.

    >>> def block_bootstrap(x, seed: int | None = None):
    ...     rng = get_rng(seed)
    ...     return rng.choice(x, size=len(x))

Passing ``seed=None`` resolves to :data:`DEFAULT_SEED`, so behaviour is
deterministic by default. Passing an explicit integer overrides it, which
is what tests and sensitivity analyses do. Determinism-by-default is the
right choice for a research framework: an accidental non-seeded run that
silently produces different numbers on every invocation is a far more
expensive failure than an over-cautious fixed seed.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED: int = int(os.environ.get("QUANT_SEED", 28))


def resolve_seed(seed: int | None = None) -> int:
    """
    Resolves an optional seed argument to a concrete integer.

    Parameters
    ----------
    seed : int | None, default None
        ``None`` resolves to :data:`DEFAULT_SEED`. An explicit integer is
        returned unchanged.

    Returns
    -------
    int
        The seed actually in effect. Useful for logging and for recording
        the seed in a results manifest, so that a saved result carries the
        seed that produced it.

    Raises
    ------
    TypeError
        If `seed` is neither None nor an integer. Caught explicitly because
        ``np.random.default_rng`` accepts a surprising range of types, and
        silently accepting a float or a string here would produce a run
        that is deterministic but not reproducible from the logged value.
    """
    if seed is None:
        return DEFAULT_SEED
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            f"seed must be an int or None, got {type(seed).__name__!r}."
        )
    return int(seed)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """
    Returns an independent, seeded NumPy ``Generator``.

    This is the only sanctioned way for framework code to obtain
    randomness.

    Parameters
    ----------
    seed : int | None, default None
        Resolved through :func:`resolve_seed`.

    Returns
    -------
    np.random.Generator
        A fresh ``PCG64`` generator. Fresh matters: returning a shared
        module-level generator would make every result depend on the order
        in which functions happened to be called, so re-running one cell of
        an analysis in isolation would not reproduce it.

    Notes
    -----
    ``default_rng`` is used rather than the legacy ``np.random.seed`` /
    ``RandomState`` API. The legacy API mutates process-global state, which
    means an unrelated third-party import can consume draws and shift every
    downstream result. ``Generator`` also uses PCG64 rather than the
    Mersenne Twister, which has better statistical properties for the
    bootstrap and permutation work this framework relies on.
    """
    return np.random.default_rng(resolve_seed(seed))


def set_global_seed(seed: int | None = None) -> int:
    """
    Seeds the process-global generators of ``random`` and ``numpy.random``.

    Parameters
    ----------
    seed : int | None, default None
        Resolved through :func:`resolve_seed`.

    Returns
    -------
    int
        The seed applied, for logging.

    Notes
    -----
    Intended for entry points only, as a defensive backstop against third
    party code (``scikit-learn`` estimators left at ``random_state=None``,
    ``statsmodels`` starting-value jitter) that reaches for the legacy
    global generators where the framework cannot pass an explicit
    ``Generator``.

    Framework code must still call :func:`get_rng` and must not rely on
    this. Global seeding does not survive re-imports, does not compose
    across threads, and gives no guarantee about draw ordering -- it
    narrows non-reproducibility, it does not eliminate it.
    """
    resolved = resolve_seed(seed)
    random.seed(resolved)
    np.random.seed(resolved)
    return resolved
