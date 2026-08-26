"""
Quantitative research framework.

Layered so that dependencies point one way only:

    utils      -> (nothing)
    stats      -> utils
    data       -> utils
    features   -> stats, utils
    signals    -> features, stats, utils
    evaluation -> stats, utils
    analysis   -> stats, utils
    framework  -> all of the above

`framework` owns data loading and walk-forward orchestration; `signals`
owns strategy logic; `evaluation` owns the statistical validation that
decides whether a signal survives.

Logging is configured by the application entry point, never on import --
see `src.utils.logging_config`.
"""
