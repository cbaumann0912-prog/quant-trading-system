"""
Centralized logging configuration for the research framework.

Design contract
---------------
Library code (everything under ``src/``) only ever calls :func:`get_logger`
and emits records. It never configures handlers, never sets a level, and
never writes to stdout directly. Configuration is the exclusive
responsibility of the application entry point (``research/run_research.py``,
a notebook, or a test fixture), which calls :func:`configure_logging` once.

This is the standard library-vs-application split recommended by the
``logging`` documentation. It matters here for a specific research reason:
a module that configures logging on import will duplicate handlers every
time it is re-imported, and will hijack the root logger of any downstream
consumer. Both produce noisy, non-reproducible run logs -- and a research
log that cannot be trusted to be complete is worse than no log at all,
because it invites the reader to assume nothing was suppressed.

A ``NullHandler`` is attached to the package root so that importing ``src``
without configuring logging is silent rather than emitting the
"No handlers could be found" warning.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PACKAGE_LOGGER_NAME = "src"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """
    Returns the module-scoped logger that library code should emit through.

    Parameters
    ----------
    name : str
        Always pass ``__name__``. This yields dotted names such as
        ``src.framework.data_loader``, which lets a consumer raise or
        lower verbosity for one subsystem
        (``logging.getLogger("src.signals").setLevel(logging.DEBUG)``)
        without touching the rest of the framework.

    Returns
    -------
    logging.Logger
        A logger with no handlers of its own. Records propagate up to the
        ``src`` logger, and from there to whatever :func:`configure_logging`
        installed.

    Notes
    -----
    Calling this at module scope is safe and cheap: ``logging.getLogger``
    is idempotent and returns the same object for the same name, so repeated
    imports do not accumulate loggers.
    """
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    capture_warnings: bool = True,
) -> logging.Logger:
    """
    Configures handlers on the ``src`` package logger. Call once, from an
    application entry point -- never from library code.

    Parameters
    ----------
    level : int | str, default ``logging.INFO``
        Threshold for the package logger. Accepts either a ``logging``
        constant or its name as a string (``"DEBUG"``).
    log_file : str | Path | None, default None
        If given, records are additionally written to this path in append
        mode, and parent directories are created. A persisted log is what
        makes a research run auditable after the fact: it records the
        parameters, the row counts, and the failures that the results
        tables alone do not show.
    fmt, datefmt : str
        Standard ``logging`` format strings.
    capture_warnings : bool, default True
        Routes ``warnings.warn`` through logging. Recommended for research
        runs, because ``statsmodels``, ``scipy``, and ``pandas`` signal
        convergence failures and silent dtype coercions through the
        warnings machinery. Left on stderr those scroll past unrecorded;
        captured, they land in the run log next to the result they
        contaminated.

    Returns
    -------
    logging.Logger
        The configured ``src`` package logger.

    Notes
    -----
    Idempotent. Existing handlers are removed before new ones are attached,
    so calling this twice (for example, once in a notebook cell that is
    re-run) does not duplicate every log line.

    Propagation to the root logger is disabled so that the framework's
    records do not appear twice when the host application has also called
    ``logging.basicConfig``.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if capture_warnings:
        logging.captureWarnings(True)
        py_warnings = logging.getLogger("py.warnings")
        py_warnings.handlers = list(logger.handlers)
        py_warnings.setLevel(level)

    return logger
