"""
Logging utils
=======
src/cmo/logging.py
"""

import logging
import os
from typing import Optional, Union

from rich.logging import RichHandler


class Logger:
    """
    Logger utility class to configure and get loggers.

    Usage
    -----
    ```python
    Logger.configure(level="DEBUG")
    logger = Logger.get("cmo")
    logger.debug("message")
    ```
    """

    _configured: set[str] = set()

    @classmethod
    def configure(
        cls,
        level: Optional[Union[str, int]] = None,
        name: str = "cmo",
        force: bool = False,
    ) -> logging.Logger:
        """
        Configure logger with given name.
        No-op if already configured unless `force=True`.

        Parameters:
            level: `int` or `str` name (E.g. "DEBUG", "INFO") or `None` (uses `CMO_LOG_LEVEL` env or `WARNING`).
            name: `str` name of the logger.
            force: `bool`, if `True`, re-configure even if already configured.
        """
        logger = logging.getLogger(name)
        lvl = level if level is not None else os.getenv("CMO_LOG_LEVEL", "WARNING")

        if logger.handlers and not force:
            logger.setLevel(lvl)
            return logger

        handler = RichHandler(
            show_time=False, show_path=False, markup=True, rich_tracebacks=True
        )
        fmt = "%(message)s"
        handler.setFormatter(logging.Formatter(fmt))

        if force:
            for h in logger.handlers:
                logger.removeHandler(h)

        logger.addHandler(handler)
        logger.setLevel(lvl)
        logger.propagate = False

        cls._configured.add(name)
        return logger

    @classmethod
    def get(cls, name: str = "cmo", ensure_configured: bool = True) -> logging.Logger:
        """
        Return the `stdlib.Logger` for `name`.
        If `ensure_configured=True`, will call `configure()` with default settings when logger is not yet configured.
        """
        logger = logging.getLogger(name)
        if ensure_configured and name not in cls._configured:
            cls.configure(name=name)
        return logger


logger = Logger.configure()
