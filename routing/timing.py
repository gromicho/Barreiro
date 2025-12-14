# routing/timing.py

import logging
import time
from contextlib import contextmanager


@contextmanager
def timeblock(label: str, log_list: list[str]) -> None:
    """
    Measure and log the execution time of a code block.

    Args:
        label: Text label describing the block.
        log_list: List that collects human-readable timing lines.
    """
    start_s = time.perf_counter()
    logging.info('START: %s', label)
    try:
        yield
    finally:
        delta_s = time.perf_counter() - start_s
        msg = f'{label}: {delta_s:.3f} seconds'
        logging.info('END:   %s', msg)
        log_list.append(msg)
