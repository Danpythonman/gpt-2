import datetime as dt
import logging
import typing

import humanize
import torch


def get_main_logger() -> logging.Logger:
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(name)s> %(message)s'
        ))
        logger.addHandler(handler)
    return logger


def get_worker_logger(worker_number: int) -> logging.Logger:
    logger = logging.getLogger(f'worker:{worker_number}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(name)s> %(message)s'
        ))
        logger.addHandler(handler)
    return logger


def get_null_logger() -> logging.Logger:
    logger = logging.getLogger('null')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.NullHandler()
        logger.addHandler(handler)
    return logger


class TrainingLogger:

    _logger: logging.Logger
    _max_steps: int

    _total_duration_so_far: dt.timedelta
    _total_iterations_so_far: int

    def __init__(self, logger: logging.Logger, max_steps: int):
        self._logger = logger
        self._max_steps = max_steps

        self._total_duration_so_far = dt.timedelta(0)
        self._total_iterations_so_far =  0

    def log(
        self,
        current_time: dt.datetime,
        previous_time: dt.datetime,
        step: int,
        accumulated_loss: torch.types.Number,
        lr: float,
        norm: torch.types.Number
    ) -> None:
        self._logger.info(f'time: {current_time.strftime('%H:%M:%S')} | step: {step:>3} | loss: {accumulated_loss.item():9>.4f} | lr: {lr:8>.4e} | norm: {norm.item():8>.4f}')

        diff = current_time - previous_time

        self._total_duration_so_far += diff
        self._total_iterations_so_far += 1

        average_duration = self._total_duration_so_far / self._total_iterations_so_far
        steps_remaining = self._max_steps - step
        estimated_remaining_duration = steps_remaining * average_duration

        self._logger.info(f'Last iteration took {humanize.naturaldelta(diff)}, estimated time remaining: {humanize.naturaldelta(estimated_remaining_duration)}')
