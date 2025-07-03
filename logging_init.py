import logging


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
