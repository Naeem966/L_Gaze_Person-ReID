import logging
import os
import sys

def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Stream handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
