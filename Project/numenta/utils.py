import logging
import randomname


def get_logger(name):
    return logging.getLogger(f"lidapy.{name}")

logger = get_logger(__name__)

random_name = lambda: randomname.get_name()