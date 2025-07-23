# gulps/logging_config.py
import logging

logger = logging.getLogger("gulps")
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Optional: suppress propagation to root logger
logger.propagate = False
