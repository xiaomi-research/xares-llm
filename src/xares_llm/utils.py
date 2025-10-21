from loguru import logger
import torch
import random
import numpy as np


def attr_from_py_path(path: str, endswith: str | None = None) -> type:
    from importlib import import_module

    module_name = path.replace("/", ".")
    # Strip ending
    if module_name.endswith(".py"):
        module_name = module_name[:-3]  # Remove last 3 characters (".py")

    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        logger.exception(f"Module not found {module_name}")
        raise ValueError(f"Module not found: {module_name}")

    attr_list = [m for m in dir(module) if not endswith or m.endswith(endswith)]
    if len(attr_list) != 1:
        raise ValueError(f"Expected 1 class with endswith={endswith}, got {len(attr_list)}")

    return getattr(module, attr_list[0])

def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    logger.debug(f"Setting global seed to {seed}...")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def setup_global_logger():
    import sys

    # Make the logger with this format the default for all loggers in this package
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "format": "<fg #FFA903>(X-ARES-LLM)</fg #FFA903> [<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow>] "
                "<level>{message}</level>",
                "level": "DEBUG",
                "colorize": True,
            }
        ]
    )
    logger.level("ERROR", color="<red>")
    logger.level("INFO", color="<white>")
