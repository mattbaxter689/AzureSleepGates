from pathlib import Path
from typing import TypeVar
import logging

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# Declare generic loader for configs in pipeline
def load_config(config_path: str | Path, model: type[T]) -> T:
    """
    Load a YAML file and validate it against the given Pydantic model.
    """
    logger.info(f"Attempting to load {model.__name__} from {config_path}")
    path = Path(config_path)

    if not path.is_absolute():
        logger.info("Path is not absolute - creating absolute path to access")
        path = Path.cwd() / path

    if not path.exists():
        logger.error(f"Config file not found for {model.__name__}: {path}")
        raise FileNotFoundError(f"Config file not found for {model.__name__}: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return model.model_validate(data)
