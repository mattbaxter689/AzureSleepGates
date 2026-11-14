import logging
from pathlib import Path

from loader.config_loader import load_config
from model.config.training_config import TuningConfig

logger = logging.getLogger(__name__)


def load_tuning_config(path: str | Path) -> TuningConfig:

    logger.info("Loading custom model tuning config")
    config = load_config(path, TuningConfig)
    logger.info(
        f"Tuning config successfully loaded: {config.model_dump_json(indent=2)}"
    )

    return config
