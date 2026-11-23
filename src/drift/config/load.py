import logging
from pathlib import Path

from drift.config.drift_config import DriftConfig
from loader.config_loader import load_config

logger = logging.getLogger(__name__)


def load_drift_config(path: str | Path) -> DriftConfig:

    logger.info("Loading custom model tuning config")
    config = load_config(path, DriftConfig)
    logger.info(
        f"Tuning config successfully loaded: {config.model_dump_json(indent=2)}"
    )

    return config
