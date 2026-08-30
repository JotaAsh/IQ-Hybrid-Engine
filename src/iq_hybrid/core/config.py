"""Configuration loader and settings manager for IQ-Hybrid Engine."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "solver": "greedy",
    "wide_ladder": False,
    "imatrix_method": "add",
    "imatrix_output_dir": "iMatrix",
    "verbose": False,
}

CONFIG_FILENAME = "config.json"


def load_config(config_file: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from config.json if it exists, merging with default settings.

    Code defaults are overridden by values present in config.json.
    """
    config = dict(DEFAULT_CONFIG)
    target_path = Path(config_file) if config_file else Path(CONFIG_FILENAME)

    if target_path.is_file():
        try:
            with open(target_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if k in config or k in (
                        "solver",
                        "wide_ladder",
                        "imatrix_method",
                        "imatrix_output_dir",
                        "verbose",
                    ):
                        config[k] = v
                logger.debug("Loaded configuration from %s: %s", target_path, config)
            else:
                logger.warning(
                    "Config file %s did not contain a valid JSON object. Using defaults.",
                    target_path,
                )
        except Exception as e:
            logger.warning("Failed to read config file %s (%s). Using defaults.", target_path, e)

    return config
