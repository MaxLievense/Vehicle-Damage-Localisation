from unittest.mock import MagicMock

from omegaconf import DictConfig


# pylint: disable=too-few-public-methods
class MockSelf:
    """Mock class for testing purposes."""

    def __init__(self, cfg: dict):
        """
        Initializes the MockSelf object.

        Args:
            cfg (dict): Configuration dictionary to emulate Hydra's DictConfig.
        """
        self.cfg = DictConfig(cfg)
        self.log = MagicMock()
