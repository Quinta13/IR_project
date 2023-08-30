"""

Global Configuration File
-------------------------

This module contains global variables in dictionary form for:

"""

from __future__ import annotations

from model.dataset import DataConfig

DATASETS_CONFIG = {
    "original": DataConfig(name="original", docs=-1, terms=-1, eps=0.),
    "tiny": DataConfig(name="tiny", docs=10000, terms=8000, eps=0.35)
}
