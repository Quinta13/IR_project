"""

Global Configuration File
-------------------------

This module contains global variables in dictionary form for:

"""

from __future__ import annotations

from model.settings import DataConfig

DATACONFIG = {
    "original": DataConfig(name="original", docs=-1, terms=-1, eps=0., n_cluster=150),
    "tiny": DataConfig(name="tiny", docs=50000, terms=15000, eps=0.3, n_cluster=150)
}
