"""

Global Configuration File
-------------------------
"""

from __future__ import annotations

from model.solver import DataConfig

DATACONFIG = {
    "original": DataConfig(name="original", docs=-1, terms=-1, eps=0., n_cluster=150),
    "sample": DataConfig(name="tiny", docs=50000, terms=15000, eps=0.3, n_cluster=150),
    "heavy": DataConfig(name="heavy", docs=100000, terms=-1, eps=0.35, n_cluster=175)
}
