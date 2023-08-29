"""

Global Configuration File
-------------------------

This module contains global variables in dictionary form for:

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    docs: int
    terms: int
    eps: float


DATASETS = {
    "tiny": Dataset(name="tiny", docs=10000, terms=8000, eps=0.35)
}
