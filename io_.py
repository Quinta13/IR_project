"""
Input / Output Functions
----------------------

This module contains input/output functions for various tasks
 such as (logging, directory and files path handling, file operations, ...)

"""

from __future__ import annotations

import json
import os
from os import path
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

"""
Global Logging Configuration

This boolean flag controls whether general logging is enabled or disabled,
 Use this flag to control the level of general information logging in the module.
 
- LOG` (bool): Set to `True` to enable general logging, and `False` to disable it.
- `LOG_IO` (bool): Set to `True` to enable I/O specific logging, and `False` to disable it.

"""

LOG: bool = False
LOG_IO: bool = False


# DIRECTORIES AND FILES

"""
Directory Names

This dictionary defines common directory names used in the module for different purposes.

Attributes:
- `dataset` (str): Name of the directory containing the RCV1 dataset.
"""

DIR_NAMES: Dict[str, str] = {
    "dataset": "dataset"
}


"""
File Names

This dictionary defines common file names used in the module for different purposes (including their extension).
"""

FILES: Dict[str, str] = {
    "data": "data.npz",
    "sorted": "data_sorted.npz"
}

# LOGGING FUNCTIONS


def log(info: str):
    """
    Log the provided information if general logging is enabled.

    :param info: information to be logged.
    """
    if LOG:
        print(f"INFO: {info}")


def log_io(info: str):
    """
    Log the provided input/output information if I/O logging is enabled.

    :param info: information to be logged.
    """

    if LOG_IO:
        print(f"I/O: {info}")


# DIRECTORY FUNCTIONS

def make_dir(path_: str):
    """
    Create a directory at the specified path if it does not already exist.

    :param path_: Path to the directory to be created.
    """

    if os.path.exists(path_):
        # directory already exists
        log_io(f"Directory {path_} already exists ")
    else:
        os.makedirs(path_)
        # directory doesn't exist
        log_io(f"Created directory {path_}")


def get_root_dir() -> str:
    """
    Get the path to project root directory.

    :return: path to the root directory.
    """

    return str(path.abspath(path.join(__file__, "../")))


def get_dataset_dir() -> str:
    """
    Get dataset directory.

    :return: path to the dataset directory.
    """

    return path.join(get_root_dir(), DIR_NAMES["dataset"])


def get_collection_dir(collection_name: str) -> str:
    """
    Get sample directory for given sample name.

    :param collection_name: sample name.
    :return: path to the sample directory.
    """

    return path.join(get_dataset_dir(), collection_name)


# FILES


def get_data_fp() -> str:
    """
    Get data file.

    :return: path to data file.
    """

    return path.join(get_dataset_dir(), FILES["data"])


def get_data_sorted_fp() -> str:
    """
    Get sorted data file.

    :return: path to data file.
    """

    return path.join(get_dataset_dir(), FILES["sorted"])


# OPERATIONS

SAVE = True

def load_json(path_: str) -> Dict | List:
    """
    Load a dictionary or list from a local JSON file.

    :param path_: path to the JSON file to be read.
    :return: loaded JSON object (dictionary or list).
    """

    log_io(info=f"Loading {path_} ")

    with open(path_) as json_file:
        return json.load(fp=json_file)


def store_json(path_: str, obj: Dict | List):
    """
    Store a dictionary or list as a JSON file at the specified path.

    :param path_: path for the JSON file (to be created or overwritten).
    :param obj: JSON object (dictionary or list) to be stored.
    """

    json_string = json.dumps(obj=obj)

    # Check if file already existed
    info_op = "Overwriting" if path.exists(path_) else "Saving"

    log_io(info=f"{info_op} {path_} ")

    with open(path_, 'w') as json_file:
        json_file.write(json_string)


def load_sparse_matrix(path_: str) -> csr_matrix:
    """
    Load a sparse matrix from disk.

    :param path_: path to the .npz file to be read.
    :return: loaded sparse matrix.
    """

    log_io(info=f"Loading {path_}. ")

    return load_npz(file=path_)


def store_sparse_matrix(path_: str, mat: csr_matrix):
    """
    Store a sparse matrix as a .npz file at the specified path.

    :param path_: path for the .npy file (to be created or overwritten).
    :param mat: sparse matrix to be stored.
    """

    # Check if file already existed
    info_op = "Overwriting" if path.exists(path_) else "Saving"

    log_io(info=f"{info_op} {path_}. ")

    save_npz(file=path_, matrix=mat)


def load_dense_matrix(path_: str) -> np.ndarray:
    """
    Load a dense matrix from disk.

    :param path_: path to the .npy file to be read.
    :return: loaded dense matrix.
    """

    log_io(info=f"Loading {path_}. ")

    return np.load(file=path_)


def store_dense_matrix(path_: str, mat: np.ndarray):
    """
    Store a dense matrix as a .npy file at the specified path.

    :param path_: path for the .npy file (to be created or overwritten).
    :param mat: dense matrix to be stored.
    """

    # Check if file already existed
    info_op = "Overwriting" if path.exists(path_) else "Saving"

    log_io(info=f"{info_op} {path_}. ")

    np.save(file=path_, arr=mat)
