"""General use functions."""

import time, os, glob
import torch as th
import numpy as np
from collections.abc import Mapping

################################################################################


def sync(i: int, start_time: float, timestep: float) -> None:
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > 0.04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i * timestep):
            time.sleep(timestep * i - elapsed)


def recursive_dict_update(d: dict, u: dict) -> dict:
    """Recursively update a dictionary with another dictionary.

    Parameters
    ----------
    d : dict
        The original dictionary to be updated.
    u : dict
        The dictionary with new values to update the original dictionary.

    Returns
    -------
    d : dict
        The updated dictionary.

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            if not isinstance(d.get(k), Mapping):
                d[k] = {}
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def recursive_enum_mapping(d: dict, m: dict) -> dict:
    """Recursively map enum values in the dictionary.

    Parameters
    ----------
    d : dict
        The original dictionary to be updated.
    m : dict
        The dictionary with enum mappings.

    Returns
    -------
    d : dict
        The updated dictionary with enum values.

    """
    for k, v in d.items():
        if isinstance(v, Mapping):
            d[k] = recursive_enum_mapping(d.get(k, {}), m)
        elif k in m:
            d[k] = getattr(m[k], v, None)
            if d[k] is None:
                raise ValueError(f"Invalid value '{v}' for key '{k}'.")
    return d


def fix_none_values(d: dict) -> None:
    """Recursively fix 'None' string values in the dictionary.

    Parameters
    ----------
    d : dict
        The original dictionary to be updated.

    """
    for k, v in d.items():
        if isinstance(v, dict):
            fix_none_values(v)
        elif v == "None":
            d[k] = None


def set_seed(seed: int) -> None:
    """Set the seed for numpy and torch.

    Parameters
    ----------
    seed : int
        The seed value to be set.

    """
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def get_latest_run_id(log_path: str = "", log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: Path to the log folder containing several runs.
    :param log_name: Name of the experiment. Each run is stored
        in a folder named ``log_name_1``, ``log_name_2``, ...
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id
