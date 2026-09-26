"""
Reward Building Blocks

Reads functionObject output, averages it from the CFD time step onto the control
steps, and runs the user's reward function over the result.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

RewardFn = Callable[[xr.Dataset], "np.ndarray | float"]


def _column_names(header: list[str], n_columns: int) -> list[str]:
    """Names from the last header line that has one per column, 'time' first."""
    for line in reversed(header):
        names = line.lstrip("#").split()
        if len(names) == n_columns:
            return ["time"] + names[1:]
    raise ValueError(f"no header line names the {n_columns} columns")


def read_function_object(
    case_dir: str | Path,
    name: str,
    file: str | None = None,
) -> xr.Dataset:
    """
    Read the output of one functionObject into an xr.Dataset.

    The time directories a case accumulates across restarts are concatenated;
    where they overlap, the sample from the latest start time wins.

    Parameters:
        case_dir (str or Path): the OpenFOAM case.
        name (str): functionObject name, i.e. the directory under postProcessing.
        file (str or None): which file to read when the functionObject writes
            more than one. None reads the only file.

    Returns:
        xr.Dataset indexed by 'time', with one variable per column.
    """
    root = Path(case_dir) / "postProcessing" / name
    time_dirs = sorted(root.iterdir(), key=lambda d: float(d.name))

    if file is None:
        files = {p.name for d in time_dirs for p in d.iterdir()}
        if len(files) != 1:
            raise ValueError(f"{root} holds {sorted(files)}; pass file= to choose one")
        file = files.pop()

    tables = []
    for directory in time_dirs:
        path = directory / file
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        header = [line for line in lines if line.startswith("#")]
        rows = [line.split() for line in lines if line.strip() and not line.startswith("#")]
        if rows:
            tables.append(np.asarray(rows, dtype=np.float64))

    table = np.vstack(tables)
    # later restarts overwrite earlier samples at the same time
    _, keep = np.unique(table[::-1, 0], return_index=True)
    table = table[::-1][keep]

    columns = _column_names(header, table.shape[1])
    return xr.Dataset(
        {column: ("time", table[:, i]) for i, column in enumerate(columns) if i > 0},
        coords={"time": table[:, 0]},
    )


def align_to_control(
    series: xr.Dataset | xr.DataArray,
    control_times: np.ndarray,
) -> xr.Dataset:
    """
    Average a signal sampled at CFD resolution over each control interval.

    Parameters:
        series (xr.Dataset or xr.DataArray): indexed by 'time'.
        control_times (np.ndarray): the end of each control interval, as the
            trajectory labels its rows.

    Returns:
        xr.Dataset indexed by 'time' at the control instants, holding the mean
        over the interval that ends at each one, or the last earlier sample
        where that interval holds none.
    """
    if isinstance(series, xr.DataArray):
        series = series.to_dataset(name=series.name or "value")

    control_times = np.asarray(control_times, dtype=np.float64)
    source_times = series["time"].values
    interval = (
        np.diff(control_times).mean()
        if len(control_times) > 1
        else control_times[0] - source_times[0]
    )
    starts = np.concatenate(([control_times[0] - interval], control_times[:-1]))
    lo = np.searchsorted(source_times, starts, side="right")
    hi = np.searchsorted(source_times, control_times, side="right")

    empty = hi <= lo
    if np.any(empty):
        logger.warning(
            "%d of %d control intervals contain no samples; holding the previous "
            "sample there",
            int(empty.sum()),
            len(control_times),
        )

    reduced = {}
    for variable, data in series.data_vars.items():
        values = data.values.astype(np.float64)
        means = [values[a:b].mean() if b > a else values[max(b - 1, 0)] for a, b in zip(lo, hi)]
        reduced[variable] = ("time", np.array(means))
    return xr.Dataset(reduced, coords={"time": control_times})


def attach(trajectory: xr.Dataset, *series: xr.Dataset) -> xr.Dataset:
    """
    Merge signals, averaged onto the control times, into a trajectory: the
    dataset the reward function receives.
    """
    merged = trajectory
    for item in series:
        merged = merged.merge(align_to_control(item, trajectory["time"].values))
    return merged


def evaluate_reward(trajectory: xr.Dataset, reward_fn: RewardFn) -> np.ndarray:
    """
    Run a user reward function over a trajectory.

    Returns:
        np.ndarray of shape (n_steps,), one finite reward per control step.
    """
    n_steps = trajectory.sizes["time"]
    rewards = np.asarray(reward_fn(trajectory), dtype=np.float64).ravel()
    if rewards.size != n_steps:
        raise ValueError(
            f"the reward function returned {rewards.size} values for {n_steps} "
            "control steps; PPO needs one reward per step"
        )
    if not np.all(np.isfinite(rewards)):
        raise ValueError(
            f"the reward function returned {int((~np.isfinite(rewards)).sum())} "
            "non-finite value(s)"
        )
    return rewards
