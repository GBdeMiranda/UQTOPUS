"""
Reward Building Blocks

Reads functionObject output, averages it from the CFD time step onto the control
steps, and runs the user's reward function over the result.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

RewardFn = Callable[[xr.Dataset], "np.ndarray | float"]

_HEADER_NAME = re.compile(r"\S*\([^()]*\)\S*|\S+")
_TOKEN = re.compile(r"[()]|[^\s()]+")


def _column_names(header: list[str], n_columns: int) -> list[str]:
    """Names from the last header line that has one per column, 'time' first."""
    for line in reversed(header):
        names = _HEADER_NAME.findall(line.lstrip("#"))
        if len(names) == n_columns:
            return ["time"] + names[1:]
    raise ValueError(f"no header line names the {n_columns} columns")


def _cells(line: str) -> list[list[str]]:
    """The values of one data line, a parenthesized vector or tensor as one cell."""
    cells: list[list[str]] = []
    depth = 0
    for token in _TOKEN.findall(line):
        if token == "(":
            if depth == 0:
                cells.append([])
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth:
            cells[-1].append(token)
        else:
            cells.append([token])
    return cells


def _component_names(name: str, size: int) -> list[str]:
    """Variable names for a column of size values: x, y, z for a vector, else 0, 1, ..."""
    if size == 1:
        return [name]
    suffixes = ("x", "y", "z") if size == 3 else range(size)
    return [f"{name}.{suffix}" for suffix in suffixes]


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
        xr.Dataset indexed by 'time', with one variable per column, and one per
        component for a column holding a vector or tensor.
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
        rows = [_cells(line) for line in lines if line.strip() and not line.startswith("#")]
        if rows:
            sizes = [len(cell) for cell in rows[0]]
            tables.append(np.array([[float(v) for cell in row for v in cell] for row in rows]))

    table = np.vstack(tables)
    # later restarts overwrite earlier samples at the same time
    _, keep = np.unique(table[::-1, 0], return_index=True)
    table = table[::-1][keep]

    names = _column_names(header, len(sizes))
    columns = [c for name, size in zip(names, sizes) for c in _component_names(name, size)]
    return xr.Dataset(
        {column: ("time", table[:, i]) for i, column in enumerate(columns) if i > 0},
        coords={"time": table[:, 0]},
    )


def align_to_control(
    series: xr.Dataset | xr.DataArray,
    control_times: np.ndarray,
) -> xr.Dataset:
    """
    Time-average a signal sampled at CFD resolution over each control interval.

    Each sample counts for the time step that ends at it, and the first sample
    for a step as long as the second.

    Parameters:
        series (xr.Dataset or xr.DataArray): indexed by 'time'.
        control_times (np.ndarray): the end of each control interval, as the
            trajectory labels its rows.

    Returns:
        xr.Dataset indexed by 'time' at the control instants, holding the time
        average over the interval that ends at each one, or the last earlier
        sample where that interval holds none.
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
    steps = np.diff(source_times, prepend=source_times[0])
    steps[0] = steps[1] if len(steps) > 1 else 1.0

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
        means = [
            np.average(values[a:b], weights=steps[a:b]) if b > a else values[max(b - 1, 0)]
            for a, b in zip(lo, hi)
        ]
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
