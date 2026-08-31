"""
Reward Building Blocks

Reads functionObject output, aligns it from the CFD time step onto the control
steps, and runs the user's reward function over the result.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

POSTPROCESSING = "postProcessing"

RewardFn = Callable[[xr.Dataset], "np.ndarray | float"]


# Reading OpenFOAM functionObject output

def _parse_column_names(header_lines: list[str], n_columns: int) -> list[str]:
    """
    Recover column names from an OpenFOAM functionObject header.

    The convention is that the last comment line holds the names, e.g.
    '# Time  Cd  Cd(f)  Cd(r)  Cl ...'. Falls back to positional names when the
    header is absent or does not line up.
    """
    for line in reversed(header_lines):
        names = line.lstrip("#").replace("\t", " ").split()
        if len(names) == n_columns:
            return ["time" if n.lower() == "time" else n for n in names]

    logger.debug("Could not read column names from header; using positional names")
    return ["time"] + [f"c{i}" for i in range(1, n_columns)]


def read_function_object(
    case_dir: str | Path,
    name: str,
    file: str | None = None,
) -> xr.Dataset:
    """
    Read the output of one functionObject into an xr.Dataset.

    Handles the several time directories a case accumulates across restarts:
    they are concatenated and, where they overlap, the sample from the latest
    start time wins.

    Parameters:
        case_dir (str or Path): the OpenFOAM case.
        name (str): functionObject name, i.e. the directory under postProcessing.
        file (str or None): which file to read when the functionObject writes
            more than one. None reads the only file, and raises if ambiguous.

    Returns:
        xr.Dataset indexed by 'time', with one variable per column.
    """
    root = Path(case_dir) / POSTPROCESSING / name
    if not root.is_dir():
        raise ValueError(f"no functionObject output at {root}")

    time_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir()),
        key=lambda d: float(d.name) if _is_number(d.name) else float("inf"),
    )
    if not time_dirs:
        raise ValueError(f"{root} has no time directories")

    if file is None:
        files = {p.name for d in time_dirs for p in d.iterdir() if p.is_file()}
        if len(files) != 1:
            raise ValueError(
                f"{root} holds several files {sorted(files)}; pass file= to choose"
            )
        file = files.pop()

    frames: list[tuple[list[str], np.ndarray]] = []
    for directory in time_dirs:
        path = directory / file
        if not path.exists():
            continue
        header: list[str] = []
        rows: list[list[float]] = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                header.append(line)
            else:
                rows.append([float(v) for v in line.replace("\t", " ").split()])
        if rows:
            table = np.asarray(rows, dtype=np.float64)
            frames.append((_parse_column_names(header, table.shape[1]), table))

    if not frames:
        raise ValueError(f"no data rows in any {file} under {root}")

    columns = frames[0][0]
    if any(names != columns for names, _ in frames):
        raise ValueError(f"inconsistent columns across time directories in {root}")

    table = np.vstack([t for _, t in frames])
    # later restarts overwrite earlier samples at the same time
    _, keep = np.unique(table[::-1, 0], return_index=True)
    table = table[::-1][keep]

    return xr.Dataset(
        {name: ("time", table[:, i]) for i, name in enumerate(columns) if i > 0},
        coords={"time": table[:, 0]},
        attrs={"source": str(root / file)},
    )


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# Aligning a CFD-resolution signal onto control steps

def align_to_control(
    series: xr.Dataset | xr.DataArray,
    control_times: np.ndarray,
    *,
    how: Literal["mean", "last", "interp"] = "mean",
    interval: float | None = None,
) -> xr.Dataset:
    """
    Resample a signal sampled at CFD resolution onto the control times.

    Parameters:
        series (xr.Dataset or xr.DataArray): indexed by 'time'.
        control_times (np.ndarray): the trajectory's control instants.
        how: 'mean' averages over the interval ending at each control time,
            which is what a reward normally wants: what happened since the
            previous decision. 'last' takes the final sample of that interval.
            'interp' interpolates at the instant itself.
        interval (float or None): length of the interval preceding the first
            control time. None infers it from the spacing of control_times.

    Returns:
        xr.Dataset indexed by 'time' at the control instants.
    """
    if isinstance(series, xr.DataArray):
        series = series.to_dataset(name=series.name or "value")

    control_times = np.asarray(control_times, dtype=np.float64).ravel()
    if control_times.size == 0:
        raise ValueError("control_times is empty")

    source_times = np.asarray(series["time"].values, dtype=np.float64)
    if source_times.size == 0:
        raise ValueError("the series has no samples")

    if how == "interp":
        return series.interp(time=control_times)

    if interval is None:
        interval = (
            float(np.diff(control_times).mean())
            if control_times.size > 1
            else float(control_times[0] - source_times[0]) or 1.0
        )

    starts = np.concatenate(([control_times[0] - interval], control_times[:-1]))
    lo = np.searchsorted(source_times, starts, side="right")
    hi = np.searchsorted(source_times, control_times, side="right")

    empty = hi <= lo
    if np.any(empty):
        logger.warning(
            "%d of %d control intervals contain no samples; falling back to the "
            "nearest earlier sample there",
            int(empty.sum()),
            len(control_times),
        )

    reduced = {}
    for variable, data in series.data_vars.items():
        values = np.asarray(data.values, dtype=np.float64)
        out = np.empty(len(control_times), dtype=np.float64)
        for k, (a, b) in enumerate(zip(lo, hi)):
            if b > a:
                window = values[a:b]
                out[k] = window.mean() if how == "mean" else window[-1]
            else:
                out[k] = values[min(max(b - 1, 0), len(values) - 1)]
        reduced[variable] = ("time", out)

    return xr.Dataset(reduced, coords={"time": control_times}, attrs=dict(series.attrs))


def attach(trajectory: xr.Dataset, *series: xr.Dataset, **named: xr.Dataset) -> xr.Dataset:
    """
    Merge signals aligned to the control times into a trajectory.

    The result is what gets handed to the reward function: one dataset holding
    the observations, the actions and whatever else the reward needs.
    """
    merged = trajectory
    for item in series:
        merged = merged.merge(align_to_control(item, trajectory["time"].values))
    for prefix, item in named.items():
        aligned = align_to_control(item, trajectory["time"].values)
        merged = merged.merge(aligned.rename({v: f"{prefix}.{v}" for v in aligned.data_vars}))
    return merged


# Utilities rewards tend to need

def moving_average(
    values: xr.DataArray | np.ndarray,
    window: int,
    *,
    center: bool = False,
) -> np.ndarray:
    """
    Trailing moving average, with the leading steps averaged over what exists.

    A reward built on an oscillating quantity usually has to average over a
    period, otherwise it mostly measures the phase of the oscillation rather
    than the effect of the action.

    Parameters:
        values: the series to smooth.
        window (int): number of samples in the window.
        center (bool): center the window instead of trailing it. Only valid
            offline; a reward used during training must stay causal.
    """
    array = np.asarray(
        values.values if isinstance(values, xr.DataArray) else values, dtype=np.float64
    ).ravel()
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return array.copy()

    cumulative = np.concatenate(([0.0], np.cumsum(array)))
    out = np.empty_like(array)
    for i in range(array.size):
        if center:
            a = max(0, i - window // 2)
            b = min(array.size, a + window)
        else:
            a, b = max(0, i - window + 1), i + 1
        out[i] = (cumulative[b] - cumulative[a]) / (b - a)
    return out


def evaluate_reward(trajectory: xr.Dataset, reward_fn: RewardFn) -> np.ndarray:
    """
    Run a user reward function over a trajectory.

    Raises if the result is not finite or does not have one value per
    control step.

    Returns:
        np.ndarray of shape (n_steps,), one reward per control step.
    """
    n_steps = trajectory.sizes["time"]
    result = reward_fn(trajectory)

    rewards = np.asarray(result, dtype=np.float64).ravel()
    if rewards.size == 1 and n_steps != 1:
        raise ValueError(
            f"the reward function returned a single value for {n_steps} control "
            "steps; PPO needs one reward per step"
        )
    if rewards.size != n_steps:
        raise ValueError(
            f"the reward function returned {rewards.size} values for {n_steps} "
            "control steps"
        )
    if not np.all(np.isfinite(rewards)):
        raise ValueError(
            f"the reward function returned {int((~np.isfinite(rewards)).sum())} "
            "non-finite value(s)"
        )
    return rewards
