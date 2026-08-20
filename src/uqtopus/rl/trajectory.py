"""
Trajectory Files

The solver -> Python channel: one row per control step holding the time, the
observation as fed to the network, and the action applied. Follows the OpenFOAM
postProcessing convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import xarray as xr

from .spec import CONTRACT_VERSION, PolicySpec

logger = logging.getLogger(__name__)

TRAJECTORY_SUBDIR = "postProcessing/uqtopusPolicy"
TRAJECTORY_NAME = "trajectory.dat"


class TrajectoryError(Exception):
    """Raised when a trajectory file does not match the contract."""


# Writing (test fixtures, and standing in for the solver before it exists)

def write_trajectory(
    path: str | Path,
    spec: PolicySpec,
    times: Sequence[float] | np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int | None = None,
) -> Path:
    """
    Write a trajectory file in the canonical format.

    This exists so the reader, the reward functions and the PPO loop can be
    developed and tested against realistic files before the solver-side
    controller is written. It is also the reference the C++ implementation must
    reproduce byte for byte.

    Parameters:
        path (str or Path): destination file.
        spec (PolicySpec): the contract the trajectory belongs to.
        times (sequence of float): control times, strictly increasing.
        observations (np.ndarray): shape (n_steps, obs_dim).
        actions (np.ndarray): shape (n_steps, act_dim).
        seed (int or None): RNG seed the solver sampled actions with.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times, dtype=np.float64).ravel()
    observations = np.atleast_2d(np.asarray(observations, dtype=np.float64))
    actions = np.asarray(actions, dtype=np.float64).reshape(len(times), -1)

    if observations.shape != (len(times), spec.obs_dim):
        raise ValueError(
            f"observations must have shape {(len(times), spec.obs_dim)}, "
            f"got {observations.shape}"
        )
    if actions.shape != (len(times), spec.act_dim):
        raise ValueError(
            f"actions must have shape {(len(times), spec.act_dim)}, "
            f"got {actions.shape}"
        )

    columns = spec.trajectory_columns()[:-1]  # 'seed' lives in the header
    header = [
        "# uqtopus trajectory",
        f"# contractVersion  {spec.contract_version}",
        f"# specHash         {spec.hash}",
        f"# seed             {'' if seed is None else seed}",
        f"# columns          {' '.join(columns)}",
    ]

    lines = list(header)
    for t, obs, act in zip(times, observations, actions):
        values = " ".join(f"{v:.10g}" for v in (t, *obs, *act))
        lines.append(values)

    path.write_text("\n".join(lines) + "\n")
    return path


# Reading

def _parse_header(lines: Sequence[str]) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in lines:
        body = line.lstrip("#").strip()
        if not body or " " not in body:
            continue
        key, _, value = body.partition(" ")
        header[key] = value.strip()
    return header


def default_reader(path: Path) -> tuple[dict[str, str], np.ndarray]:
    """
    Parse the canonical trajectory format into (header, table).

    Replace this with your own callable if the solver writes something else;
    it must return the same pair, with the table shaped (n_steps, n_columns).
    """
    header_lines: list[str] = []
    rows: list[list[float]] = []

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_lines.append(line)
                continue
            rows.append([float(v) for v in line.split()])

    if not rows:
        raise TrajectoryError(f"{path} contains no data rows")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise TrajectoryError(
            f"{path} has rows of differing widths: {sorted(widths)}"
        )

    return _parse_header(header_lines), np.asarray(rows, dtype=np.float64)


def find_trajectory(case_dir: str | Path) -> Path:
    """
    Locate the trajectory file inside an OpenFOAM case directory.

    Looks under postProcessing/uqtopusPolicy/<time>/trajectory.dat and returns
    the one from the latest start time, matching how OpenFOAM organizes
    functionObject output across restarts.
    """
    case_dir = Path(case_dir)
    root = case_dir / TRAJECTORY_SUBDIR
    if not root.is_dir():
        raise TrajectoryError(f"no {TRAJECTORY_SUBDIR} directory in {case_dir}")

    candidates = sorted(root.glob(f"*/{TRAJECTORY_NAME}"))
    if not candidates:
        raise TrajectoryError(f"no {TRAJECTORY_NAME} found under {root}")

    def start_time(p: Path) -> float:
        try:
            return float(p.parent.name)
        except ValueError:
            return float("inf")

    return max(candidates, key=start_time)


def read_trajectory(
    source: str | Path,
    spec: PolicySpec | None = None,
    *,
    reader: Callable[[Path], tuple[dict[str, str], np.ndarray]] | None = None,
    strict: bool = True,
) -> xr.Dataset:
    """
    Read one episode into an xr.Dataset.

    Parameters:
        source (str or Path): the trajectory file, or a case directory to search.
        spec (PolicySpec or None): contract to validate against. When given, the
            spec hash recorded by the solver must match, and the widths must
            agree with obs_dim and act_dim.
        reader (callable or None): custom parser returning (header, table). None
            uses the canonical format.
        strict (bool): raise on contract mismatch instead of warning.

    Returns:
        xr.Dataset with 'observation' (time, obs_component) and 'action'
        (time, act_component), and the spec hash and seed in attrs.
    """
    path = Path(source)
    if path.is_dir():
        path = find_trajectory(path)

    header, table = (reader or default_reader)(path)

    recorded_hash = header.get("specHash", "")
    if spec is not None and recorded_hash and recorded_hash != spec.hash:
        message = (
            f"{path} was written for contract {recorded_hash} but the caller "
            f"expects {spec.hash}; the case dictionary and the policy disagree"
        )
        if strict:
            raise TrajectoryError(message)
        logger.warning(message)

    version = header.get("contractVersion", "")
    if version and version != CONTRACT_VERSION:
        logger.warning(
            "%s declares contract version %s, this build expects %s",
            path,
            version,
            CONTRACT_VERSION,
        )

    times = table[:, 0]
    if np.any(np.diff(times) <= 0):
        raise TrajectoryError(f"{path} has non-increasing control times")

    if spec is not None:
        expected = 1 + spec.obs_dim + spec.act_dim
        if table.shape[1] != expected:
            raise TrajectoryError(
                f"{path} has {table.shape[1]} columns but the spec implies "
                f"{expected} (1 time + {spec.obs_dim} obs + {spec.act_dim} action)"
            )
        obs_dim, act_dim = spec.obs_dim, spec.act_dim
        obs_names = spec.observation.component_names()
        act_names = spec.action.component_names()
    else:
        columns = header.get("columns", "").split()
        if not columns:
            raise TrajectoryError(
                f"{path} has no 'columns' header entry; pass a spec to read it"
            )
        if len(columns) != table.shape[1]:
            raise TrajectoryError(
                f"{path} declares {len(columns)} columns but rows have "
                f"{table.shape[1]} values"
            )
        # without a spec, everything between time and the last column is
        # treated as observation
        obs_dim = table.shape[1] - 2
        act_dim = 1
        obs_names = columns[1 : 1 + obs_dim]
        act_names = columns[1 + obs_dim :]

    observations = table[:, 1 : 1 + obs_dim]
    actions = table[:, 1 + obs_dim : 1 + obs_dim + act_dim]

    if not np.all(np.isfinite(table)):
        raise TrajectoryError(f"{path} contains non-finite values")

    seed = header.get("seed", "")
    dataset = xr.Dataset(
        data_vars={
            "observation": (("time", "obs_component"), observations),
            "action": (("time", "act_component"), actions),
        },
        coords={
            "time": times,
            "obs_component": obs_names,
            "act_component": act_names,
        },
        attrs={
            "spec_hash": recorded_hash,
            "seed": int(seed) if seed else -1,
            "source": str(path),
            "n_steps": int(len(times)),
        },
    )
    return dataset


def concat_trajectories(trajectories: Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Stack episodes along a new 'episode' dimension.

    Episodes of differing length are padded with NaN, and a boolean 'valid'
    variable marks the real steps, so a run that diverged early stays in the
    batch instead of forcing the whole rollout to be discarded.
    """
    if not trajectories:
        raise ValueError("concat_trajectories requires at least one trajectory")

    lengths = [ds.sizes["time"] for ds in trajectories]
    longest = max(lengths)

    padded = []
    for ds in trajectories:
        n = ds.sizes["time"]
        item = ds.drop_vars("time")
        if n < longest:
            # padding the mask alongside the data would cast NaN to True
            item = item.pad(time=(0, longest - n), constant_values=np.nan)
        mask = np.zeros(longest, dtype=bool)
        mask[:n] = True
        padded.append(item.assign(valid=("time", mask)))

    stacked = xr.concat(padded, dim="episode")
    stacked = stacked.assign_coords(
        episode=np.arange(len(trajectories)),
        time=("time", np.asarray(trajectories[int(np.argmax(lengths))].time)),
    )
    stacked.attrs = {
        "n_episodes": len(trajectories),
        "lengths": lengths,
        "spec_hash": trajectories[0].attrs.get("spec_hash", ""),
    }
    return stacked
