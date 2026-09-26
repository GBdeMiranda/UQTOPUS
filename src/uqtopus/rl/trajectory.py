"""
Trajectory Files

The solver -> Python channel, one row per control step: the end of the interval
the action covers, the observation read at its start, and the action as sampled,
before the ramp and the clip to the bounds. The solver writes it to
postProcessing/uqtopusPolicy/<start time>/trajectory.dat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .spec import PolicySpec


def read_trajectory(source: str | Path, spec: PolicySpec) -> xr.Dataset:
    """
    Read one episode into an xr.Dataset.

    Parameters:
        source (str or Path): the trajectory file, or a case directory, from
            which the file of the latest start time is read.
        spec (PolicySpec): contract the episode ran under; the spec hash the
            solver recorded must match it.

    Returns:
        xr.Dataset with 'observation' (time, obs_component) and 'action'
        (time, act_component), and the recorded spec hash and seed in attrs.
    """
    path = Path(source)
    if path.is_dir():
        candidates = list((path / "postProcessing" / "uqtopusPolicy").glob("*/trajectory.dat"))
        if not candidates:
            raise ValueError(f"no postProcessing/uqtopusPolicy/<time>/trajectory.dat in {path}")
        path = max(candidates, key=lambda p: float(p.parent.name))

    header = {}
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            key, _, value = line.lstrip("#").strip().partition(" ")
            header[key] = value.strip()
        elif line.strip():
            rows.append(line.split())

    if header.get("specHash") != spec.hash:
        raise ValueError(
            f"{path} was written for contract {header.get('specHash')} but the caller "
            f"expects {spec.hash}; the case dictionary and the policy disagree"
        )
    if not rows:
        raise ValueError(f"{path} holds no control steps")

    table = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path} contains non-finite values")

    return xr.Dataset(
        data_vars={
            "observation": (("time", "obs_component"), table[:, 1 : 1 + spec.obs_dim]),
            "action": (("time", "act_component"), table[:, 1 + spec.obs_dim :]),
        },
        coords={
            "time": table[:, 0],
            "obs_component": spec.observation.component_names(),
            "act_component": spec.action.component_names(),
        },
        attrs={"spec_hash": spec.hash, "seed": int(header["seed"]), "source": str(path)},
    )
