from pathlib import Path
from functools import partial
import multiprocessing as mp
import logging
import re
import os
import json

from tqdm import tqdm
import numpy as np
import xarray as xr
import yaml
from fluidfoam import readmesh, readfield

logger = logging.getLogger(__name__)


def _is_time_dir(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _scan_time_dirs(case_dir: str) -> dict[float, str]:
    """
    Scans a case directory and returns all valid time directories.
    """
    return {
        float(p.name): p.name
        for p in Path(case_dir).iterdir()
        if p.is_dir() and _is_time_dir(p.name)
    }


def _resolve_time_dirs(
    available: dict[float, str],
    requested: list,
    tol: float = 1e-6,
) -> list[str]:
    """
    Resolves a list of requested times (float or string) to actual directory
    names found on disk, using nearest-neighbour matching within a relative
    tolerance to handle OpenFOAM floating-point drift.

    Parameters:
        available : mapping returned by _scan_time_dirs.
        requested : list of times as float or string (e.g. [1e6, '5e+06']).
        tol       : max distance between requested and nearest available time.

    Returns:
        List of directory name strings in the same order as requested.
    """
    resolved = []
    for t in requested:
        t_float = float(t)
        closest = min(available.keys(), key=lambda x: abs(x - t_float))
        denom = max(abs(t_float), 1e-10)
        if abs(closest - t_float) / denom > tol:
            raise ValueError(
                f"Requested time {t} not found in case directory "
                f"(closest available: {available[closest]})."
            )
        resolved.append(available[closest])
    return resolved


def parse_openfoam_case(
    case_dir: str,
    variables: list[str],
    time_dirs: list | str | None = None,
) -> xr.Dataset:
    """
    Parses an OpenFOAM case directory and reads field data into an xr.Dataset.

    Parameters:
        case_dir  : Path to the root of the OpenFOAM case.
        variables : Field names to read (e.g. ['U', 'p', 'Sb']).
        time_dirs : Which time steps to read. Accepts:
                      - None          : all detected time directories.
                      - 'last'        : only the last time directory.
                      - list of float : times matched by nearest neighbour.
                      - list of str   : directory names matched by nearest
                                        neighbour (tolerant to formatting).

    Returns:
        xr.Dataset with dimensions (time, cell) and spatial coordinates x, y, z.
    """
    available = _scan_time_dirs(case_dir)

    if not available:
        raise FileNotFoundError(f"No time directories found in {case_dir}.")

    if time_dirs is None:
        resolved = [available[t] for t in sorted(available)]
    elif isinstance(time_dirs, str) and time_dirs == "last":
        resolved = [available[max(available)]]
    else:
        if not isinstance(time_dirs, list):
            time_dirs = [time_dirs]
        resolved = _resolve_time_dirs(available, time_dirs)

    times = [float(d) for d in resolved]

    # Store all data
    all_data = {}

    # Read all data first
    for time_dir in resolved:
        all_data[time_dir] = {}
        for field_file in variables:
            try:
                all_data[time_dir][field_file] = readfield(
                    case_dir, time_dir, field_file, verbose=False
                ).T
            except Exception as e:
                logger.warning("Could not read field '%s' at time '%s': %s", field_file, time_dir, e)

    x, y, z = readmesh(case_dir, verbose=False)
    max_elements = len(x)

    # Handling uniform fields (single value in file)
    for time_data in all_data.values():
        for fname, field in time_data.items():
            if field.ndim == 1 and field.shape[0] == 1:     # scalar uniform field
                time_data[fname] = np.stack([field] * max_elements, axis=0).flatten()
            elif field.ndim == 2 and (field.shape[0] == 1 or field.shape[1] == 1):   # vector uniform field
                time_data[fname] = np.stack(
                    [field[0]] * max_elements, axis=0
                ).reshape(max_elements, -1)

    # Create xarray data variables
    data_vars = {}
    for var in variables:
        # Stack time data for this variable
        var_data = [all_data[d][var] for d in resolved if var in all_data[d]]
        if not var_data:
            logger.warning("No data collected for variable '%s', skipping.", var)
            continue
        var_array = np.stack(var_data, axis=0)

        # Create appropriate dimensions based on shape
        if var_array.ndim == 2:
            dims = ["time", "cell"]
        elif var_array.ndim == 3:
            dims = ["time", "cell", "component"]
        else:
            dims = ["time"] + [f"dim_{i}" for i in range(1, var_array.ndim)]
        data_vars[var] = xr.DataArray(var_array, dims=dims)

    return xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "x": ("cell", x),
            "y": ("cell", y),
            "z": ("cell", z),
        },
    )


def read_uq_experiment(
    case_dir: str,
    variables: list[str],
    n_samples: int,
    time_dirs: list | str | None = None,
    nthreads: int = 1,
) -> xr.Dataset:
    """
    Reads all sample cases from a UQ experiment into a single xr.Dataset.

    Parameters:
        case_dir  : Path to the directory containing sample_0000, sample_0001, ...
        variables : Field names to read.
        n_samples : Number of samples to read.
        time_dirs : Passed directly to parse_openfoam_case for each sample.
        nthreads  : Number of parallel workers.

    Returns:
        xr.Dataset with dimensions (sample, time, cell).
    """
    case_dir = Path(case_dir)

    logger.info("Reading %d samples from %s", n_samples, case_dir)

    with mp.get_context("spawn").Pool(nthreads) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(
                        parse_openfoam_case,
                        variables=variables,
                        time_dirs=time_dirs,
                    ),
                    [str(case_dir / f"sample_{i:04d}") for i in range(n_samples)],
                ),
                total=n_samples,
                desc="Reading cases",
                unit="case",
                mininterval=1.0,
            )
        )
    # Concatenate along sample dimension
    combined_ds = xr.concat(results, dim="sample")
    combined_ds = combined_ds.assign_coords(sample=list(range(n_samples)))
    return combined_ds


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads a YAML or JSON configuration file.

    Parameters:
        config_path : Path to the configuration file.

    Returns:
        dict with the configuration contents.
    """
    try:
        with open(config_path, "r") as f:
            if config_path.lower().endswith(".json"):
                return json.load(f)
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Error loading config from %s: %s", config_path, e)
        return {}


# =============================================================================
# LEGACY - not imported by the main module
# =============================================================================

def read_openfoam_field(file_path):
    """
    LEGACY - use fluidfoam.readfield instead.
    """
    logger.warning("read_openfoam_field is deprecated. Use fluidfoam.readfield instead.")

    try:
        with open(file_path, "r") as f:
            content = f.readlines()
        # Find the 'internalField' line
        start_index = next(
            i for i, line in enumerate(content) if line.startswith("internalField")
        )
        # Check if the field is uniform
        field_info = content[start_index]

        if field_info.split()[1] == "uniform":
            data = re.findall(r"[-+]?\d*\.\d+|\d+", field_info)
            return np.array([float(d) for d in data])
        # Non uniform has the number of elements in the data block
        num_elements = int(content[start_index + 1])
        
        # Extract the data block
        data = content[start_index + 3 : start_index + 3 + num_elements]

        # Parse data into NumPy array
        values = []
        for line in data:
            line = line.strip().strip("()")
            if " " in line:  # Vector or multiple values
                try:
                    values.append(np.array([float(x) for x in line.split()]))
                except ValueError:
                    logger.debug("Skipping malformed vector line: %s", line)
            else:  # Single value
                try:
                    values.append(float(line))
                except ValueError:
                    logger.debug("Skipping malformed scalar line: %s", line)

        return np.array(values)

    except Exception as e:
        logger.error("Error reading file %s: %s", file_path, e)
        return None