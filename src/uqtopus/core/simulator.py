"""
OpenFOAM Simulator

Bridge between an OpenFOAM case template and Python optimization/RL workflows.
Handles the simulation lifecycle (copy template, render Jinja2, run solver,
parse results) for a single run. Loop logic belongs to the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Callable

import uuid
import numpy as np
import xarray as xr

import multiprocessing as mp
from functools import partial
from tqdm import tqdm

from .runner import run_simulation
from ..utils import parse_openfoam_case

logger = logging.getLogger(__name__)


class OpenFOAMSimulator:
    """
    Bridge between an OpenFOAM case template and a Python callable.

    Each call to run() executes one full simulation and returns the parsed
    field data as an xr.Dataset. What you do with that data (optimize, train
    an RL agent, fit parameters) is entirely up to the caller.

    Parameters:
        template_path (str or Path)
            Path to the OpenFOAM template case directory.

        solver_script (str)
            Name of the solver/run script inside the template (e.g. 'Allrun').

        output_path (str or Path)
            Base directory for simulation outputs. Each run creates a
            run_XXXX/ subdirectory inside it.

        qoi_variables (list)
            OpenFOAM field names to read after each simulation (e.g. ['U', 'p']).

        qoi_times (list or str or None)
            Time directory names to read (e.g. ['0.1', '0.2']).
            None reads all available time directories.
    """

    def __init__(
        self,
        template_path: str | Path,
        solver_script: str,
        output_path: str | Path,
        qoi_variables: list[str],
        qoi_times: list[str] | str | None = None,
    ) -> None:
        self.template_path = Path(template_path)
        self.solver_script = solver_script
        self.output_path = Path(output_path)
        self.qoi_variables = qoi_variables
        self.qoi_times = qoi_times
        self._run_count: int = 0

        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template path does not exist: {self.template_path}"
            )

    @property
    def run_count(self) -> int:
        """Total number of simulation runs executed by this instance."""
        return self._run_count

    def run(
        self,
        params: dict[str, float],
        step: int | None = None,
        verbose: bool = False,
        cleanup: bool = False,
    ) -> xr.Dataset:
        """
        Execute one simulation with the given parameters and return results.

        Parameters:
            params (dict)
                Parameter values keyed as 'folder__filename__paramname'.
                Example: {'constant__transportProperties__nu': 1e-5}

            step (int or None)
                Explicit index for the output directory name (run_XXXX).
                If None, uses and increments the internal run counter.
                Passing an explicit value does not change the internal counter.

            verbose (bool)
                Forward verbose output from the solver and parser.

        Returns an xr.Dataset with dims [time, cell] for scalars and
        [time, cell, component] for vectors, with coords x, y, z.
        """
        idx = step if step is not None else self._run_count
        run_hash = uuid.uuid4().hex[:8]
        run_path = self.output_path / f"run_{idx:04d}_{run_hash}"

        exp_config = {
            "input_path": str(self.template_path),
            "output_path": str(run_path),
            "solver": self.solver_script,
        }

        if verbose:
            logger.info("Run %d: params=%s", idx, params)

        run_simulation(params=params, exp_config=exp_config, verbose=verbose)

        dataset = parse_openfoam_case(
            str(run_path),
            variables=self.qoi_variables,
            time_dirs=self.qoi_times,
        )

        if step is None:
            self._run_count += 1

        if cleanup:
            try:
                shutil.rmtree(run_path)
            except OSError as e:
                logger.warning("Failed to clean up directory %s: %s", run_path, e)

        return dataset

    def reset(self) -> None:
        """
        Reset the internal run counter to zero.

        Call this between independent experiments or RL episodes so that
        output directories restart from run_0000.
        """
        self._run_count = 0

    def run_batch(
        self,
        params_list: list[dict[str, float]],
        n_jobs: int = -1,
        verbose: bool = False,
        cleanup: bool = False,
    ) -> list[xr.Dataset]:
        """
        Execute multiple simulations in parallel.
        
        Parameters:
            params_list (list): List of parameter dictionaries.
            n_jobs (int): Number of parallel workers. If -1, uses all CPU cores.
            verbose (bool): Forward verbose output.
            
        Returns:
            list[xr.Dataset]: List of xarray datasets corresponding to the inputs.
        """
        
        if n_jobs < 1:
            n_jobs = mp.cpu_count()
            
        process_func = partial(self._process_single_run, verbose=verbose, cleanup=cleanup)
        
        datasets = []
        # Using spawn to ensure thread safety with OpenFOAM subprocesses
        with mp.get_context('spawn').Pool(n_jobs) as pool:
            for ds in tqdm(
                pool.imap(process_func, params_list),
                total=len(params_list),
                desc='Running parallel batch'
            ):
                datasets.append(ds)
                
        # Update parent run count to reflect the batch size
        self._run_count += len(params_list)
        return datasets

    def _process_single_run(self, params: dict[str, float], verbose: bool, cleanup: bool) -> xr.Dataset:
        """Helper for multiprocessing."""
        return self.run(params, verbose=verbose, cleanup=cleanup)

    def as_objective(
        self,
        metric_fn: Callable[[xr.Dataset], float],
        param_keys: list[str],
    ) -> Callable[[np.ndarray], float]:
        """
        Wrap the simulator as a scalar objective function.

        Returns a function f(params_array) -> float compatible with
        scipy.optimize.minimize, optuna, and similar libraries.
        The loop and convergence logic stay entirely in the optimization library.

        Parameters:
            metric_fn (callable)
                Extracts a scalar from the simulation results.
                Example: lambda ds: float(ds['p'].max())

            param_keys (list)
                Ordered parameter keys matching positions in the input array.
                Must use the 'folder__file__param' encoding.
        """
        def objective(params_array: np.ndarray) -> float:
            params = dict(zip(param_keys, np.asarray(params_array).tolist()))
            return metric_fn(self.run(params))

        return objective

    def as_residual(
        self,
        residual_fn: Callable[[xr.Dataset], np.ndarray],
        param_keys: list[str],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Wrap the simulator as a vector residual function for parameter fitting.

        Returns a function r(params_array) -> np.ndarray compatible with
        scipy.optimize.least_squares.

        Parameters:
            residual_fn (callable)
                Computes the residual vector between simulation output and targets.
                Example: lambda ds: ds['U'].mean('cell').values.flatten() - target

            param_keys (list)
                Ordered parameter keys matching positions in the input array.
        """
        def residual(params_array: np.ndarray) -> np.ndarray:
            params = dict(zip(param_keys, np.asarray(params_array).tolist()))
            return np.asarray(residual_fn(self.run(params)))

        return residual

    def __repr__(self) -> str:
        return (
            f"OpenFOAMSimulator("
            f"template={self.template_path.name!r}, "
            f"solver={self.solver_script!r}, "
            f"runs={self._run_count})"
        )
