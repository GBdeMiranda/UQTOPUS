"""
Simulation Orchestration and Execution Module for UQTOPUS

Handles the simulation lifecycle (copy template, render Jinja2 templates, run OpenFOAM solvers,
and parse outputs) both for single runs and parallel batch studies.
"""

from __future__ import annotations

import os
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Any
from functools import partial
import multiprocessing as mp

import numpy as np
import xarray as xr
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Internal package imports
from .utils import load_config, parse_openfoam_case
from .sampler import generate_samples
from .exceptions import SolverDivergedError

logger = logging.getLogger(__name__)

_DESTINATION_FOLDER = Path('experiments/temp')   # Default destination folder for experiments


def run_simulation(params: dict[str, float], exp_config: dict[str, Any], verbose: bool = False) -> None:
    """
    Runs an OpenFOAM simulation with the given parameters.

    Parameters:
        params (dict): Dictionary containing the parameters for the simulation.
        exp_config (dict): Configuration dictionary containing experiment details.
        verbose (bool): Whether to output verbose print statements.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    if not params:
        raise ValueError("params must not be empty")
    if not isinstance(exp_config, dict):
        raise ValueError("exp_config must be a dictionary")
    if not exp_config:
        raise ValueError("exp_config must not be empty")

    base_dir = Path(exp_config['input_path'])
    output_path = Path(exp_config.get('output_path', _DESTINATION_FOLDER))
    solver_script = exp_config['solver']

    if 'input_path' not in exp_config:
        raise ValueError("exp_config must contain an 'input_path' key")

    # Handling the iter_ or sample_xxx folder case
    exp_name = output_path.name    
    parent_folder = Path(output_path).parent
    if exp_name.startswith("sample_"):
        parent_folder = parent_folder.parent
    if not Path(parent_folder).exists():
        Path(parent_folder).mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created parent directory: {parent_folder}")

    try:
        if not output_path.exists():
            output_path.mkdir(parents=True)
        else:
            if verbose:
                print(" -- The directory already exists. Files will be overwritten. --")
            
        result = subprocess.run(
            ["rsync", "-av", "--delete", f'{str(base_dir)}/', f'{str(output_path)}/'],
            check=True,
            capture_output=True,
            text=True
        )
        if verbose:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error copying the files:", e.stderr)

    env = Environment(
        loader=FileSystemLoader(base_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )

    # ======================================================================
    # REORGANIZING RENDERIZATION STRATEGY
    # Create a new dict with template paths with their respective params
    paths_n_vars = {}
    for param_path, value in params.items():
        path_parts = param_path.split('__')
        if len(path_parts) < 2:
            raise ValueError(f"Parameter key '{param_path}' is not in the correct format. Use 'folder__filename__paramname' format.")
        param = path_parts[-1]

        template_path = str(Path(*path_parts[:-1]))

        if template_path not in paths_n_vars:
            paths_n_vars[template_path] = {}
        paths_n_vars[template_path][param] = value

    # For each template path render all its params at once
    for template_path, params_dict in paths_n_vars.items():
        template = env.get_template(str(template_path))
        output = template.render(params_dict, undefined=StrictUndefined)

        target_path = output_path / template_path
        target_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dirs exist
        target_path.write_text(output)
    # ======================================================================

    try:
        solver_path = output_path / solver_script
        if not solver_path.exists():
            raise FileNotFoundError(f"Solver script not found: {solver_path}")

        result = subprocess.run(
            [f"./{solver_script}"],
            cwd=str(output_path),
            check=True,
            capture_output=True,
            text=True
        )
        if verbose:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Solver failed with code {e.returncode}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
        raise SolverDivergedError(
            f"OpenFOAM solver failed or diverged with exit code {e.returncode}.",
            returncode=e.returncode,
            stdout=e.stdout,
            stderr=e.stderr
        ) from e


def _process_random_sim(
    param_data: tuple[int, dict[str, float]],
    exp_config: dict[str, Any],
    verbose: bool = False,
) -> tuple[int, str | None]:
    """
    Process a single simulation (helper function for randomized multiprocessing).

    Returns:
        Tuple of the sample index and the error message, the latter being None
        when the simulation succeeded.
    """
    i, params = param_data
    exp_path = Path(exp_config.get('output_path', _DESTINATION_FOLDER))
    experiment_name = exp_path / f"sample_{i:04d}"

    # Copy configuration to prevent mutating shared state across workers
    local_config = exp_config.copy()
    local_config['output_path'] = str(experiment_name)

    try:
        run_simulation(
            params=params,
            exp_config=local_config,
            verbose=verbose
        )
    except Exception as e:
        logger.error("Sample %d failed: %s: %s", i, type(e).__name__, e)
        return i, f"{type(e).__name__}: {e}"

    return i, None


def _run_ensemble(
    X: np.ndarray | list,
    keys: list[str],
    config: dict[str, Any],
    nthreads: int = 1,
    verbose: bool = False,
) -> list[tuple[int, str]]:
    """
    Run one simulation per row of an experimental design, in parallel.

    Parameters:
        X: Experimental design, one row of parameter values per sample.
        keys: Parameter names, in the same order as the columns of X.
        config: Experiment configuration forwarded to run_simulation.
        nthreads: Number of worker processes.
        verbose: Whether to print the solver output of each run.

    Returns:
        List of (sample index, error message) for the samples that failed,
        sorted by index. Empty when every sample succeeded.
    """
    process_func = partial(
        _process_random_sim,
        exp_config=config,
        verbose=verbose
    )

    iparams = list(enumerate([dict(zip(keys, x)) for x in X]))

    failures: list[tuple[int, str]] = []
    with mp.get_context('spawn').Pool(nthreads) as pool:
        for i, error in tqdm(
            pool.imap_unordered(process_func, iparams),
            total=len(iparams),
            desc='Running simulations',
            mininterval=1.0
        ):
            if error is not None:
                failures.append((i, error))

    failures.sort()
    return failures


def _report_failures(failures: list[tuple[int, str]], n_samples: int) -> None:
    """
    Print a summary of the samples that failed during an ensemble run.

    Parameters:
        failures: Output of _run_ensemble.
        n_samples: Total number of samples in the experimental design.
    """
    if not failures:
        return

    logger.warning("%d of %d samples failed", len(failures), n_samples)
    print(f"WARNING: {len(failures)} of {n_samples} samples failed.")
    for i, error in failures:
        print(f"  sample_{i:04d}: {error}")


def uq_simulation(X: np.ndarray, Params: dict[str, Any]) -> None:
    """
    Function to run openFOAM simulations for an experimental design (ED) defined 
    by a table of input parameters. The function creates a directory for each sample (row) in the ED.
    """
    for k in Params.keys():
        if k not in [
            'input_path', 'output_path',
            'parameter_ranges', 'nthreads', 'solver',
            'theModel'
        ]:
            raise Exception(f"Unknown key '{k}' in Params")

    input_path = Params.get('input_path', None)
    output_path = Params.get('output_path', _DESTINATION_FOLDER)
    solver = Params.get('solver', None)
    keys = list(Params['parameter_ranges'].keys()) if 'parameter_ranges' in Params else None

    if keys is None or solver is None or input_path is None:
        raise Exception("The parameters 'input_path', 'solver', and 'parameter_ranges' must be provided as arguments in Params")
    else:
        if not os.path.exists(input_path):
            raise ValueError('The "input_path" path passed as parameter does not exist')
        if not isinstance(keys, list):
            keys = list(keys)
        if len(keys) != X.shape[1]:
            raise ValueError('The number of sampled parameters passed must be equal to the number of the input columns in the experimental design X')

    nthreads = Params['nthreads'] if 'nthreads' in Params else 1

    failures = _run_ensemble(X, keys, Params, nthreads)
    _report_failures(failures, len(X))

    print(f"UQ study completed. Results saved in '{output_path}' folder")


def run_uq_study(config_file: str | Path, n_samples: int, verbose: bool = False) -> None:
    """
    Standalone function to run UQ study for scalar parameters.
    """
    config = load_config(config_file)

    input_path = config.get('input_path', None)
    output_path = config.get('output_path', _DESTINATION_FOLDER)
    solver = config.get('solver', None)
    parameter_ranges = config.get('parameter_ranges', None)

    if input_path is None or solver is None or parameter_ranges is None:
        raise ValueError("The parameters 'input_path', 'solver', and 'parameter_ranges' must be provided as arguments in config_file")

    X = generate_samples(
        n_samples=n_samples,
        param_ranges=parameter_ranges,
        method='lhs',
        seed=42
    )
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X.tolist()

    nthreads = config['nthreads'] if 'nthreads' in config else 1
    keys = config['parameter_ranges'].keys()
    if keys is None:
        raise Exception("The parameter 'parameter_ranges' must be provided in the config file")

    failures = _run_ensemble(X, list(keys), config, nthreads)
    _report_failures(failures, len(X))

    if verbose:
        print(f"UQ study completed. Results saved in '{output_path}' folder")


class OpenFOAMSimulator:
    """
    Bridge between an OpenFOAM case template and a Python callable.

    Each call to run() executes one full simulation and returns the parsed
    field data as an xr.Dataset.
    """

    def __init__(
        self,
        template_path: str | Path,
        solver_script: str,
        output_path: str | Path,
        qoi_variables: list[str],
        qoi_times: list[str] | str | None = None,
        experiment_name: str = "run",
    ) -> None:
        self.template_path = Path(template_path)
        self.solver_script = solver_script
        self.output_path = Path(output_path)
        self.qoi_variables = qoi_variables
        self.qoi_times = qoi_times
        self.experiment_name = experiment_name
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
        overwrite: bool = True,
    ) -> xr.Dataset:
        """
        Execute one simulation with the given parameters and return results.

        Parameters:
            params: Dictionary of parameter values keyed as 'folder__filename__paramname'.
            step: Optional explicit run index.
                If provided, the run directory is named as f"{experiment_name}_{step:04d}".
                If None, the internal counter `_run_count` is used as index.
            verbose: Whether to print solver output.
            cleanup: Whether to delete the run directory after parsing results.
            overwrite: If True, existing run directory with same name is replaced.
        """
        idx = step if step is not None else self._run_count
        run_path = self.output_path / f"{self.experiment_name}_{idx:04d}"

        if run_path.exists():
            if overwrite:
                shutil.rmtree(run_path)
            else:
                raise FileExistsError(
                    f"Run directory already exists: {run_path}. "
                    "Pass overwrite=True to replace it, or use a different "
                    "'step'/'experiment_name'."
                )

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

        # number of executions performed
        self._run_count += 1

        if cleanup:
            try:
                shutil.rmtree(run_path)
            except OSError as e:
                logger.warning("Failed to clean up directory %s: %s", run_path, e)

        return dataset

    def reset(self) -> None:
        """Reset the internal run counter to zero."""
        self._run_count = 0

    def run_batch(
        self,
        params_list: list[dict[str, float]],
        n_jobs: int = -1,
        verbose: bool = False,
        cleanup: bool = False,
        overwrite: bool = True,
    ) -> list[xr.Dataset]:
        """
        Execute multiple simulations in parallel.

        Each run is given a deterministic index (0..len(params_list)-1) so that
        parallel workers never collide on the same run directory, even though
        each worker process has its own copy of `_run_count`.
        """
        if n_jobs < 1:
            n_jobs = mp.cpu_count()

        process_func = partial(
            self._process_single_run, verbose=verbose, cleanup=cleanup, overwrite=overwrite
        )

        indexed_params = list(enumerate(params_list))

        datasets = []
        with mp.get_context('spawn').Pool(n_jobs) as pool:
            for ds in tqdm(
                pool.imap(process_func, indexed_params),
                total=len(params_list),
                desc='Running parallel batch'
            ):
                datasets.append(ds)

        self._run_count += len(params_list)
        return datasets

    def _process_single_run(
        self, indexed_params: tuple[int, dict[str, float]], verbose: bool, cleanup: bool, overwrite: bool
    ) -> xr.Dataset:
        """Helper for multiprocessing. Uses the batch-local index to avoid
        directory name collisions between parallel workers."""
        idx, params = indexed_params
        return self.run(params, step=idx, verbose=verbose, cleanup=cleanup, overwrite=overwrite)

    def as_objective(
        self,
        metric_fn: Callable[[xr.Dataset], float],
        param_keys: list[str],
    ) -> Callable[[np.ndarray], float]:
        """
        Wrap the simulator as a scalar objective function.
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