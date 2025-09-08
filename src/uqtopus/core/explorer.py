import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Callable, Any, Tuple
import logging

from .runner import run_simulation
from ..utils import parse_openfoam_case


logger = logging.getLogger(__name__)


def run_iteration(
    iteration: int,
    parameters: Dict[str, float],
    exp_config: Dict[str, Any],
    qoi_extractor: Callable[[xr.Dataset], float],
    qoi_variables: list[str],
    qoi_times: list[float]|str = None,
    verbose: bool = False
) -> Tuple[float, xr.Dataset, Path]:
    """
    Execute a single iteration of parameter exploration.

    Returns: Tuple[float, xr.Dataset, Path]
        (qoi_value, simulation_results, case_directory)
    """
    output_path = Path(exp_config['output_path'])
    iter_path = output_path / f"iter_{iteration:04d}"

    exp_config_iter = exp_config.copy()
    exp_config_iter['output_path'] = str(iter_path)
    
    if verbose:
        logger.info(f"Iteration {iteration}: Running with parameters {parameters}")
    
    try:
        # Red arrow: Run simulation
        run_simulation(
            params=parameters,
            exp_config=exp_config_iter,
            verbose=verbose
        )
        
        # Green arrow: Parse results
        results = parse_openfoam_case(str(iter_path), variables=qoi_variables, time_dirs=qoi_times)
        
        # Extract QoI
        qoi_value = qoi_extractor(results)
    
        
        return qoi_value, results, iter_path
        
    except Exception as e:
        logger.error(f"Error in iteration {iteration}: {e}")
        raise