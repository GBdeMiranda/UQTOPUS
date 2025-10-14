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
) -> float:
    """
    Execute a single iteration of parameter exploration.

    Returns (qoi_value, simulation_results)
    """
    output_path = Path(exp_config['output_path'])
    iter_path = output_path / f"sample_{iteration:04d}" # sample will stand for iter

    exp_config_iter = exp_config.copy()
    exp_config_iter['output_path'] = str(iter_path)
    
    if verbose:
        logger.info(f"Iteration {iteration}: Running with parameters {parameters}")
    
    try:
        # 1) Run simulation with given parameters
        run_simulation(
            params=parameters,
            exp_config=exp_config_iter,
            verbose=verbose
        )
        
        # 2) Parse results and extract QoI with the function provided
        results = parse_openfoam_case(str(iter_path), variables=qoi_variables, time_dirs=qoi_times)
        qoi_value = qoi_extractor(results)
        
        return qoi_value
        
    except Exception as e:
        logger.error(f"Error in iteration {iteration}: {e}")
        raise

    

def exploration_loop(
    initial_parameters: Dict[str, float],
    parameter_updater: Callable[[Dict[str, float], float, int], Dict[str, float] | None],
    exp_config: Dict[str, Any],
    qoi_extractor: Callable[[xr.Dataset], float],
    qoi_variables: list[str],
    qoi_times: list[float] | str = None,
    max_iterations: int = 999,
    verbose: bool = False
) -> Tuple[list[Dict[str, float]], list[float]]:
    """
    Main exploration loop.
    
    Parameters:
        initial_parameters: Starting parameters
        parameter_updater: Function that takes (current_params, qoi_value) and returns new_params or None to stop
        exp_config: UQTOPUS experiment configuration
        qoi_extractor: Function to extract QoI from results
        qoi_variables: Variables to extract from OpenFOAM
        qoi_times: Time directories to parse
        max_iterations: Maximum number of iterations
        verbose: Enable verbose output
        
    Returns
    -------
    Tuple[list, list, list]
        (parameters_history, qoi_history, results_history)
    """
    parameters_history = []
    qoi_history = []
    
    current_params = initial_parameters.copy()
    
    for iteration in range(max_iterations):
        # Run iteration
        qoi_value = run_iteration(
            iteration=iteration,
            parameters=current_params,
            exp_config=exp_config,
            qoi_extractor=qoi_extractor,
            qoi_variables=qoi_variables,
            qoi_times=qoi_times,
            verbose=verbose
        )
        
        # Store history
        parameters_history.append(current_params.copy())
        qoi_history.append(qoi_value)
        
        # Update parameters
        new_params = parameter_updater(current_params, qoi_value, iteration)
        
        if new_params is None:
            if verbose:
                print(f"Exploration stopped at iteration {iteration}")
            break
            
        current_params = new_params
    
    return parameters_history, qoi_history