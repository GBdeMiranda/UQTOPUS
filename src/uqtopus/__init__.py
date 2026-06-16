"""
OpenFOAM Uncertainty Quantification Toolkit

Provides tools and a interface for running UQ, RL and Optimization 
studies for OpenFOAM simulations.
"""

from .core import (
    run_uq_study, uq_simulation, run_simulation,
    generate_samples,
    OpenFOAMSimulator,
)
from .utils import load_config, read_openfoam_field, parse_openfoam_case, read_uq_experiment

from importlib.metadata import version

__version__ = version("uqtopus")

__all__ = [
    'OpenFOAMSimulator',
    'run_uq_study', 'uq_simulation', 'run_simulation', 'generate_samples',
    'load_config', 'read_openfoam_field', 'parse_openfoam_case', 'read_uq_experiment',
]