
import numpy as np
from pyDOE3 import lhs, fullfact, pbdesign, bbdesign, ccdesign

def generate_samples(n_samples, param_ranges, method='lhs', seed=None):
    """Generate parameter samples for UQ study."""
    
    if seed is not None:
        np.random.seed(seed)
    
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    
    if method == 'lhs':
        unit_samples = lhs(n_params, samples=n_samples, criterion='centermaximin')
    elif method == 'random':
        unit_samples = np.random.random((n_samples, n_params))
    # elif method == 'grid' or method == 'fullfact':
    #     n_levels = int(np.ceil(n_samples ** (1/n_params)))
    #     unit_samples = fullfact([n_levels] * n_params) / (n_levels - 1)
    #     unit_samples = unit_samples[:n_samples]
    # elif method == 'plackett_burman':
    #     unit_samples = (pbdesign(n_params) + 1) / 2
    #     unit_samples = unit_samples[:n_samples]
    # elif method == 'box_behnken':
    #     unit_samples = (bbdesign(n_params) + 1) / 2
    #     unit_samples = unit_samples[:n_samples]
    # elif method == 'central_composite':
    #     unit_samples = (ccdesign(n_params) + 1) / 2
    #     unit_samples = np.clip(unit_samples, 0, 1)
    #     unit_samples = unit_samples[:n_samples]
    else:
        raise ValueError(f"Unknown sampling method: {method}. Available methods: 'lhs', 'random' ")
    
    samples = np.zeros_like(unit_samples)
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)
    
    return samples