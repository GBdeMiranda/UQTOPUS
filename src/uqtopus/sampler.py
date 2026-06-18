import numpy as np
from pyDOE3 import lhs, fullfact, pbdesign, bbdesign, ccdesign

def generate_samples(n_samples, param_ranges, method='lhs', seed=None, **kwargs):
    """
    Generate parameter samples for UQ study.
    
    Parameters:
        n_samples (int): Number of samples to generate.
        param_ranges (dict): Dictionary with parameter ranges.
        method (str): Built-in method ('lhs', 'random').
        seed (int, optional): Random seed.
        **kwargs: Additional keyword arguments passed to the sampler.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    
    if method == 'lhs':
        unit_samples = lhs(n_params, samples=n_samples, criterion='centermaximin')
    elif method == 'random':
        unit_samples = np.random.random((n_samples, n_params))
    elif method in ('grid', 'fullfact'):
        levels = kwargs.get('levels')
        if levels is None:
            n_levels = int(np.ceil(n_samples ** (1 / n_params)))
            levels = [n_levels] * n_params
        elif isinstance(levels, int):
            levels = [levels] * n_params

        grid_indices = fullfact(levels)
        unit_samples = np.zeros_like(grid_indices, dtype=np.float64)
        for i, l in enumerate(levels):
            if l > 1:
                unit_samples[:, i] = grid_indices[:, i] / (l - 1)
            else:
                unit_samples[:, i] = 0.5
    elif method == 'plackett_burman':
        pb_matrix = pbdesign(n_params)
        unit_samples = (pb_matrix + 1) / 2
        if n_samples is not None and n_samples < len(unit_samples):
            unit_samples = unit_samples[:n_samples]
    elif method == 'box_behnken':
        if n_params < 3:
            raise ValueError("Box-Behnken design requires at least 3 parameters.")
        bb_matrix = bbdesign(n_params)
        unit_samples = (bb_matrix + 1) / 2
        if n_samples is not None and n_samples < len(unit_samples):
            unit_samples = unit_samples[:n_samples]
    elif method == 'central_composite':
        center = kwargs.get('center', (4, 4))
        alpha = kwargs.get('alpha', 'orthogonal')
        face = kwargs.get('face', 'faced')
        cc_matrix = ccdesign(n_params, center=center, alpha=alpha, face=face)
        
        min_val_cc = cc_matrix.min()
        max_val_cc = cc_matrix.max()
        if max_val_cc > min_val_cc:
            unit_samples = (cc_matrix - min_val_cc) / (max_val_cc - min_val_cc)
        else:
            unit_samples = np.zeros_like(cc_matrix)
            
        if n_samples is not None and n_samples < len(unit_samples):
            unit_samples = unit_samples[:n_samples]
    else:
        raise ValueError(
            f"Unknown sampling method: {method}. "
            "Available methods: 'lhs', 'random', 'grid'/'fullfact', "
            "'plackett_burman', 'box_behnken', 'central_composite' or a custom callable."
        )
    
    samples = np.zeros_like(unit_samples)
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)
    
    return samples
