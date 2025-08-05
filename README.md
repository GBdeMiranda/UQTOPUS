# OpenUQFOAM

OpenUQFOAM is a Python framework for uncertainty quantification (UQ) studies with OpenFOAM computational fluid dynamics simulations. The framework provides a streamlined approach to parameter sampling, simulation execution, and statistical analysis for CFD applications.

## Features

- **Parameter Sampling**: Latin Hypercube Sampling (LHS), random sampling, and grid sampling methods
- **Template Management**: Jinja2-based templating for OpenFOAM parameter files
- **Parallel Execution**: Built-in support for parallel simulation workflows

## Installation

### Requirements

- Python 3.8+
- OpenFOAM 9+ (for running actual simulations)
- Required Python packages:

  ```
  numpy
  pandas
  jinja2
  pyyaml
  matplotlib
  tqdm
  pyDOE2 (optional, for LHS sampling)
  ```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/GBdeMiranda/OpenUQFOAM.git
   cd OpenUQFOAM
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure OpenFOAM is installed and sourced:
   ```bash
   source /opt/openfoam9/etc/bashrc  # Adjust path as needed
   ```

## Directory Structure

```
OpenUQFOAM/
├── src/                      # Main Python modules
│   ├── openfoam_tools.py     # OpenFOAM I/O and simulation functions
│   ├── template_manager.py   # Jinja2 template processing
│   └── uq_runner.py          # Main UQ orchestration logic
├── templates/                # OpenFOAM case templates
│   └── base_case/            # Base case with Jinja2 placeholders
│       ├── constant/
│       ├── system/
│       └── 0/
├── notebooks/                # Example scripts and analysis
├── experiments/              # UQ study results
│   └── [study_name]/         # Individual study results
│       ├── sample_001/       # OpenFOAM case for sample 1
│       ├── sample_002/       # OpenFOAM case for sample 2
│       ...
├── scripts/                  # Auxiliary scripts
│   └── run_solver.sh         # OpenFOAM solver execution script
├── config.yaml               # Main configuration file
└── README.md                 # This file
```

## Examples

See `notebooks/simple_example.ipynb` for a workflow demonstration.

## Contributing

This framework prioritizes simplicity and scientific utility. Contributions should maintain these principles:

- Keep the codebase minimal and focused
- Preserve compatibility with existing workflows
- Document scientific applications and validation cases
- Maintain clear separation between core functionality and extensions

## License

MIT License - see LICENSE file for details.