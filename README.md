<div align="center">
  <img src="https://raw.githubusercontent.com/GBdeMiranda/UQTOPUS/main/assets/uqtopus.png" alt="UQTOPUS Logo" width="128" height="128">
  <h1>UQTOPUS</h1>
  <p><b>U</b>ncertainty <b>Q</b>uantification <b>T</b>oolbox for <b>O</b>penFOAM and <b>P</b>ython <b>U</b>nified <b>S</b>imulation workflows<br></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

UQTOPUS is a Python framework for uncertainty quantification (UQ) studies with OpenFOAM CFD simulations 
It automates sampling, case templating, execution, and results analysis to enable reproducible and scalable UQ experiments. The pipeline is designed to be extensible, including hooks for surrogate modeling and custom post-processing around OpenFOAM runs.

Ultimately, this project aims to enable the usage of OpenFOAM simulator to perform efficient and automated uncertainty quantification studies aided by surrogate modeling techniques wrapping the OpenFOAM simulation process.

<img src="https://raw.githubusercontent.com/GBdeMiranda/UQTOPUS/main/assets/simulator_wrapper.png" alt="Simulator wrapper overview" width="600">

Built on top of `Jinja2`, `xarray` and `fluidfoam`.

## Features

- End-to-end UQ workflow: sample → render cases → run → collect → analyze
- OpenFOAM-native integration: Jinja2-templated dictionaries and Allrun orchestration
- Parallel execution built-in support for sampled scenarios
- Reproducible experiment management and basic statistics with CSV/NumPy-friendly outputs
- Extensible hooks for surrogate modeling with `uqpylab` (https://uqpylab.uq-cloud.io/) and custom post-processing 

## Installation

### OpenFOAM Installation

For Ubuntu/Debian systems:
```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get -y install openfoam9
```

Source OpenFOAM environment:
```bash
source /opt/openfoam9/etc/bashrc
```

For other systems, visit: https://openfoam.org/download/

### Requirements

- Python 3.9+
- OpenFOAM 9+ (for running actual simulations)

### Setup

1. Install Python package:
   ```bash
   pip install uqtopus
   ```

2. Ensure OpenFOAM is installed and sourced:
   ```bash
   source /opt/openfoam9/etc/bashrc  # Adjust path as needed
   ```

## Basic Usage

- Set a config file:
```yaml
# config.yaml
output_path: experiments/myUQStudy
input_path: templates/templateSimulation
solver: mySolverScript
parameter_ranges:
  [folder_path]__[file_name]__[param_name]: [0.01, 0.3]
nthreads: 2
```

- Templatize the simulation file with the desired variables in Jinja2 format (double curly braces):
```
...
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

param1              {{param1}};
param2              {{param2}};
param3              {{param3}};
...
```

- Set the solver script:
```bash
#!/bin/bash
cd ${0%/*} || exit 1    # Run from this directory

# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

runApplication blockMesh
runApplication $(getApplication)
```

- Run a UQ study:
```python
import uqtopus as uqt
config = 'config.yaml'
uqt.run_uq_study(config, n_samples=50)
```

For more information on configuring and running UQ studies, please refer to the example notebooks.

## Directory Structure

```
UQTOPUS/
├── src/                      # Python package
├── examples/                 # Example of usage with scripts and templates
│   ├── templates/            # OpenFOAM case templates
│   │   └── base_case/        # Base case with Jinja2 placeholders
│   │       ├── constant/
│   │       ├── system/
│   │       ├── 0/
│   │       └── [script_file]     # Run script (e.g., Allrun)
│   ├── experiments/          # UQ study results
│   │   └── [study_name]/     # Individual study results
│   │       ├── sample_001/   # OpenFOAM case for sample 1
│   │       ├── sample_002/   # OpenFOAM case for sample 2
│   │       └── ...
│   └── config.yaml           # Configuration file for examples
└── README.md                 # This file
```

## Configuring a template case

1) Add Jinja2 placeholders in OpenFOAM dictionaries where parameters vary.
   - Example (constant/transportProperties):
     ```foam
     transportModel  {{ transportModel | default('Newtonian') }};
     nu              [0 2 -1 0 0 0 0] {{ nu | default(1e-5) }};
     ```
   - Example with conditionals (constant/turbulenceProperties):
     ```foam
     simulationType RAS;
     RAS
     {
       RASModel      {% if turbulence == "kEpsilon" %}kEpsilon{% else %}kOmegaSST{% endif %};
       turbulence    on;
       printCoeffs   on;
     }
     ```
   - Jinja tips:
     - Use default: {{ var|default(1.0) }}
     - Use rounding: {{ diameter|round(5) }}
     - Use conditionals/loops for switching models or patch sets.

2) Provide a run script in the template root (e.g., templates/base_case/Allrun). Keep it non-templated.
   - Minimal example (mesh + solver):
      ```bash
      cd ${0%/*} || exit 1    # Run from this directory

      # Source tutorial run functions
      . $WM_PROJECT_DIR/bin/tools/RunFunctions

      runApplication blockMesh
      runApplication $(getApplication)
      ```

3) Place the template under `templates` directory to ensure organization with standard OpenFOAM layout (0/, constant/, system/). The UQ runner will render the Jinja placeholders per sample and invoke your run script.

## Examples

See `examples` folder for notebooks with workflow demonstrations.

## License

MIT License - see LICENSE file for details.
