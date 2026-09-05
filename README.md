<div align="center">
  <img src="https://raw.githubusercontent.com/GBdeMiranda/UQTOPUS/main/assets/uqtopus.png" alt="UQTOPUS Logo" width="128" height="128">
  <h1>UQTOPUS</h1>
  <p><b>U</b>ncertainty <b>Q</b>uantification <b>T</b>oolbox for <b>O</b>penFOAM and <b>P</b>ython <b>U</b>nified <b>S</b>imulation<br></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

UQTOPUS is a Python framework for uncertainty quantification (UQ) studies with OpenFOAM CFD simulations.
It automates sampling, case templating, execution, and results analysis to enable reproducible and scalable UQ experiments. The pipeline is designed to be extensible, including hooks for surrogate modeling and custom post-processing around OpenFOAM runs.

Ultimately, this project aims to enable the usage of OpenFOAM simulator to perform efficient and automated uncertainty quantification studies aided by surrogate modeling techniques wrapping the OpenFOAM simulation process.

<img src="https://raw.githubusercontent.com/GBdeMiranda/UQTOPUS/main/assets/simulator_wrapper.png" alt="Simulator wrapper overview" width="600">

Built on top of `Jinja2`, `xarray` and `fluidfoam`.

## Features

- End-to-end UQ workflow: sample -> render cases -> run -> collect -> analyze
- OpenFOAM-native integration: Jinja2-templated dictionaries and Allrun orchestration
- Parallel execution built-in support for sampled scenarios
- Reproducible experiment management and basic statistics with CSV/NumPy-friendly outputs
- Extensible hooks for surrogate modeling with `uqpylab` (https://uqpylab.uq-cloud.io/) and custom post-processing 

## Installation

### OpenFOAM

UQTOPUS targets OpenFOAM 9 from the OpenFOAM Foundation (openfoam.org).

```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get -y install openfoam9
source /opt/openfoam9/etc/bashrc
```

### Python package

```bash
pip install uqtopus
```

Requires Python 3.10 or newer.

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
├── src/uqtopus/              # Python package
│   ├── rl/                   # closed-loop reinforcement learning
│   └── policy_src/           # OpenFOAM library: the ONNX boundary condition
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

## Closed-loop RL module

Needed only for `uqtopus.rl`. The package ships an OpenFOAM library that adds the `uqtopusBoundaryCondition` boundary condition. The solver loads it through the `libs()` line of `controlDict`.

### 1. Install it

```bash
pip install "uqtopus[rl]"
```

### 2. Build the OF library

Just run
```bash
uqtopus rl-build --install-onnxruntime
```
to find the OpenFOAM installation, download the ONNX Runtime C++ package if none is present, compile and write the library where the solver already looks for it..

Drop `--install-onnxruntime` if you already have the runtime; it is found through `ONNXRUNTIME_ROOT` or under `~/opt`. Useful options:

| option | what it does |
|---|---|
| `--check` | run the checks and stop, building nothing |
| `--foam PATH` | use a specific `etc/bashrc` or installation root |
| `--onnxruntime PATH` | use a specific ONNX Runtime release |

### 3. Mark the case

Two lines in `system/controlDict` are **added by the user to a working case**: one to load the library, one to mark where the contract goes.
```
application     pimpleFoam;
libs            ("libuqtopusPolicy.so");

{{ controller }}
```

Then, just define **each controlled patch names the boundary condition**. It takes the action component of the target that carries its name:
```
patch
{
    type            uqtopusBoundaryCondition;
    value           uniform (0 0 0);
}
```

### 4. Declare what the policy reads and drives

A `PolicySpec` is the contract both sides read. `ActionSpec` names the patches the action drives, one component each, in the order written. The observation is a list of sources, concatenated in declaration order, and there are two kinds.

`ProbeSource` reads a field at points. The solver samples the field held in memory during the step, so nothing is read back from a written file:
```python
import numpy as np
import uqtopus.rl as rl

upstream = np.array([
    [-0.8,  0.3, 0.0],  # one row per probe
    [-0.8, -0.3, 0.0],  # in mesh coordinates: x, y, z
])

rl.ProbeSource(field_name="p", positions=upstream, name="upstream")
```
`name` labels the group in the trajectory columns, which come out as `upstream.p.0` and `upstream.p.1` for each probe.

`RegistrySource` brings in a custom quantity the case computed itself, by the name it stored it under:
```python
rl.RegistrySource(name="myCustomSource")
```
One source per quantity, so two flow rates means two of these. Use it for what is not a point value of a field: a flux through a patch, a running total, average over a patch or mesh. The case should publish the quantity itself, as section 6 shows.

```python
spec = rl.PolicySpec(
    observation=rl.ObservationSpec(
        sources=(
            rl.ProbeSource(field_name="p", positions=upstream, name="upstream"),
            rl.RegistrySource(name="myCustomSource"),
        )
    ),
    action=rl.ActionSpec(name="omega", targets=("patchA", "patchB"), low=-5.0, high=5.0),
    control_interval=0.5,
)
```

### 5. Let `uqtopus` render it!

`uqtopus.rl` fills the placeholder before every episode. The key names where it goes, as `folder__file__variable`:

```python
runner = rl.ClosedLoopRunner(
    uqt.OpenFOAMSimulator(case, "Allrun", runs, ["U", "p"]),
    spec,
    reward_fn,
    controller_keys="system__controlDict__controller",
    function_objects=["forces"],
)
```

What lands in the case is the whole contract, so the solver and the network agree on where the observation is measured and what the bounds are:
```
uqtopusPolicy
{
    type            uqtopusPolicy;
    policy          "/abs/path/policy.onnx";
    specHash; controlInterval; startTime; seed; 
    observation
    {
        dim; sources;
    }
    action
    {
        name; nComponents; distribution; rampFraction; low; high; targets;
    }
}
```
No worries with the details here, it is all handled by the Python module itself.

### 6. Publish a quantity the mesh does not hold

`RegistrySource` reads a scalar by name, so the case stores it first. OpenFOAM compiles a block of `controlDict` for that, and the case author writes the quantity into it. This one averages a field over the whole mesh, which no single point carries:

```
myCustomSource
{
    type coded;
    libs ("libutilityFunctionObjects.so");

    codeInclude
    #{
        #include "uniformDimensionedFields.H"
    #};

    codeExecute
    #{
        const volScalarField& p = mesh().lookupObject<volScalarField>("p");

        mesh().lookupObjectRef<uniformDimensionedScalarField>("myCustomSource").value() =
            gSum(p.primitiveField()*mesh().V())/gSum(mesh().V());
    #};
}
```

_See [`src/uqtopus/policy_src/README.md`](src/uqtopus/policy_src/README.md) for the OpenFOAM controller library details_.

## Examples

See `examples` folder for notebooks with workflow demonstrations.

## License

MIT License - see [LICENSE](LICENSE) file for details.
