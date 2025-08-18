from setuptools import setup, find_packages

setup(
    name="uqtopus",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pyDOE3",
        "tqdm",
        "jinja2",
        "pyyaml",
        "xarray",
        "fluidfoam"
    ],
)