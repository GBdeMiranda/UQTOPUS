"""
Build of the solver-side policy library.

Locates an OpenFOAM installation and the ONNX Runtime C++ release, compiles the
sources in policy_src with wmake, and reports where the shared library landed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ONNX_VERSION = "1.17.3"
ONNX_URL = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    "v{version}/onnxruntime-linux-x64-{version}.tgz"
)

FOAM_BASHRC_GLOBS = (
    "/opt/openfoam*/etc/bashrc",
    "/usr/lib/openfoam/openfoam*/etc/bashrc",
    "~/OpenFOAM/OpenFOAM-*/etc/bashrc",
)


@dataclass(frozen=True)
class FoamEnv:
    """
    The parts of a sourced OpenFOAM environment the build needs.

    Attributes:
        bashrc (Path): the etc/bashrc that produces this environment.
        version (str): WM_PROJECT_VERSION.
        options (str): WM_OPTIONS.
        user_dir (Path): WM_PROJECT_USER_DIR, where user code is built.
        user_libbin (Path): FOAM_USER_LIBBIN, where the library is written.
    """

    bashrc: Path
    version: str
    options: str
    user_dir: Path
    user_libbin: Path

    @property
    def library(self) -> Path:
        return self.user_libbin / "libuqtopusPolicy.so"


def policy_sources() -> Path:
    """
    Directory holding the C++ sources shipped with the package.

    Returns:
        Path: the policy_src directory.
    """
    return Path(__file__).resolve().parent / "policy_src"


def find_openfoam(explicit: str | os.PathLike | None = None) -> Path:
    """
    Locate an OpenFOAM etc/bashrc.

    Parameters:
        explicit (path or None): a bashrc or an installation root to use
            instead of searching.

    Returns:
        Path: the bashrc to source.
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_dir():
            candidate = candidate / "etc" / "bashrc"
        if not candidate.exists():
            raise RuntimeError(f"no OpenFOAM bashrc at {candidate}")
        return candidate

    project_dir = os.environ.get("WM_PROJECT_DIR")
    if project_dir:
        candidate = Path(project_dir) / "etc" / "bashrc"
        if candidate.exists():
            return candidate

    found = []
    for pattern in FOAM_BASHRC_GLOBS:
        expanded = Path(pattern).expanduser()
        found.extend(sorted(Path(expanded.anchor).glob(str(expanded.relative_to(expanded.anchor)))))
    if not found:
        raise RuntimeError(
            "no OpenFOAM installation found. Source its etc/bashrc first, or "
            "pass the path with --foam."
        )
    return found[-1]


def probe_openfoam(bashrc: Path) -> FoamEnv:
    """
    Read the variables the build needs out of a sourced OpenFOAM environment.

    Parameters:
        bashrc (Path): the etc/bashrc to source.

    Returns:
        FoamEnv: version, platform string and the user directories.
    """
    script = (
        f'. "{bashrc}" >/dev/null 2>&1 || exit 1; '
        'printf "%s\\n%s\\n%s\\n%s\\n" "$WM_PROJECT_VERSION" "$WM_OPTIONS" '
        '"$WM_PROJECT_USER_DIR" "$FOAM_USER_LIBBIN"'
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    lines = done.stdout.splitlines()
    if done.returncode != 0 or len(lines) < 4 or not lines[0]:
        raise RuntimeError(f"sourcing {bashrc} did not produce an OpenFOAM environment")
    return FoamEnv(
        bashrc=bashrc,
        version=lines[0].strip(),
        options=lines[1].strip(),
        user_dir=Path(lines[2].strip()),
        user_libbin=Path(lines[3].strip()),
    )


def find_onnxruntime(explicit: str | os.PathLike | None = None) -> Path | None:
    """
    Locate an ONNX Runtime C++ release.

    Parameters:
        explicit (path or None): a directory to use instead of searching.

    Returns:
        Path or None: the release root, or None when nothing usable is found.
    """
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    elif os.environ.get("ONNXRUNTIME_ROOT"):
        candidates.append(Path(os.environ["ONNXRUNTIME_ROOT"]).expanduser())
    else:
        candidates.extend(sorted(Path.home().glob("opt/onnxruntime-linux-x64-*")))

    for candidate in candidates:
        if _is_onnxruntime(candidate):
            return candidate
    return None


def _is_onnxruntime(root: Path) -> bool:
    """Whether root holds both the C++ header and the shared library."""
    return (root / "include" / "onnxruntime_cxx_api.h").exists() and (
        root / "lib" / "libonnxruntime.so"
    ).exists()


def install_onnxruntime(version: str = DEFAULT_ONNX_VERSION, prefix: Path | None = None) -> Path:
    """
    Download and extract an ONNX Runtime C++ release.

    Parameters:
        version (str): release version.
        prefix (Path or None): where to extract. Defaults to ~/opt.

    Returns:
        Path: the extracted release root.
    """
    prefix = Path(prefix).expanduser() if prefix else Path.home() / "opt"
    prefix.mkdir(parents=True, exist_ok=True)
    root = prefix / f"onnxruntime-linux-x64-{version}"
    if _is_onnxruntime(root):
        return root

    with tempfile.NamedTemporaryFile(suffix=".tgz") as archive:
        urllib.request.urlretrieve(ONNX_URL.format(version=version), archive.name)
        with tarfile.open(archive.name) as tar:
            tar.extractall(prefix)
    return root


def _run_wmake(work: Path, foam: FoamEnv, onnx: Path) -> subprocess.CompletedProcess:
    """Run wmake libso in work with the OpenFOAM environment sourced."""
    script = (
        f'. "{foam.bashrc}" >/dev/null 2>&1 || exit 1; '
        f'export ONNXRUNTIME_ROOT="{onnx}"; '
        f'cd "{work}" && wmake libso'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def build(foam: FoamEnv, onnx: Path) -> Path:
    """
    Compile the policy library from the packaged sources.

    Parameters:
        foam (FoamEnv): the OpenFOAM environment to build against.
        onnx (Path): the ONNX Runtime release root.

    Returns:
        Path: the shared library that was written.
    """
    sources = policy_sources()
    work = foam.user_dir / "src" / "uqtopusPolicy"
    if work.exists():
        shutil.rmtree(work)
    (work / "Make").mkdir(parents=True)

    for item in sources.glob("*.[CH]"):
        shutil.copy(item, work / item.name)
    for item in ("files", "options"):
        shutil.copy(sources / "Make" / item, work / "Make" / item)

    done = _run_wmake(work, foam, onnx)
    if done.returncode != 0 or not foam.library.exists():
        tail = "\n".join(done.stderr.splitlines()[-25:])
        raise RuntimeError(f"wmake failed in {work}\n\n{tail}")
    return foam.library
