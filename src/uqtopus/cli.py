"""
Command line interface.

Entry point for the `uqtopus` command. Subcommands live in their own functions
and return a process exit status.
"""

from __future__ import annotations

import argparse
import sys

from . import policy_build


def rl_build(args: argparse.Namespace) -> int:
    """
    Compile the solver-side policy library and report where it went.

    Parameters:
        args (Namespace): parsed options of the rl-build subcommand.

    Returns:
        int: process exit status.
    """
    try:
        foam = policy_build.probe_openfoam(policy_build.find_openfoam(args.foam))
        print(f"OpenFOAM   {foam.version} at {foam.bashrc}")

        onnx = policy_build.find_onnxruntime(args.onnxruntime)
        if onnx is None:
            if not args.install_onnxruntime:
                raise RuntimeError(
                    "no ONNX Runtime C++ release found. Set ONNXRUNTIME_ROOT, pass "
                    "--onnxruntime, or add --install-onnxruntime to download one."
                )
            print(f"ONNX       downloading {args.onnx_version}")
            onnx = policy_build.install_onnxruntime(args.onnx_version)
        print(f"ONNX       {onnx}")

        if args.check:
            print("checks passed, nothing built")
            return 0

        print(f"building   {foam.options}")
        library = policy_build.build(foam, onnx)
    except (RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"built      {library} ({library.stat().st_size / 1024:.0f} KB)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Assemble the argument parser.

    Returns:
        argparse.ArgumentParser: parser with every subcommand registered.
    """
    parser = argparse.ArgumentParser(prog="uqtopus", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    rl = subcommands.add_parser(
        "rl-build",
        help="compile the OpenFOAM boundary condition that runs the ONNX policy",
    )
    rl.add_argument("--foam", metavar="PATH", help="OpenFOAM etc/bashrc or installation root")
    rl.add_argument("--onnxruntime", metavar="PATH", help="ONNX Runtime C++ release root")
    rl.add_argument(
        "--install-onnxruntime",
        action="store_true",
        help="download the ONNX Runtime into ~/opt when none is found",
    )
    rl.add_argument(
        "--onnx-version",
        default=policy_build.DEFAULT_ONNX_VERSION,
        help=f"version to download (default: {policy_build.DEFAULT_ONNX_VERSION})",
    )
    rl.add_argument("--check", action="store_true", help="run the checks without building")
    rl.set_defaults(handler=rl_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run the command line interface.

    Parameters:
        argv (list of str or None): arguments, defaulting to sys.argv.

    Returns:
        int: process exit status.
    """
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
