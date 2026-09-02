"""
Training Progress and Log

A progress bar over the training iterations, and the same run as a fixed-width
table on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tqdm import tqdm


class TrainingLog:
    """
    Progress bar over training iterations, and a fixed-width table on disk.

    Parameters:
        path (str or Path): the table file, truncated on construction.
        total (int): number of iterations.
        echo (bool): also print every line, through the bar.
    """

    def __init__(self, path: str | Path, total: int, *, echo: bool = False) -> None:
        self.path = Path(path)
        self.path.write_text("")
        self.total = int(total)
        self.echo = echo
        self.bar = tqdm(total=self.total, desc="training", unit="iter")
        self._widths: dict[str, int] = {}

    def _write(self, line: str) -> None:
        with open(self.path, "a") as handle:
            handle.write(line + "\n")
        if self.echo:
            tqdm.write(line)

    def comment(self, *lines: str) -> None:
        """Write lines prefixed with '#'."""
        for line in lines:
            self._write(f"# {line}" if line else "#")

    def row(self, **columns: str) -> None:
        """Append one line, plus an 'elapsed' column. The first call fixes the columns."""
        elapsed = tqdm.format_interval(self.bar.format_dict["elapsed"])
        cells = dict(columns, elapsed=elapsed)

        if not self._widths:
            self._widths = {n: max(len(n), len(c)) for n, c in cells.items()}
            header = " ".join(f"{n:>{w}s}" for n, w in self._widths.items())
            self._write(header)
            self._write("-" * len(header))

        self._write(
            " ".join(
                f"{cells.get(n, ''):>{w}s}" for n, w in self._widths.items()
            )
        )

    def __iter__(self) -> Iterator[int]:
        """Yield the iteration index, advancing the bar."""
        for iteration in range(self.total):
            yield iteration
            self.bar.update(1)
        self.bar.close()
