"""
Custom exceptions for UQTOPUS.
"""

class SolverDivergedError(RuntimeError):
    """
    Exception raised when the OpenFOAM solver fails or diverges.
    
    Attributes:
        message (str): Explanation of the error.
        returncode (int): The exit code of the solver process.
        stdout (str): The standard output of the solver process.
        stderr (str): The standard error of the solver process.
    """
    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
