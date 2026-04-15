"""Research package for CEO succession under candidate-access frictions."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("whogetsconsidered")
except PackageNotFoundError:  # pragma: no cover - local editable installs before metadata exists.
    __version__ = "0.1.0"

__all__ = ["__version__"]
