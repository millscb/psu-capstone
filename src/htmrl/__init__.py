"""PSU Capstone project for Hierarchical Temporal Memory.

Attributes:
    __version__: Version string of the package.
    PROJECT_ROOT: Root directory of the project.
    DATA_PATH: Default path to the sample data file.
    PYTHONPATH: Python path for the project.
"""

import os
from importlib.metadata import PackageNotFoundError, version

import htmrl.agent_layer as agent_tools
import htmrl.encoder_layer as encoder_tools
import htmrl.environment as environment_tools
import htmrl.grapher as grapher_tools
import htmrl.input_layer as input_tools
import htmrl.log as log_tools

try:
    __version__ = version("htmrl")
except PackageNotFoundError:
    __version__ = "0.9.0"

# __file__ is /home/millscb/repos/htmrl/src/htmrl/__init__.py
# 1st dirname: /home/millscb/repos/htmrl/src/htmrl/
# 2nd dirname: /home/millscb/repos/htmrl/src/
# 3rd dirname: /home/millscb/repos/htmrl/ (The Project Root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# points to /home/millscb/repos/htmrl/data/easyData.xlsx
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")


__all__ = [
    "agent_tools",
    "encoder_tools",
    "environment_tools",
    "input_tools",
    "log_tools",
    "grapher_tools",
]
