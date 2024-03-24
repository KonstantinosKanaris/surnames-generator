""" Contains the project metadata,
    e.g., *title*, *version*, *summary* etc.
"""

from datetime import datetime

__MAJOR__ = 0
__MINOR__ = 1
__PATCH__ = 0

__title__ = "surnames_generator"
__version__ = ".".join([str(__MAJOR__), str(__MINOR__), str(__PATCH__)])
__summary__ = (
    "A Python project with PyTorch for generating surnames from "
    "various nationalities."
)
__author__ = "Konstantinos Kanaris"
__copyright__ = f"Copyright (C) {datetime.now().date().year}  {__author__}"
__email__ = "konskan95@outlook.com.gr"
