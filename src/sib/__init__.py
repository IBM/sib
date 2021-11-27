# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

"""
...
"""

# init file

# import cython created shared object files
import sib.c_package  # cython with cpp version

# import core functionality
from .sib_main import *


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
