# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import platform

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx

if platform.system() == "Windows":
    compile_extra_args = []
    link_extra_args = []
elif platform.system() == "Linux":
    compile_extra_args = ["-O3"]
    link_extra_args = ["-O3"]
elif platform.system() == "Darwin":
    compile_extra_args = ["-O3", "-std=c++11", "-mmacosx-version-min=10.9"]
    link_extra_args = ["-O3", "-stdlib=libc++", "-mmacosx-version-min=10.9"]


ext = [Extension('c_sib_optimizer_sprase',
                 ['c_sib_optimizer_sprase.pyx'],
                 extra_compile_args=compile_extra_args,
                 extra_link_args=link_extra_args,
                 language="c++")]

setup(name = 'c_sib_optimizer_sprase', ext_modules=ext, cmdclass = { 'build_ext': build_pyx })

