import os
import platform
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import versioneer


with open("README.md", "r") as fh:
    long_description = fh.read()

if platform.system() == "Windows":
    compile_extra_args = []
    link_extra_args = []
elif platform.system() == "Linux":
    compile_extra_args = ["-O3"]
    link_extra_args = ["-O3"]
elif platform.system() == "Darwin":
    compile_extra_args = ["-O3", "-std=c++11", "-mmacosx-version-min=10.9"]
    link_extra_args = ["-O3", "-stdlib=libc++", "-mmacosx-version-min=10.9"]

extensions = [
    Extension('sib.c_package.c_sib_optimizer', # name/path of generated .so file
              ['src/sib/c_package/c_sib_optimizer.pyx'], # cython file
              extra_compile_args=compile_extra_args,
              extra_link_args=link_extra_args,
              language="c++")
]

# collecting all dependencies from requirements.txt
current_dir = os.path.dirname(os.path.realpath(__file__))
requirements_file = os.path.join(current_dir, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_file):
    with open(requirements_file) as f:
        install_requires = f.read().splitlines()
else:
    raise Exception('error: unable to locate requirements.txt')


setup(
    name="sib-clustering",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Assaf Toledo",
    author_email="assaf.toledo@ibm.com",
    description="sequential Information Bottleneck",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/sib",
    include_package_data=True,
    packages=['sib',
              'sib.c_package',
    ],
    package_dir={'': 'src'},
    ext_modules = cythonize(extensions),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',        
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
)
