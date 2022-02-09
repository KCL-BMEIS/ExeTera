from setuptools import setup, find_packages
from pkg_resources import parse_requirements
from Cython.Build import cythonize

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# read the version and other strings from _version.py
with open(path.join(this_directory, "exetera", "_version.py")) as o:
    exec(o.read())

# read install requirements from requirements.txt
with open(path.join(this_directory, "requirements.txt")) as o:
    requirements = [str(r) for r in parse_requirements(o.read())]

pyxfiles = ['ops.pyx']
pyx_full_path = [path.join(this_directory, 'exetera', '_libs', pyx) for pyx in pyxfiles]

setup(
    name='exetera',
    version=__version__,
    description='High-volume key-value store and analytics, based on hdf5',
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url='https://github.com/kcl-bmeis/ExeTera',
    author='Ben Murray',
    author_email='benjamin.murray@kcl.ac.uk',
    license='http://www.apache.org/licenses/LICENSE-2.0',
    packages=find_packages(),
    scripts=['exetera/bin/exetera'],
    ext_modules = cythonize(pyx_full_path),
    python_requires='>=3.7',
    install_requires=requirements
)
