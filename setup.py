from setuptools import setup

from exetera import __version__

setup(
    name='exetera',
    version=__version__,
    description='High-volume key-value store and analytics, based on hdf5',
    long_description="file: README.md",
    long_description_content_type = "text / markdown; charset = UTF - 8",
    url='https://github.com/kcl-bmeis/ExeTera',
    author='Ben Murray',
    author_email='benjamin.murray@kcl.ac.uk',
    license='http://www.apache.org/licenses/LICENSE-2.0',
    packages=['exetera', 'exetera.core', 'exetera.processing'],
    scripts=['exetera/bin/exetera'],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'numba',
        'pandas',
        'h5py'
    ]
)
