from setuptools import setup

from exetera import __version__

setup(
    name='exetera',
    version=__version__,
    description='High-volume key-value store and analytics, based on hdf5',
    url='https://github.com/kcl-bmeis/zoe-data-store',
    author='Ben Murray',
    author_email='benjamin.murray@kcl.ac.uk',
    license='http://www.apache.org/licenses/LICENSE-2.0',
    packages=['exetera', 'exetera.contrib',
              'exetera.core', 'exetera.covidspecific', 'exetera.processing'],
    scripts=['exetera/bin/hystorex'],
    install_requires=[
        'numpy',
        'numba',
        'pandas',
        'h5py'
    ]
)
