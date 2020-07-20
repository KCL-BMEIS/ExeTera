from setuptools import setup

setup(
    name='hystore',
    version='0.2.4.dev2',
    description='High-volume key-value store and analytics, based on hdf5',
    url='https://github.com/kcl-bmeis/zoe-data-store',
    author='Ben Murray',
    author_email='benjamin.murray@kcl.ac.uk',
    license='http://www.apache.org/licenses/LICENSE-2.0',
    packages=['hystore'],
    install_requires=[
        'numpy',
        'numba',
        'pandas',
        'h5py'
    ]
)
