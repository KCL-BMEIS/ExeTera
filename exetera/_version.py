
#__version__ = '0.6.0b'

import re
from os import path

try:
    import importlib.metadata as imeta
except ImportError:
    import importlib_metadata as imeta

try:
    __version__ = imeta.version("exetera")
except:
    filename="pyproject.toml"
    
    if not path.exists(filename):
        this_directory = path.abspath(path.dirname(__file__))
        filename = path.abspath(path.join(this_directory, "..", filename))
    
    with open(filename) as o:
        dat=o.read()
        __version__ = re.search("version *= *\"(.*)\"", dat).group(1)

