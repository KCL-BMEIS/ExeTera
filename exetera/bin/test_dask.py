import dask
import dask.dataframe as dd
from exetera.core import utils

#installation
# . sudo apt-get install python3.x-dev (where 3.x is the python version number)
# . pip install dask[complete]

# filename = '/home/ben/covid/assessments_export_20200720040016.csv'
#
# with utils.Timer("loading {} with dask".format(filename)):
#     df = dd.read_csv(filename)
#     print(df.columns)
#     dtypes = {c: 'str' for c in df.columns}
#
# with utils.Timer("loading () with dask".format(filename)):
#     df = dd.read_csv(filename, dtype=dtypes)
#
# with utils.Timer("getting shape {}"):
#     x = df.shape
#     dask.compute(x)
