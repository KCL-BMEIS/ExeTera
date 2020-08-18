from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import h5py

from hystore.core import persistence
from hystore.core.utils import build_histogram
from hystore.core import exporter
filename = '/home/ben/covid/ds_20200720_full.hdf5'
filenametemp = '/home/ben/covid/ds_temp.hdf5'
ds = persistence.DataStore()
with h5py.File(filename, 'r') as src:
    with h5py.File(filenametemp, 'w') as tmp:

        s_p = src['patients']
        print(s_p.keys())

        for k in ('lsoa11cd', 'lsoa11nm', 'msoa11cd', 'msoa11nm', 'ladcd', 'outward_postcode_region'):
            d = defaultdict(int)
            f = ds.get_reader(s_p[k])[:]
            for v in f:
                d[v] += 1
            print(k, sorted(d.items(), key=lambda x: x[1], reverse=True)[:10])

