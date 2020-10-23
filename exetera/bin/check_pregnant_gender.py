from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import h5py

from exetera.core import persistence
from exetera.core.utils import build_histogram
from exetera.core import exporter
from exetera.processing.test_type_from_mechanism import test_type_from_mechanism_v1


filename = '/home/ben/covid/ds_20200720_full.hdf5'
filenametemp = '/home/ben/covid/ds_temp.hdf5'
ds = persistence.DataStore()
with h5py.File(filename, 'r') as src:
    with h5py.File(filenametemp, 'w') as tmp:
        s_pat = src['patients']
        combos = defaultdict(int)
        g = ds.get_reader(s_pat['gender'])[:]
        p = ds.get_reader(s_pat['is_pregnant'])[:]
        for i_r in range(len(g)):
            combos[(g[i_r], p[i_r])] += 1

        sgp = sorted(list(combos.items()))
        for v in sgp:
            print(v)
