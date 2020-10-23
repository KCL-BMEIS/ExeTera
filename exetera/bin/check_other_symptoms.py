import math
from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core.persistence import DataStore

def search_for_substring(hgram, substring, exclusions=tuple()):
    blood_count = 0
    hfilter = np.zeros(len(hgram), dtype=np.bool)
    for ie, e in enumerate(hgram):
        entry = e[1].lower()
        if substring in entry and entry not in exclusions:
            blood_count += e[0]
            hfilter[ie] = True
    return blood_count, hfilter


filename = '/home/ben/covid/ds_20200720_full.hdf5'
filenametemp = '/home/ben/covid/ds_temp.hdf5'
ds = persistence.DataStore()
with h5py.File(filename, 'r') as src:
    with utils.Timer("getting other symptoms")
        other_symptoms = ds.get_reader(src['assessments'])[:]
