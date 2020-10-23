import math
from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core.persistence import DataStore



def compare_datasets(ds, src1, src2):

    spaces1 = set(src1.keys())
    spaces2 = set(src2.keys())
    print(spaces1.difference(spaces2))
    print(spaces2.difference(spaces1))
    for sk in spaces1.intersection(spaces2):
        print(sk)
        sp1 = set(src1[sk].keys())
        sp2 = set(src2[sk].keys())
        print(sp1.difference(sp2))
        print(sp2.difference(sp1))


if __name__ == '__main__':
    datastore = DataStore()
    src_file_1 = '/home/ben/covid/ds_20200731_full.hdf5'
    src_file_2 = '/home/ben/covid/ds_20200731_sessiontest_full.hdf5'
    with h5py.File(src_file_1, 'r') as src_1:
        with h5py.File(src_file_2, 'r') as src_2:
            compare_datasets(datastore, src_1, src_2)
