from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core import readerwriter as rw
from exetera.core.persistence import DataStore

def get_schema(ds, dataset):
    for s in dataset.keys():
        if s != 'trash':
            print(s)
            for k in dataset[s].keys():
                r = ds.get_reader(dataset[s][k])
                if isinstance(r, rw.IndexedStringReader):
                    print("  {}: {}".format(k, 'indexedstring'))
                elif isinstance(r, rw.FixedStringReader):
                    print("  {}: {}".format(k, r.dtype()))
                elif isinstance(r, rw.NumericReader):
                    print("  {}: {}".format(k, r.dtype()))
                elif isinstance(r, rw.CategoricalReader):
                    print("  {}: {} ({})".format(k, r.dtype(), r.keys))
                elif isinstance(r, rw.TimestampReader):
                    print("  {}: {}".format(k, 'timestamp (float64)'))

if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20200731_full.hdf5'
    with h5py.File(src_file, 'r+') as src_data:
        get_schema(datastore, src_data)
