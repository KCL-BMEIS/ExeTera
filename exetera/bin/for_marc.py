import math
from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core.persistence import DataStore
from exetera.processing.nat_medicine_model import nature_medicine_model_1
from exetera.processing.method_paper_model import method_paper_model


def check_stuff(ds, src, start_ts):

    s_ptnts = src['patients']
    p_cats = ds.get_reader(s_ptnts['created_at'])[:]
    p_ccs = ds.get_reader(s_ptnts['country_code'])[:]

    p_filter = p_cats < start_ts
    print(np.count_nonzero(p_filter), np.count_nonzero(p_filter == False))
    p_cc_hist = sorted(utils.build_histogram(ds.apply_filter(p_filter, p_ccs)))
    print(p_cc_hist)


if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20200731_full.hdf5'
    with h5py.File(src_file, 'r') as src_data:
        start_timestamp = datetime.timestamp(datetime(2020, 6, 8))
        check_stuff(datastore, src_data, start_timestamp)
