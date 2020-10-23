#!/usr/bin/env python

from collections import defaultdict
from datetime import datetime, timezone
import time

import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core.persistence import DataStore
from exetera.processing.nat_medicine_model import nature_medicine_model_1
from exetera.core.dataset import Dataset

# England
# http://geoportal.statistics.gov.uk/datasets/index-of-multiple-deprivation-december-2019-lookup-in-england/data
# Northern Ireland
# http://geoportal.statistics.gov.uk/datasets/index-of-multiple-deprivation-december-2017-lookup-in-northern-ireland
# Scotland
# https://www.gov.scot/publications/scottish-index-of-multiple-deprivation-2020v2-ranks/
# Wales
# http://geoportal.statistics.gov.uk/datasets/index-of-multiple-deprivation-december-2019-lookup-in-wales


def incorporate_imd_data(ds, src, imd_src):
    print(imd_src.row_count())
    print(imd_src.names_)
    imd_map = {v.encode(): i for i, v in enumerate(imd_src.field_by_name('lsoa11cd'))}
    imd = imd_src.field_by_name('imd')
    imd = [int(v.replace(',', '')) for v in imd]
    imdd = imd_src.field_by_name('imd_decile')
    ur = imd_src.field_by_name('ruc11cd')

    s_ptnts = src['patients']
    ccs = ds.get_reader(s_ptnts['country_code'])[:]
    lsoas = ds.get_reader(s_ptnts['lsoa11cd'])[:]

    imd_filter = np.zeros(len(lsoas), dtype=np.bool)
    # indices = np.zeros(len(lsoas), dtype=np.int32)
    imd_ranks = np.zeros(len(lsoas), dtype=np.int32)
    imd_deciles = np.zeros(len(lsoas), dtype=np.int8)
    urban_class = np.zeros(len(lsoas), dtype=np.int8)
    uc_categories = {'': 0, 'A1': 1, 'B1': 2, 'C1': 3, 'C2': 4, 'D1': 5, 'D2': 6, 'E1': 7, 'E2': 8}

    missing_lsoas = defaultdict(int)
    present_lsoas = defaultdict(int)
    for i_r in range(len(lsoas)):
        index = imd_map.get(lsoas[i_r], len(imd_map))

        if index == len(imd_map):
            if lsoas[i_r] == b'':
                missing_lsoas[''] += 1
            else:
                missing_lsoas[lsoas[i_r].tobytes().decode()[0]] += 1
        else:
            present_lsoas[lsoas[i_r].tobytes().decode()[0]] += 1
        imd_filter[i_r] = index != len(imd_map)
        if index != len(imd_map):
            imd_ranks[i_r] = imd[index]
            imd_deciles[i_r] = imdd[index]
            urban_class[i_r] = uc_categories[ur[index]]
    print(np.unique(urban_class, return_counts=True))
    print((lsoas == b'').sum(), len(lsoas))
    print(imd_filter.sum())
    print(missing_lsoas)
    print(present_lsoas)

    ds.get_numeric_writer(s_ptnts, 'has_imd_data', 'bool').write(imd_filter)
    ds.get_numeric_writer(s_ptnts, 'imd_rank', 'int32').write(imd_ranks)
    ds.get_numeric_writer(s_ptnts, 'imd_decile', 'int8').write(imd_deciles)
    ds.get_categorical_writer(s_ptnts, 'ruc11cd', uc_categories).write(urban_class)


if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20201014_full.hdf5'
    src_imd_file = '/home/ben/covid/EW_lsoa11cd_lookups.csv'
    with open(src_imd_file) as f:
        dset = Dataset(f)

    with h5py.File(src_file, 'r+') as src_data:
        incorporate_imd_data(datastore, src_data, dset)

