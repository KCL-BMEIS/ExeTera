#!/usr/bin/env python

# TODO: Deprecated: due for removal
import argparse
from collections import defaultdict
import numpy as np
import h5py

from exetera.core.persistence import DataStore
from exetera.core.dataset import Dataset

# England
# http://geoportal.statistics.gov.uk/datasets/index-of-multiple-deprivation-december-2019-lookup-in-england/data
# Northern Ireland
# http://geoportal.statistics.gov.uk/datasets/index-of-multiple-deprivation-december-2017-lookup-in-northern-ireland
# Scotland
# https://www.gov.scot/publications/scottish-index-of-multiple-deprivation-2020v2-ranks/
# Wales
# http://geoportal.statistics.gov.uk/datasets/index-of-multiple-deprivation-december-2019-lookup-in-wales

# https://www.statistics.digitalresources.jisc.ac.uk/dataset/2011-uk-townsend-deprivation-scores/resource/0083b0bf-9241-4d73-bfdb-da33de2bd5cc#{view-graph:{graphOptions:{hooks:{processOffset:{},bindEvents:{}}}},graphOptions:{hooks:{processOffset:{},bindEvents:{}}},view-grid:{columnsWidth:[{column:!GEO_CODE,width:369}]}}

def has_data(src, fields_to_check):
    has_data = True
    fields = list()
    for f in fields_to_check:
        try:
            fields.append(src.field_by_name(f))
        except ValueError as e:
            print(e)
            return has_data, None
    return has_data, fields


def incorporate_imd_data(ds, src, imd_src):
    print(imd_src.row_count())
    print(imd_src.names_)
    imd_map = {v.encode(): i for i, v in enumerate(imd_src.field_by_name('lsoa11cd'))}
    # imd = imd_src.field_by_name('imd')
    # imd = [int(v.replace(',', '')) for v in imd]
    # imdd = imd_src.field_by_name('imd_decile')

    has_imd, imd_fields = has_data(imd_src, ('imd', 'imd_decile'))
    has_ruc, ruc_fields = has_data(imd_src, ('ruc11cd',))
    has_tds, tds_fields = has_data(imd_src, ('tds', 'tds_quintile'))

    s_ptnts = src['patients']
    lsoas = ds.get_reader(s_ptnts['lsoa11cd'])[:]


    imd_filter = np.zeros(len(lsoas), dtype=np.bool)

    if has_imd:
        src_imd, src_imd_rank = imd_fields
        src_imd = [int(v.replace(',', '')) for v in src_imd]
        imd_rank = np.zeros(len(lsoas), dtype=np.int32)
        imd_decile = np.zeros(len(lsoas), dtype=np.int8)
    if has_ruc:
        src_ruc = ruc_fields[0]
        urban_class = np.zeros(len(lsoas), dtype=np.int8)
        uc_categories = {'': 0, 'A1': 1, 'B1': 2, 'C1': 3, 'C2': 4, 'D1': 5, 'D2': 6, 'E1': 7, 'E2': 8}
    if has_tds:
        src_tds_score, src_tds_quintile = tds_fields
        tds_score = np.zeros(len(lsoas), dtype=np.float32)
        tds_quintile = np.zeros(len(lsoas), dtype=np.int8)

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
            if has_imd:
                imd_rank[i_r] = src_imd[index]
                imd_decile[i_r] = src_imd_rank[index]
            if has_ruc:
                urban_class[i_r] = uc_categories[src_ruc[index]]
            if has_tds:
                tds_score[i_r] = src_tds_score[index]
                tds_quintile[i_r] = src_tds_quintile[index]

    print((lsoas == b'').sum(), len(lsoas))
    print(imd_filter.sum())
    print(missing_lsoas)
    print(present_lsoas)

    ds.get_numeric_writer(s_ptnts, 'has_imd_data', 'bool', writemode='overwrite').write(imd_filter)
    if has_imd:
        ds.get_numeric_writer(s_ptnts, 'imd_rank', 'int32', writemode='overwrite').write(imd_rank)
        ds.get_numeric_writer(s_ptnts, 'imd_decile', 'int8', writemode='overwrite').write(imd_decile)
    if has_ruc:
        ds.get_categorical_writer(s_ptnts, 'ruc11cd', uc_categories, writemode='overwrite').write(urban_class)
    if has_tds:
        ds.get_numeric_writer(s_ptnts, 'tds_score', 'float32', writemode='overwrite').write(tds_score)
        ds.get_numeric_writer(s_ptnts, 'tds_quintile', 'int8', writemode='overwrite').write(tds_quintile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True,
                        help='The source file into which goe data should be added')
    parser.add_argument('-g', '--geodata', required=True,
                        help="The geo data that should be added to 'source'")
    args = parser.parse_args()

    datastore = DataStore()
    with open(args.geodata) as f:
        dset = Dataset(f)

    with h5py.File(args.source, 'r+') as src_data:
        incorporate_imd_data(datastore, src_data, dset)

