import numpy as np
import pandas as pd

from exetera.core.session import Session
from exetera.core.utils import Timer
from exetera.core.dataframe import merge

with Session() as s:
    src = s.open_dataset('/home/ben/covid/ds_20210331_fullx.hdf5', 'r', 'src')
    dest = s.open_dataset('/home/ben/covid/ds_benchmark.hdf5', 'w', 'dest')

    src_ptnts = src['patients']
    src_asmts = src['assessments']

    print(pd.unique([2, 1, 3, 2, 3]))
    print(np.unique([2, 1, 3, 2, 3]))

    with Timer('calculating shared index'):
        v = s.get_shared_index((src_ptnts['id'], src_asmts['patient_id']))

    dest_asmts = dest.create_dataframe('assessments')

    with Timer("left join assessments <- patients"):
        merge(src_asmts, src_ptnts, dest_asmts, 'patient_id', 'id',
              ('patient_id',), ('age', 'height_cm', 'weight_kg'),
              how='left', hint_left_keys_ordered=True, hint_right_keys_ordered=True)

    print('done!')
    # from io import BytesIO
    # bio = BytesIO()
    # src = s.open_dataset(bio, 'w', 'src')
    #
    # ptnts = src.create_dataframe('patients')
    # ptnts.create_fixed_string('id', 2).data.write(
    #     np.asarray(['a', 'b', 'c', 'd', 'e'], dtype='S2'))
    # ptnts.create_numeric('age', 'int32').data.write(
    #     np.asarray([20, 40, 60, 80, 100], dtype=np.int32))
    #
    # asmts = src.create_dataframe('assessments')
    # asmts.create_fixed_string('id', 2).data.write(
    #     np.asarray(['a1', 'a2', 'a3', 'b1', 'd1', 'd2', 'e1', 'e2', 'e3'], dtype='S2'))
    # asmts.create_fixed_string('patient_id', 2).data.write(
    #     np.asarray(['a', 'a', 'a', 'b', 'd', 'd', 'e', 'e', 'e'], dtype='S2'))
    #
    # dest_asmts = src.create_dataframe('dest_assessments')
    #
    # merge(asmts, ptnts, dest_asmts, 'patient_id', 'id',
    #       ('patient_id',), ('age',),
    #       how='left', hint_left_keys_ordered=True, hint_right_keys_ordered=True)
