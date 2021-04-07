import unittest

from datetime import datetime

import numpy as np
from io import BytesIO

import h5py

from exetera.core import operations as ops
from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per
from exetera.core import journal


class Schema:
    def __init__(self):
        self.fields = dict({'id': None, 'val': None})


class TestSessionMerge(unittest.TestCase):

    def test_journal_full(self):

        ts1 = datetime(2020, 8, 1).timestamp()
        ts2 = datetime(2020, 9, 1).timestamp()
        tsf = ops.MAX_DATETIME.timestamp()

        d1_id = np.chararray(9)
        d1_id[:] = np.asarray(['a', 'a', 'b', 'b', 'c', 'e', 'e', 'e', 'g'])
        d1_v1 = np.asarray([100, 101, 200, 201, 300, 500, 501, 502, 700])
        print(d1_id)

        d1_jvf = np.asarray([ts1, ts1, ts1, ts1, ts1, ts1, ts1, ts1, ts1])
        d1_jvt = np.asarray([tsf, tsf, tsf, tsf, tsf, tsf, tsf, tsf, tsf])

        d2_id = np.chararray(7)
        d2_id[:] = np.asarray(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        d2_v1 = np.asarray([101, 202, 300, 400, 503, 600, 700])

        d2_jvf = np.asarray([ts2, ts2, ts2, ts2, ts2, ts2, ts2])
        d2_jvt = np.asarray([tsf, tsf, tsf, tsf, tsf, tsf, tsf])

        d1_bytes = BytesIO()
        d2_bytes = BytesIO()
        dr_bytes = BytesIO()
        s = session.Session()
        with session.Session() as s:
            dst1=s.open_dataset(d1_bytes,'r+','d1')
            d1_hf=dst1.create_dataframe('d1')
            d2_hf=dst1.create_dataframe('d2')
            dr_hf=dst1.create_dataframe('df')

            s.create_fixed_string(d1_hf, 'id', 1).data.write(d1_id)
            s.create_numeric(d1_hf, 'val', 'int32').data.write(d1_v1)
            s.create_timestamp(d1_hf, 'j_valid_from').data.write(d1_jvf)
            s.create_timestamp(d1_hf, 'j_valid_to').data.write(d1_jvt)

            s.create_fixed_string(d2_hf, 'id', 1).data.write(d2_id)
            s.create_numeric(d2_hf, 'val', 'int32').data.write(d2_v1)
            s.create_timestamp(d2_hf, 'j_valid_from').data.write(d2_jvf)
            s.create_timestamp(d2_hf, 'j_valid_to').data.write(d2_jvt)

            journal.journal_table(s, Schema(), d1_hf, d2_hf, 'id', dr_hf)


