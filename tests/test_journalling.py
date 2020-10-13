import unittest

from datetime import datetime

import numpy as np
from io import BytesIO

import h5py

from hystore.core import operations as ops
from hystore.core import session
from hystore.core import fields
from hystore.core import persistence as per
from hystore.core import journal


class Schema:
    def __init__(self):
        self.fields = dict({'id': None, 'val': None})


class TestSessionMerge(unittest.TestCase):

    def test_journal_full(self):

        ts1 = datetime(2020, 8, 1).timestamp()
        ts2 = datetime(2020, 9, 1).timestamp()
        tsf = ops.MAX_TIMESTAMP

        d1_id = np.chararray(5)
        d1_id[:] = np.asarray(['a', 'b', 'c', 'e', 'g'])
        d1_v1 = np.asarray([100, 200, 300, 500, 700])
        print(d1_id)

        d1_jvf = np.asarray([ts1, ts1, ts1, ts1, ts1])
        d1_jvt = np.asarray([tsf, tsf, tsf, tsf, tsf])

        d2_id = np.chararray(7)
        d2_id[:] = np.asarray(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        d2_v1 = np.asarray([100, 201, 300, 400, 501, 600, 700])

        d2_jvf = np.asarray([ts2, ts2, ts2, ts2, ts2, ts2, ts2])
        d2_jvt = np.asarray([tsf, tsf, tsf, tsf, tsf, tsf, tsf])

        d1_bytes = BytesIO()
        d2_bytes = BytesIO()
        dr_bytes = BytesIO()
        with h5py.File(d1_bytes) as d1_hf:
            with h5py.File(d2_bytes) as d2_hf:
                with h5py.File(dr_bytes) as dr_hf:
                    s = session.Session()

                    s.create_fixed_string(d1_hf, 'id', 1).data.write(d1_id)
                    s.create_numeric(d1_hf, 'val', 'int32').data.write(d1_v1)
                    s.create_timestamp(d1_hf, 'j_valid_from').data.write(d1_jvf)
                    s.create_timestamp(d1_hf, 'j_valid_to').data.write(d1_jvt)

                    s.create_fixed_string(d2_hf, 'id', 1).data.write(d2_id)
                    s.create_numeric(d2_hf, 'val', 'int32').data.write(d2_v1)
                    s.create_timestamp(d2_hf, 'j_valid_from').data.write(d2_jvf)
                    s.create_timestamp(d2_hf, 'j_valid_to').data.write(d2_jvt)

                    journal.journal_table(s, Schema(), d1_hf, d2_hf, 'id', dr_hf)
