import unittest

import numpy as np
from io import BytesIO

import h5py

from hystore.core import session
from hystore.core import fields
from hystore.core import persistence as per
from hystore.core import operations as ops



class TestAggregation(unittest.TestCase):


    def test_non_indexed_apply_spans(self):
        values = np.asarray([1, 2, 3, 3, 2, 1, 1, 2, 2, 1, 1], dtype=np.int32)
        spans = np.asarray([0, 3, 6, 8, 10, 11], dtype=np.int32)
        dest = np.zeros(len(spans)-1, dtype=np.int32)
        ops.apply_spans_index_of_min(spans, values, dest)
        print(dest)
        ops.apply_spans_index_of_max(spans, values, dest)
        print(dest)
        ops.apply_spans_index_of_first(spans, dest)
        print(dest)
        ops.apply_spans_index_of_last(spans, dest)
        print(dest)

    def test_non_indexed_apply_spans_filter(self):
        values = np.asarray([1, 2, 3, 3, 2, 1, 1, 2, 2, 1, 1], dtype=np.int32)
        spans = np.asarray([0, 3, 6, 8, 10, 11], dtype=np.int32)
        dest = np.zeros(len(spans)-1, dtype=np.int32)
        flt = np.zeros(len(spans)-1, dtype=np.int32)
        ops.apply_spans_index_of_min_filter(spans, values, dest, flt)
        print(dest, flt)
        ops.apply_spans_index_of_max_filter(spans, values, dest, flt)
        print(dest, flt)
        ops.apply_spans_index_of_first_filter(spans, dest, flt)
        print(dest, flt)
        ops.apply_spans_index_of_last_filter(spans, dest, flt)
        print(dest, flt)

        spans = np.asarray([0, 3, 3, 6, 8, 8, 10, 11], dtype=np.int32)
        dest = np.zeros(len(spans)-1, dtype=np.int32)
        flt = np.zeros(len(spans)-1, dtype=np.int32)
        ops.apply_spans_index_of_min_filter(spans, values, dest, flt)
        print(dest, flt)
        ops.apply_spans_index_of_max_filter(spans, values, dest, flt)
        print(dest, flt)
        ops.apply_spans_index_of_first_filter(spans, dest, flt)
        print(dest, flt)
        ops.apply_spans_index_of_last_filter(spans, dest, flt)
        print(dest, flt)
