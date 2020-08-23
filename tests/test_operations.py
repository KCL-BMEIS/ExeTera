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

    def test_ordered_map_to_right_both_unique(self):
        raw_ids = [0, 1, 2, 3, 5, 6, 7, 9]
        a_ids = np.asarray(raw_ids, dtype=np.int64)
        b_ids = np.asarray([1, 2, 3, 4, 5, 7, 8, 9], dtype=np.int64)
        results = np.zeros(len(b_ids), dtype=np.int64)
        ops.ordered_map_to_right_both_unique(a_ids, b_ids, results)
        expected = np.array([1, 2, 3, ops.INVALID_INDEX, 4, 6, ops.INVALID_INDEX, 7],
                            dtype=np.int64)
        self.assertTrue(np.array_equal(expected, results))

    def test_ordered_map_to_right_left_unique(self):
        raw_ids = [0, 1, 2, 3, 5, 6, 7, 9]
        a_ids = np.asarray(raw_ids, dtype=np.int64)
        b_ids = np.asarray([1, 2, 3, 4, 5, 7, 8, 9], dtype=np.int64)
        results = np.zeros(len(b_ids), dtype=np.int64)
        ops.ordered_map_to_right_left_unique(a_ids, b_ids, results)
        expected = np.array([1, 2, 3, ops.INVALID_INDEX, 4, 6, ops.INVALID_INDEX, 7],
                            dtype=np.int64)
        self.assertTrue(np.array_equal(results, expected))


    def test_ordered_map_to_right_left_unique_streamed(self):
        s = session.Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            a_ids = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18],
                               dtype=np.int64)
            b_ids = np.asarray([0, 1, 1, 2, 4, 5, 5, 6, 8, 9, 9, 10, 12, 13, 13, 14,
                                16, 17, 17, 18], dtype=np.int64)
            a_ids_f = s.create_numeric(hf, 'a_ids', 'int64')
            a_ids_f.data.write(a_ids)
            b_ids_f = s.create_numeric(hf, 'b_ids', 'int64')
            b_ids_f.data.write(b_ids)
            left_to_right_result = s.create_numeric(hf, 'left_result', 'int64')
            ops.ordered_map_to_right_left_unique_streamed(a_ids_f, b_ids_f,
                                                       left_to_right_result)
            print(left_to_right_result.data[:])

    def test_ordered_inner_map_result_size(self):
        a_ids = np.asarray([0, 1, 2, 2, 3, 5, 5, 5, 6, 8], dtype=np.int64)
        b_ids = np.asarray([1, 1, 2, 3, 5, 5, 6, 7, 8, 8, 8], dtype=np.int64)
        result_size = ops.ordered_inner_map_result_size(a_ids, b_ids)
        self.assertEqual(result_size, 15)
        result_size = ops.ordered_inner_map_result_size(b_ids, a_ids)
        self.assertEqual(result_size, 15)

    def test_ordered_inner_map_both_unique(self):
        a_ids = np.asarray([0, 1, 2, 3, 5, 6, 8], dtype=np.int64)
        b_ids = np.asarray([1, 2, 3, 5, 6, 7, 8], dtype=np.int64)
        result_size = ops.ordered_inner_map_result_size(a_ids, b_ids)

        a_map = np.zeros(result_size, dtype=np.int64)
        b_map = np.zeros(result_size, dtype=np.int64)
        ops.ordered_inner_map_both_unique(a_ids, b_ids, a_map, b_map)
        expected_a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        expected_b = np.array([0, 1, 2, 3, 4, 6], dtype=np.int64)
        self.assertTrue(np.array_equal(a_map, expected_a))
        self.assertTrue(np.array_equal(b_map, expected_b))

        a_map = np.zeros(result_size, dtype=np.int64)
        b_map = np.zeros(result_size, dtype=np.int64)
        ops.ordered_inner_map_left_unique(a_ids, b_ids, a_map, b_map)
        expected_a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        expected_b = np.array([0, 1, 2, 3, 4, 6], dtype=np.int64)
        self.assertTrue(np.array_equal(a_map, expected_a))
        self.assertTrue(np.array_equal(b_map, expected_b))

        a_map = np.zeros(result_size, dtype=np.int64)
        b_map = np.zeros(result_size, dtype=np.int64)
        ops.ordered_inner_map(a_ids, b_ids, a_map, b_map)
        expected_a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        expected_b = np.array([0, 1, 2, 3, 4, 6], dtype=np.int64)
        self.assertTrue(np.array_equal(a_map, expected_a))
        self.assertTrue(np.array_equal(b_map, expected_b))

    def test_ordered_inner_map_left_unique(self):
        a_ids = np.asarray([0, 1, 2, 3, 5, 6, 8], dtype=np.int64)
        b_ids = np.asarray([1, 1, 2, 3, 5, 5, 6, 7, 8, 8, 8], dtype=np.int64)
        result_size = ops.ordered_inner_map_result_size(a_ids, b_ids)

        a_map = np.zeros(result_size, dtype=np.int64)
        b_map = np.zeros(result_size, dtype=np.int64)
        ops.ordered_inner_map_left_unique(a_ids, b_ids, a_map, b_map)
        expected_a = np.array([1, 1, 2, 3, 4, 4, 5, 6, 6, 6], dtype=np.int64)
        expected_b = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10], dtype=np.int64)
        self.assertTrue(np.array_equal(a_map, expected_a))
        self.assertTrue(np.array_equal(b_map, expected_b))

        a_map = np.zeros(result_size, dtype=np.int64)
        b_map = np.zeros(result_size, dtype=np.int64)
        ops.ordered_inner_map(a_ids, b_ids, a_map, b_map)
        expected_a = np.array([1, 1, 2, 3, 4, 4, 5, 6, 6, 6], dtype=np.int64)
        expected_b = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10], dtype=np.int64)
        self.assertTrue(np.array_equal(a_map, expected_a))
        self.assertTrue(np.array_equal(b_map, expected_b))

    def test_ordered_inner_map(self):
        a_ids = np.asarray([0, 1, 2, 2, 3, 5, 5, 5, 6, 8], dtype=np.int64)
        b_ids = np.asarray([1, 1, 2, 3, 5, 5, 6, 7, 8, 8, 8], dtype=np.int64)
        result_size = ops.ordered_inner_map_result_size(a_ids, b_ids)
        a_map = np.zeros(result_size, dtype=np.int64)
        b_map = np.zeros(result_size, dtype=np.int64)
        ops.ordered_inner_map(a_ids, b_ids, a_map, b_map)
        expected_a = np.array([1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 9], dtype=np.int64)
        expected_b = np.array([0, 1, 2, 2, 3, 4, 5, 4, 5, 4, 5, 6, 8, 9, 10], dtype=np.int64)
        self.assertTrue(np.array_equal(a_map, expected_a))
        self.assertTrue(np.array_equal(b_map, expected_b))

    def test_ordered_inner_map_left_unique_streamed(self):
        s = session.Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            a_ids = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18],
                               dtype=np.int64)
            b_ids = np.asarray([0, 1, 1, 2, 4, 5, 5, 6, 8, 9, 9, 10, 12, 13, 13, 14,
                                16, 17, 17, 18], dtype=np.int64)
            a_ids_f = s.create_numeric(hf, 'a_ids', 'int64')
            a_ids_f.data.write(a_ids)
            b_ids_f = s.create_numeric(hf, 'b_ids', 'int64')
            b_ids_f.data.write(b_ids)
            left_result = s.create_numeric(hf, 'left_result', 'int64')
            right_result = s.create_numeric(hf, 'right_result', 'int64')
            ops.ordered_inner_map_left_unique_streamed(a_ids_f, b_ids_f,
                                                       left_result, right_result)
            print(left_result.data[:])
            print(right_result.data[:])
