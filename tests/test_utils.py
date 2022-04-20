# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from exetera.core import utils
from exetera.core.utils import find_longest_sequence_of, to_escaped, bytearray_to_escaped, get_min_max, validate_file_exists
from .utils import HARD_INTS, HARD_FLOATS

class TestUtils(unittest.TestCase):

    def test_find_longest_sequence_of(self):
        self.assertEqual(find_longest_sequence_of('', '`'), 0)
        self.assertEqual(find_longest_sequence_of('foo', '`'), 0)
        self.assertEqual(find_longest_sequence_of('`foo`', '`'), 1)
        self.assertEqual(find_longest_sequence_of('f`oo', '`'), 1)
        self.assertEqual(find_longest_sequence_of('foo `` bar`', '`'), 2)
        self.assertEqual(find_longest_sequence_of('``foo`` bar ```', '`'), 3)

    def test_csv_encode_decode(self):
        import csv
        from io import StringIO

        src = ['A', '"B', 'C,D', 'E"F']
        #print(src)
        with StringIO() as s:
            csvw = csv.writer(s)
            csvw.writerow(src)
            result = s.getvalue()
        #print(result)

        with StringIO(result) as s:
            csvr = csv.reader(s)
            result = next(csvr)
        #print(result)

    def test_to_escaped(self):
        self.assertEqual(to_escaped(''), '')
        self.assertEqual(to_escaped('a'), 'a')
        self.assertEqual(to_escaped('a,b'), '"a,b"')
        self.assertEqual(to_escaped('a"b'), '"a""b"')
        self.assertEqual(to_escaped('"a","b"'), '"""a"",""b"""')
        self.assertEqual(to_escaped(',",'), '","","')

    def test_bytearray_to_escaped(self):
        src = np.frombuffer(b'abcd', dtype='S1')
        dest = np.zeros(10, dtype='S1')
        self.assertTrue(
            np.array_equal(dest[:bytearray_to_escaped(src, dest)],
                           np.frombuffer(b'abcd', dtype='S1')))
        src = np.frombuffer(b'ab,cd', dtype='S1')
        dest = np.zeros(10, dtype='S1')
        self.assertTrue(
            np.array_equal(dest[:bytearray_to_escaped(src, dest)],
                           np.frombuffer(b'"ab,cd"', dtype='S1')))
        src = np.frombuffer(b'ab"cd', dtype='S1')
        dest = np.zeros(10, dtype='S1')
        self.assertTrue(
            np.array_equal(dest[:bytearray_to_escaped(src, dest)],
                           np.frombuffer(b'"ab""cd"', dtype='S1')))
        src = np.frombuffer(b'"ab","cd"', dtype='S1')
        dest = np.zeros(20, dtype='S1')
        self.assertTrue(
            np.array_equal(dest[:bytearray_to_escaped(src, dest)],
                           np.frombuffer(b'"""ab"",""cd"""', dtype='S1')))

        src1 = np.frombuffer(b'ab"cd', dtype='S1')
        src2 = np.frombuffer(b'"ab","cd"', dtype='S1')
        dest = np.zeros(30, dtype='S1')
        len1 = bytearray_to_escaped(src1, dest)
        len2 = bytearray_to_escaped(src2, dest[len1:], dest_start=len1)
        self.assertTrue(
            np.array_equal(dest[:len1 + len2],
                           np.frombuffer(b'"ab""cd""""ab"",""cd"""', dtype='S1')))


    def test_get_min_max_for_permitted_types(self):
        permitted_numeric_types_without_bool = ('float32', 'float64', 'int8', 'uint8', 'int16', 'uint16', 
                                                'int32', 'uint32', 'int64')
        expected_min_max_values = {
            'float32': (-2147483648, 2147483647),
            'float64': (-9223372036854775808, 9223372036854775807),
            'int8': (-128, 127),
            'uint8': (0, 255),
            'int16': (-32768, 32767),
            'uint16': (0, 65535),
            'int32': (-2147483648, 2147483647),
            'uint32': (0, 4294967295),
            'int64': (-9223372036854775808, 9223372036854775807)
        }
        for value_type in permitted_numeric_types_without_bool:
            (min_value, max_value) = get_min_max(value_type)
            self.assertEqual(min_value, expected_min_max_values[value_type][0])
            self.assertEqual(max_value, expected_min_max_values[value_type][1])

    def test_validate_file_exists(self):
        import os
        with self.assertRaises(FileExistsError):
            validate_file_exists('./tempfile')
        os.mkdir('./tempfile')
        with self.assertRaises(FileNotFoundError):
            validate_file_exists('./tempfile')
        os.rmdir('./tempfile')

    def test_count_flag(self):
        flag = np.array([True if i%2 == 0 else False for i in range(100)])
        output = utils.count_flag_empty(flag)
        self.assertEqual(np.sum(flag == False), output)
        output = utils.count_flag_set(flag, True)
        self.assertEqual(np.sum(flag == False), output)
        output = utils.count_flag_not_set(flag, True)
        self.assertEqual(np.sum(flag != True), output)

    def test_string_to_date(self):
        from datetime import datetime
        ts_s = '2021-11-22 11:22:33.000-0500'
        ts = utils.string_to_datetime(ts_s)
        self.assertEqual(datetime.strptime(ts_s, '%Y-%m-%d %H:%M:%S.%f%z'), ts)
        with self.assertRaises(ValueError):
            utils.string_to_datetime("foo-boo")

    def test_build_histogram(self):
        data = np.array([np.random.randint(0, 50) for i in range(1000)])
        output = utils.build_histogram(data)
        a, b = np.unique(data, return_counts=True)
        result = [(a[i], b[i]) for i in range(len(a))]
        self.assertListEqual(sorted(result), sorted(output))

    def test_convert_to_int(self):
        for i in HARD_INTS:
            with self.subTest(i):
                self.assertTrue(utils.is_int(i))
                self.assertTrue(i-1 <= utils.to_int(i) <= i+1)

    def test_convert_to_float(self):
        for i in HARD_FLOATS:
            with self.subTest(i):
                self.assertTrue(utils.is_float(i))
                if not np.isnan(i):
                    self.assertTrue(i == utils.to_float(i))

    def test_sort_mixed_list(self):
        data = [np.random.randint(0,50) for i in range(1000)]
        check_fn = utils.is_int
        sort_func = lambda x: x
        output = utils.sort_mixed_list(data, check_fn, sort_func)
        self.assertTrue(np.all(np.array(output[:-1] < output[1:])))

    #def test_guess_encoding(self):



