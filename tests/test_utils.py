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
import platform
from datetime import datetime, timezone, timedelta

import numpy as np

from exetera.core.utils import (
    DATE_TIME_EPOCH,
    find_longest_sequence_of,
    to_escaped,
    bytearray_to_escaped,
    to_timestamp,
    from_timestamp,
)

is_win = "windows" in platform.system().lower()


class TestUtils(unittest.TestCase):
    def test_find_longest_sequence_of(self):
        self.assertEqual(find_longest_sequence_of("", "`"), 0)
        self.assertEqual(find_longest_sequence_of("foo", "`"), 0)
        self.assertEqual(find_longest_sequence_of("`foo`", "`"), 1)
        self.assertEqual(find_longest_sequence_of("f`oo", "`"), 1)
        self.assertEqual(find_longest_sequence_of("foo `` bar`", "`"), 2)
        self.assertEqual(find_longest_sequence_of("``foo`` bar ```", "`"), 3)

    def test_csv_encode_decode(self):
        import csv
        from io import StringIO

        src = ["A", '"B', "C,D", 'E"F']
        print(src)
        with StringIO() as s:
            csvw = csv.writer(s)
            csvw.writerow(src)
            result = s.getvalue()
        print(result)

        with StringIO(result) as s:
            csvr = csv.reader(s)
            result = next(csvr)
        print(result)

    def test_to_escaped(self):
        self.assertEqual(to_escaped(""), "")
        self.assertEqual(to_escaped("a"), "a")
        self.assertEqual(to_escaped("a,b"), '"a,b"')
        self.assertEqual(to_escaped('a"b'), '"a""b"')
        self.assertEqual(to_escaped('"a","b"'), '"""a"",""b"""')
        self.assertEqual(to_escaped(',",'), '","","')

    def test_bytearray_to_escaped(self):
        src = np.frombuffer(b"abcd", dtype="S1")
        dest = np.zeros(10, dtype="S1")
        self.assertTrue(np.array_equal(dest[: bytearray_to_escaped(src, dest)], np.frombuffer(b"abcd", dtype="S1")))
        src = np.frombuffer(b"ab,cd", dtype="S1")
        dest = np.zeros(10, dtype="S1")
        self.assertTrue(np.array_equal(dest[: bytearray_to_escaped(src, dest)], np.frombuffer(b'"ab,cd"', dtype="S1")))
        src = np.frombuffer(b'ab"cd', dtype="S1")
        dest = np.zeros(10, dtype="S1")
        self.assertTrue(np.array_equal(dest[: bytearray_to_escaped(src, dest)], np.frombuffer(b'"ab""cd"', dtype="S1")))
        src = np.frombuffer(b'"ab","cd"', dtype="S1")
        dest = np.zeros(20, dtype="S1")
        self.assertTrue(
            np.array_equal(dest[: bytearray_to_escaped(src, dest)], np.frombuffer(b'"""ab"",""cd"""', dtype="S1"))
        )

        src1 = np.frombuffer(b'ab"cd', dtype="S1")
        src2 = np.frombuffer(b'"ab","cd"', dtype="S1")
        dest = np.zeros(30, dtype="S1")
        len1 = bytearray_to_escaped(src1, dest)
        len2 = bytearray_to_escaped(src2, dest[len1:], dest_start=len1)
        self.assertTrue(np.array_equal(dest[: len1 + len2], np.frombuffer(b'"ab""cd""""ab"",""cd"""', dtype="S1")))

    def test_to_timestamp_past(self):
        n = datetime(2020, 1, 4, 1, 2, 3)  # winter time
        ts = n.timestamp()

        self.assertAlmostEqual(ts, to_timestamp(n), 5)

    def test_to_timestamp_past_dst(self):
        n = datetime(2020, 6, 4, 1, 2, 3)  # daylight savings time
        ts = n.timestamp()

        self.assertAlmostEqual(ts, to_timestamp(n), 5)

    def test_to_timestamp_future(self):
        n = datetime(2050, 1, 4, 1, 2, 3)  # winter time
        ts = n.timestamp()

        self.assertAlmostEqual(ts, to_timestamp(n), 5)

    def test_to_timestamp_future_dst(self):
        n = datetime(2050, 6, 4, 1, 2, 3)  # daylight savings time
        ts = n.timestamp()

        self.assertAlmostEqual(ts, to_timestamp(n), 5)

    def test_to_timestamp_pre_epoch(self):
        n = DATE_TIME_EPOCH - timedelta(days=365)
        ts = -60 * 60 * 24 * 365

        self.assertAlmostEqual(ts, to_timestamp(n), 5)

        if not is_win:
            self.assertAlmostEqual(ts, n.timestamp(), 5)

    def test_to_timestamp_pre_epoch_dst(self):
        n = DATE_TIME_EPOCH - timedelta(days=180)
        ts = -60 * 60 * 24 * 180

        self.assertAlmostEqual(ts, to_timestamp(n), 5)

        if not is_win:
            self.assertAlmostEqual(ts, n.timestamp(), 5)

    def test_from_timestamp(self):
        self.assertEqual(from_timestamp(0), DATE_TIME_EPOCH)

    def test_from_timestamp_past(self):
        day = 60 * 60 * 24
        self.assertEqual(from_timestamp(day * 3), datetime(1970, 1, 4, tzinfo=timezone.utc))

    def test_from_timestamp_past_dst(self):
        day = 60 * 60 * 24
        self.assertEqual(from_timestamp(day * 100), datetime(1970, 4, 11, tzinfo=timezone.utc))

    def test_from_timestamp_future(self):
        day = 60 * 60 * 24
        self.assertEqual(from_timestamp(day * 29220), datetime(2050, 1, 1, tzinfo=timezone.utc))

    def test_from_timestamp_future_dst(self):
        day = 60 * 60 * 24
        self.assertEqual(from_timestamp(day * 29320), datetime(2050, 4, 11, tzinfo=timezone.utc))
