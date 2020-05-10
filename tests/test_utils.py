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

from utils import find_longest_sequence_of, to_escaped

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
        self.assertEqual(to_escaped(''), '')
        self.assertEqual(to_escaped('a'), 'a')
        self.assertEqual(to_escaped('a,b'), '"a,b"')
        self.assertEqual(to_escaped('a"b'), '"a""b"')
        self.assertEqual(to_escaped('"a","b"'), '"""a"",""b"""')
        self.assertEqual(to_escaped(',",'), '","","')
