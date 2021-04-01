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
import io

from exetera.core import csvdataset, utils

small_dataset = ('id,patient_id,foo,bar\n'
                 '0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,11111111111111111111111111111111,,a\n'
                 '07777777777777777777777777777777,33333333333333333333333333333333,True,b\n'
                 '02222222222222222222222222222222,11111111111111111111111111111111,False,a\n')

sorting_dataset = ('id,patient_id,created_at,updated_at\n'
                   'a_1,p_1,100,100\n'
                   'a_2,p_1,101,102\n'
                   'a_3,p_2,101,101\n'
                   'a_4,p_1,101,101\n'
                   'a_5,p_2,102,102\n'
                   'a_6,p_1,102,102\n'
                   'a_7,p_2,102,103\n'
                   'a_8,p_1,102,104\n'
                   'a_9,p_2,103,105\n'
                   'a_10,p_2,104,105\n'
                   'a_11,p_1,104,104\n')


class TestDataset(unittest.TestCase):

    def test_construction(self):
        s = io.StringIO(small_dataset)
        ds = csvdataset.Dataset(s, verbose=False)

        # field names and fields must match in length
        self.assertEqual(len(ds.names_), len(ds.fields_))

        self.assertEqual(ds.row_count(), 3)

        self.assertEqual(ds.names_, ['id', 'patient_id', 'foo', 'bar'])

        expected_values = [(0, ['0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', '11111111111111111111111111111111', '', 'a']),
                           (1, ['07777777777777777777777777777777', '33333333333333333333333333333333', 'True', 'b']),
                           (2, ['02222222222222222222222222222222', '11111111111111111111111111111111', 'False', 'a'])]

        # value works as expected
        for row in range(len(expected_values)):
            for col in range(len(expected_values[0][1])):
                self.assertEqual(ds.value(row, col), expected_values[row][1][col])

        # value_from_fieldname works as expected
        sorted_names = sorted(ds.names_)
        for n in sorted_names:
            index = ds.names_.index(n)
            for row in range(len(expected_values)):
                self.assertEqual(ds.value_from_fieldname(row, n), expected_values[row][1][index])

    def test_construction_with_early_filter(self):
        s = io.StringIO(small_dataset)
        ds = csvdataset.Dataset(s, early_filter=('bar', lambda x: x in ('a',)), verbose=False)

        # field names and fields must match in length
        self.assertEqual(len(ds.names_), len(ds.fields_))

        self.assertEqual(ds.row_count(), 2)

        self.assertEqual(ds.names_, ['id', 'patient_id', 'foo', 'bar'])

        expected_values = [(0, ['0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', '11111111111111111111111111111111', '', 'a']),
                           (2, ['02222222222222222222222222222222', '11111111111111111111111111111111', 'False', 'a'])]

        # value works as expected
        for row in range(len(expected_values)):
            for col in range(len(expected_values[0][1])):
                self.assertEqual(ds.value(row, col), expected_values[row][1][col])

        # value_from_fieldname works as expected
        sorted_names = sorted(ds.names_)
        for n in sorted_names:
            index = ds.names_.index(n)
            for row in range(len(expected_values)):
                self.assertEqual(ds.value_from_fieldname(row, n), expected_values[row][1][index])

    def test_sort(self):
        s = io.StringIO(small_dataset)
        ds = csvdataset.Dataset(s, verbose=False)

        ds.sort(('patient_id', 'id'))
        row_permutations = [2, 0, 1]

    def test_apply_permutation(self):
        permutation = [2, 0, 1]
        values = ['a', 'b', 'c']
        # temp_index = -1
        # temp_value = None
        # empty_index = -1
        #
        # for ip in range(len(permutation)):
        #     p = permutation[ip]
        #     if p != ip:
        #         if temp_index != -1:
        #             # move the temp index back into the empty space, it will be moved later
        #             if ip == empty_index:
        #                 # can move the current element (index p) to the destination
        #                 if temp_index != p:
        #                     # move the item from its current location

        n = len(permutation)
        for i in range(0, n):
            pi = permutation[i]
            while pi < i:
                pi = permutation[pi]
            values[i], values[pi] = values[pi], values[i]
        self.assertListEqual(['c', 'a', 'b'], values)

    def test_single_key_sorts(self):
        ds1 = csvdataset.Dataset(io.StringIO(sorting_dataset), verbose=False)
        ds1.sort('patient_id')
        self.assertListEqual([0, 1, 3, 5, 7, 10, 2, 4, 6, 8, 9], ds1.index_)

        ds2 = csvdataset.Dataset(io.StringIO(sorting_dataset), verbose=False)
        ds2.sort(('patient_id',))
        self.assertListEqual([0, 1, 3, 5, 7, 10, 2, 4, 6, 8, 9], ds2.index_)

    def test_multi_key_sorts(self):
        expected_ids =\
            ['a_1', 'a_2', 'a_4', 'a_6', 'a_8', 'a_11', 'a_3', 'a_5', 'a_7', 'a_9', 'a_10']
        expected_pids =\
            ['p_1', 'p_1', 'p_1', 'p_1', 'p_1', 'p_1', 'p_2', 'p_2', 'p_2', 'p_2', 'p_2']
        expected_vals1 =\
            ['100', '101', '101', '102', '102', '104', '101', '102', '102', '103', '104']
        expected_vals2 =\
            ['100', '102', '101', '102', '104', '104', '101', '102', '103', '105', '105']

        ds1 = csvdataset.Dataset(io.StringIO(sorting_dataset), verbose=False)
        ds1.sort('created_at')
        ds1.sort('patient_id')
        self.assertListEqual([0, 1, 3, 5, 7, 10, 2, 4, 6, 8, 9], ds1.index_)
        self.assertListEqual(expected_ids, ds1.field_by_name('id'))
        self.assertListEqual(expected_pids, ds1.field_by_name('patient_id'))
        self.assertListEqual(expected_vals1, ds1.field_by_name('created_at'))
        self.assertListEqual(expected_vals2, ds1.field_by_name('updated_at'))
        # for i in range(ds1.row_count()):
        #     utils.print_diagnostic_row("{}".format(i), ds1, i, ds1.names_)

        ds2 = csvdataset.Dataset(io.StringIO(sorting_dataset), verbose=False)
        ds2.sort(('patient_id', 'created_at'))
        self.assertListEqual([0, 1, 3, 5, 7, 10, 2, 4, 6, 8, 9], ds2.index_)
        self.assertListEqual(expected_ids, ds1.field_by_name('id'))
        self.assertListEqual(expected_pids, ds1.field_by_name('patient_id'))
        self.assertListEqual(expected_vals1, ds1.field_by_name('created_at'))
        self.assertListEqual(expected_vals2, ds1.field_by_name('updated_at'))
        # for i in range(ds2.row_count()):
        #     utils.print_diagnostic_row("{}".format(i), ds2, i, ds2.names_)
