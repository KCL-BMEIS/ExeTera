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
import dataset

small_dataset = ('id,patient_id,foo\n'
                 '0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,11111111111111111111111111111111,,\n'
                 '07777777777777777777777777777777,33333333333333333333333333333333,True,\n'
                 '02222222222222222222222222222222,11111111111111111111111111111111,False,\n')

class TestDataset(unittest.TestCase):

    def test_construction(self):
        s = io.StringIO(small_dataset)
        ds = dataset.Dataset(s)

        # field names and fields must match in length
        self.assertEqual(len(ds.names_), len(ds.fields_))

        self.assertEqual(ds.row_count(), 3)

        self.assertEqual(ds.names_, ['id', 'patient_id', 'foo'])

        expected_values = [(0, ['0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', '11111111111111111111111111111111', '']),
                           (1, ['07777777777777777777777777777777', '33333333333333333333333333333333', 'True']),
                           (2, ['02222222222222222222222222222222', '11111111111111111111111111111111', 'False'])]

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
        ds = dataset.Dataset(s)

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

        print(values)
