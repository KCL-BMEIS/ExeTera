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

from core import filtered_field


class TestFilteredIndex(unittest.TestCase):

    def test_filtered_field_method(self):
        field = [x for x in range(10)]
        field.reverse()

        filter = [1, 3, 4, 7, 8, 9]

        for f in filtered_field.filtered_field(field, filter):
            print(f)

    def test_filtered_field_class(self):

        field = [x for x in range(10)]
        field.reverse()

        filter = [1, 3, 4, 7, 8, 9]

        ff = filtered_field.FilteredField(field, filter)
        for i in range(len(ff)):
            print(i, ff[i])
