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

from hystore.core.utils import valid_range_fac_inc
from hystore.processing.age_from_year_of_birth import CalculateAgeFromYearOfBirth


class TestAgeFromYearOfBirth(unittest.TestCase):

    def test_age_form_year_of_birth(self):

        raw_values = [1, 10, 100, 1000, 1900, 1910, 1920, 1930, 1940, # too old
                      1950, 1960, 1980, 2000, 2004,
                      2010, 2019, 2020, 2021 # too young
                      ]
        yobs = [''] + [str(float(v)) for v in raw_values]
        ages = np.zeros(len(yobs), dtype=np.uint32)
        flags = np.zeros_like(ages)
        print(yobs)
        fn = CalculateAgeFromYearOfBirth(0x1, 0x2, valid_range_fac_inc(16, 70), 2020)
        fn(yobs, ages, flags)
        print(yobs)
        print(ages)
        print(flags)
