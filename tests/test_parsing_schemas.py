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

from  exetera.processing.covid_test import ValidateCovidTestResultsFacVersion2


class MockDataset:
    def __init__(self):
        self.fields_ = None
        self.index_ = None
        self.names_ = None

    def field_to_index(self, field):
        return self.names_.index(field)

class TestParsingSchemas(unittest.TestCase):

    def test_validate_covid_test_results_version_2_no_to_yes(self):
        dataset = MockDataset()
        dataset.fields_ = [['a', 'x', 'no'],
                           ['b', 'x', 'yes']]
        dataset.index_ = [x for x in range(len(dataset.fields_))]
        dataset.names_ = ['id', 'patient_id', 'tested_covid_positive']

        filter_status = [0] * len(dataset.index_)
        results = [0] * len(dataset.index_)
        fn = ValidateCovidTestResultsFacVersion2(dataset, filter_status, None, results, 0x1)
        fn(dataset.fields_, filter_status, 0, len(dataset.fields_) - 1)

    def test_validate_covid_test_results_version_2_na_waiting_no_waiting(self):
        dataset = MockDataset()
        dataset.fields_ = [['a', 'x', ''],
                           ['b', 'x', 'waiting'],
                           ['c', 'x', 'no'],
                           ['d', 'x', 'waiting']]
        dataset.index_ = [x for x in range(len(dataset.fields_))]
        dataset.names_ = ['id', 'patient_id', 'tested_covid_positive']

        filter_status = [0] * len(dataset.index_)
        results = [0] * len(dataset.index_)
        fn = ValidateCovidTestResultsFacVersion2(dataset, filter_status, None, results, 0x1)
        fn(dataset.fields_, filter_status, 0, len(dataset.fields_) - 1)