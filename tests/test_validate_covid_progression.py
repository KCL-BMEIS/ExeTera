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

import parsing_schemas
import processing.covid_test


class TestCovidProgression1(unittest.TestCase):

    def _do_test(self, hcts_start, hcts_expected, tcps_start, tcps_expected, flags_expected, msg):
        hcts_start = np.asarray(hcts_start, dtype=np.uint8)
        hct_results = np.zeros_like(hcts_start)
        hcts_expected = np.asarray(hcts_expected, dtype=np.uint8)
        tcps_start = np.asarray(tcps_start, dtype=np.uint8)
        tcp_results = np.zeros_like(tcps_start)
        tcps_expected = np.asarray(tcps_expected, dtype=np.uint8)
        filter_flags = np.zeros_like(hcts_start, dtype=np.uint32)
        flags_expected = np.asarray(flags_expected, dtype=np.uint32)
        validator =\
            processing.covid_test.ValidateCovidTestResultsFacVersion1(
                hcts_start, tcps_start, filter_flags, None, hct_results, tcp_results, 0x1)

        validator('abcd',filter_flags, 0, len(hcts_start)-1)
        testmsg = f"'{msg}: had_covid_test results {hct_results} not equal to {hcts_expected}"
        self.assertTrue(np.array_equal(hct_results, hcts_expected), testmsg)
        self.assertTrue(np.array_equal(tcp_results, tcps_expected), testmsg)
        self.assertTrue(np.array_equal(filter_flags, flags_expected), testmsg)

    def test_covid_progression_had_covid_test(self):
        # dimensions
        # hct: na -> no,       na -> yes,       no -> yes,       na -> no -> yes,       no -> na -> yes,       yes -> na -> yes,
        #      na -> no -> na, na -> yes -> na, no -> yes -> na, na -> no -> yes -> na, no -> na -> yes -> na, yes -> na -> yes -> na
        #      error: yes -> no, error: na -> yes -> no, error: yes -> na -> no, error: na -> yes -> na -> no
        # tcp: na

        self._do_test([0, 1], [0, 1], [0, 0], [0, 0], [0, 0],
                      'hct: na -> no; tcp: na -> na')

        self._do_test([0, 2], [0, 2], [0, 1], [0, 1], [0, 0],
                      'hct: na -> yes; tcp: na -> waiting')

        self._do_test([0, 2], [0, 2], [0, 2], [0, 2], [0, 0],
                      'hct: na -> yes; tcp: na -> no')


        self._do_test([0, 2], [0, 2], [0, 1], [0, 1], [0, 0],
                      'hct: na -> yes; tcp: na -> yes')

        self._do_test([0, 2], [0, 2], [0, 2], [0, 2], [0, 0],
                      'hct: na -> yes; tcp: na -> yes')

        self._do_test([0, 2], [0, 2], [0, 3], [0, 3], [0, 0],
                      'hct: na -> yes; tcp: na -> yes')


        self._do_test([1, 2], [1, 2], [0, 1], [0, 1], [0, 0],
                      'hct: no -> yes; tcp: na -> waiting')

        self._do_test([1, 2], [1, 2], [0, 2], [0, 2], [0, 0],
                      'hct: no -> yes; tcp: na -> no')

        self._do_test([1, 2], [1, 2], [0, 3], [0, 3], [0, 0],
                      'hct: no -> yes; tcp: na -> yes')


        self._do_test([0, 2, 0], [0, 2, 2], [0, 1, 0], [0, 1, 1], [0, 0, 0],
                      'hct: na -> yes -> na; tcp: na -> waiting -> na')

        self._do_test([0, 2, 0], [0, 2, 2], [0, 2, 0], [0, 2, 2], [0, 0, 0],
                      'hct: na -> yes -> na; tcp: na -> waiting -> na')

        self._do_test([0, 2, 0], [0, 2, 2], [0, 3, 0], [0, 3, 3], [0, 0, 0],
                      'hct: na -> yes -> na; tcp: na -> waiting -> na')


        self._do_test([0, 1, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2],
                      [0, 2, 0, 2, 0, 2], [0, 2, 2, 2, 2, 2],
                      [0, 0, 0, 0, 0, 0], 'hct na->no->yes: multiple disjoint na between no')

        self._do_test([0, 1, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2],
                      [0, 3, 0, 3, 0, 3], [0, 3, 3, 3, 3, 3],
                      [0, 0, 0, 0, 0, 0], 'hct na->no->yes: multiple disjoint na between yes')

        self._do_test([0, 0, 1, 2, 0], [0, 0, 1, 2, 0], # [0, 0, 1, 2, 2],
                      [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], # [1, 1, 2, 2, 2],
                      [1, 1, 1, 1, 1], '')

        self._do_test([0, 2, 2], [0, 2, 2],
                      [0, 2, 2], [0, 2, 2],
                      [0, 0, 0], '')