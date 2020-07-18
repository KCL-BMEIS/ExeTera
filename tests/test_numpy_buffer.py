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

from hystore.processing import numpy_buffer


class TestNumpyBuffer(unittest.TestCase):

    def test_numpy_buffer(self):
        nb = numpy_buffer.NumpyBuffer(block_pow=4, dtype=np.uint32)
        for i in range(100):
            nb.append(i)
        final = nb.finalise()

        expected = np.asarray([x for x in range(100)], dtype=np.uint32)
        self.assertTrue(np.array_equal(final, expected))

    def test_numpy_buffer2(self):
        nb = numpy_buffer.NumpyBuffer2(block_pow=4, dtype=np.uint32)
        for i in range(100):
            nb.append(i)
        final = nb.finalise()

        expected = np.asarray([x for x in range(100)], dtype=np.uint32)
        self.assertTrue(np.array_equal(final, expected))

    def test_list_buffer(self):
        nb = numpy_buffer.ListBuffer(block_pow=4)
        for i in range(100):
            nb.append(i)
        final = nb.finalise()

        expected = np.asarray([x for x in range(100)], dtype=np.uint32)
        self.assertTrue(np.array_equal(final, expected))
