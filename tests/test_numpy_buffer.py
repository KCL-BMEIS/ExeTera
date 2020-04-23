import unittest

import numpy as np

import numpy_buffer


class TestNumpyBuffer(unittest.TestCase):

    def test_numpy_buffer(self):
        nb = numpy_buffer.NumpyBuffer(block_pow=4, dtype=np.uint32)
        for i in range(100):
            nb.append(i)
        final = nb.finalise()

        expected = np.asarray([x for x in range(100)], dtype=np.uint32)
        self.assertTrue(np.array_equal(final, expected))
