import unittest

import numpy as np
from io import BytesIO

import h5py

from hystore.core import session
from hystore.core import persistence as per


class TestSessionMerge(unittest.TestCase):

    def test_merge_left(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 300, 400, 500, 600])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
        r_vals = np.asarray([1000, 2000, 2001, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001])

        s = session.Session()
        print(s.merge_left(l_id, r_id, left_fields=(l_vals,), right_fields=(r_vals,)))

        r_vals2 = np.asarray(['a0', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1', 'f0', 'f1', 'h0', 'h1'])

        s = session.Session()
        print(s.merge_left(l_id, r_id, left_fields=(l_vals,), right_fields=(r_vals2,)))


    def test_merge_right(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 300, 400, 500, 600])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
        r_vals = np.asarray([1000, 2000, 2001, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001])

        s = session.Session()
        print(s.merge_right(l_id, r_id, left_fields=(l_vals,), right_fields=(r_vals,)))


class TestSessionSort(unittest.TestCase):

    def test_dataset_sort_index_ndarray(self):

        s = session.Session(10)
        vx = np.asarray([b'a', b'b', b'c', b'd', b'e'], dtype='S1')
        va = np.asarray([1, 2, 2, 1, 1])
        vb = np.asarray([5, 4, 3, 2, 1])

        sindex = s.dataset_sort_index((va, vb), np.arange(5, dtype='uint32'))

        ava = s.apply_index(sindex, va)
        avb = s.apply_index(sindex, vb)
        avx = s.apply_index(sindex, vx)

        self.assertListEqual([1, 1, 1, 2, 2], ava.tolist())
        self.assertListEqual([1, 2, 5, 3, 4], avb.tolist())
        self.assertListEqual([b'e', b'd', b'a', b'c', b'b'], avx.tolist())


    def test_dataset_sort_readers_writers(self):

        s = session.Session(10)
        vx = np.asarray([b'a', b'b', b'c', b'd', b'e'], dtype='S1')
        va = np.asarray([1, 2, 2, 1, 1])
        vb = np.asarray([5, 4, 3, 2, 1])

        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            s.get_fixed_string_writer(hf, 'x', 1).write(vx)
            s.get_numeric_writer(hf, 'a', 'int32').write(va)
            s.get_numeric_writer(hf, 'b', 'int32').write(vb)

            ra = s.get_reader(hf['a'])
            rb = s.get_reader(hf['b'])
            rx = s.get_reader(hf['x'])
            sindex = s.dataset_sort_index((ra, rb), np.arange(5, dtype='uint32'))

            s.apply_index(sindex, ra, ra.get_writer(hf, 'a', write_mode='overwrite'))
            s.apply_index(sindex, rb, rb.get_writer(hf, 'b', write_mode='overwrite'))
            s.apply_index(sindex, rx, rx.get_writer(hf, 'x', write_mode='overwrite'))

            self.assertListEqual([1, 1, 1, 2, 2], s.get_reader(hf['a'])[:].tolist())
            self.assertListEqual([1, 2, 5, 3, 4], s.get_reader(hf['b'])[:].tolist())
            self.assertListEqual([b'e', b'd', b'a', b'c', b'b'], s.get_reader(hf['x'])[:].tolist())


    def test_dataset_sort_index_groups(self):

        s = session.Session(10)
        vx = np.asarray([b'a', b'b', b'c', b'd', b'e'], dtype='S1')
        va = np.asarray([1, 2, 2, 1, 1])
        vb = np.asarray([5, 4, 3, 2, 1])

        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            s.get_fixed_string_writer(hf, 'x', 1).write(vx)
            s.get_numeric_writer(hf, 'a', 'int32').write(va)
            s.get_numeric_writer(hf, 'b', 'int32').write(vb)

            sindex = s.dataset_sort_index((hf['a'], hf['b']), np.arange(5, dtype='uint32'))

            s.apply_index(sindex, hf['a'], hf['a'])
            s.apply_index(sindex, hf['b'], hf['b'])
            s.apply_index(sindex, hf['x'], hf['x'])

            self.assertListEqual([1, 1, 1, 2, 2], s.get_reader(hf['a'])[:].tolist())
            self.assertListEqual([1, 2, 5, 3, 4], s.get_reader(hf['b'])[:].tolist())
            self.assertListEqual([b'e', b'd', b'a', b'c', b'b'], s.get_reader(hf['x'])[:].tolist())



class TestSessionFilter(unittest.TestCase):

    def test_apply_filter(self):

        s = session.Session(10)
        vx = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
        filt = np.asarray([True, True, False, False, True, False, True, False])

        result = s.apply_filter(filt, vx)
        self.assertListEqual([1, 2, 5, 7], result.tolist())
