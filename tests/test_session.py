import unittest

import numpy as np
from io import BytesIO

import h5py

from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per


class TestSessionMerge(unittest.TestCase):

    def test_merge_left(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 300, 400, 500, 600])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
        r_vals = np.asarray([1000, 2000, 2001, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001])

        s = session.Session()
        actual = s.merge_left(l_id, r_id, right_fields=(r_vals,))

        self.assertListEqual(
            [1000, 0, 3000, 3001, 5000, 5001, 0, 6000, 6001], actual[0].tolist())

        r_vals2 = np.asarray(['a0', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1', 'f0', 'f1', 'h0', 'h1'])

        s = session.Session()
        actual = s.merge_left(l_id, r_id, right_fields=(r_vals2,))

        self.assertListEqual(
            ['a0', '', 'd0', 'd1', 'f0', 'f1', '', 'h0', 'h1'], actual[0].tolist())


    def test_merge_left_2(self):
        s = session.Session()
        p_id = np.array([100, 200, 300, 400, 500, 600, 800, 900])
        p_val = np.array([-1, -2, -3, -4, -5, -6, -8, -9])
        a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                          600, 600, 700, 700, 900, 900, 900])
        a_val = np.array([10, 11, 12, 23, 22, 40, 43, 42, 41, 60,
                          61, 63, 71, 71, 94, 93, 92])

        actual = s.merge_left(p_id, a_pid, right_fields=(a_val,))

        self.assertListEqual(
            [10, 11, 12, 23, 22, 0, 40, 43, 42, 41, 0, 60, 61, 63, 0, 94, 93, 92],
            actual[0].tolist())

    def test_merge_left_3(self):
        s = session.Session()
        p_id = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        d_pid = np.array([10, 30, 40, 60, 80])
        d_counts = np.array([2, 1, 2, 3, 2])
        d_to_p = np.zeros(len(p_id))
        import pandas as pd
        pdf = pd.DataFrame({'id': p_id})
        ddf = pd.DataFrame({'patient_id': d_pid, 'd_counts': d_counts})

        expected = [2, 0, 1, 2, 0, 3, 0, 2]

        actual = s.merge_left(left_on=p_id, right_on=d_pid, right_fields=(d_counts,))
        self.assertListEqual(expected, actual[0].tolist())

        actual = s.ordered_merge_left(left_on=p_id, right_on=d_pid,
                                      right_field_sources=(d_counts,), left_to_right_map=d_to_p,
                                      left_unique=True, right_unique=True)
        self.assertListEqual(expected, actual[0].tolist())


    def test_merge_left_dataset(self):
        bio1 = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio1, 'w', 'src')

            p_id = np.array([100, 200, 300, 400, 500, 600, 800, 900])
            p_val = np.array([-1, -2, -3, -4, -5, -6, -8, -9])
            a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                              600, 600, 700, 700, 900, 900, 900])
            a_val = np.array([10, 11, 12, 23, 22, 40, 43, 42, 41, 60,
                              61, 63, 71, 71, 94, 93, 92])
            src.create_dataframe('p')
            s.create_numeric(src['p'], 'id', 'int32').data.write(p_id)
            s.create_numeric(src['p'], 'val', 'int32').data.write(p_val)
            src.create_dataframe('a')
            s.create_numeric(src['a'], 'pid', 'int32').data.write(a_pid)

            bio2 = BytesIO()
            dst = s.open_dataset(bio2,'w','dst')
            snk=dst.create_dataframe('snk')
            s.merge_left(s.get(src['a']['pid']), s.get(src['p']['id']),
                             right_fields=(s.get(src['p']['val']),),
                             right_writers=(s.create_numeric(snk, 'val', 'int32'),))
            expected = [-1, -1, -1, -2, -2, -4, -4, -4, -4, -6, -6, -6, 0, 0, -9, -9, -9]
            actual = s.get(snk['val']).data[:]
            self.assertListEqual(expected, actual.data[:].tolist())


    def test_ordered_merge_left_2(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            hf = dst.create_dataframe('dst')

            p_id = np.array([100, 200, 300, 400, 500, 600, 800, 900])
            p_val = np.array([-1, -2, -3, -4, -5, -6, -8, -9])
            a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                              600, 600, 700, 700, 900, 900, 900])
            a_val = np.array([10, 11, 12, 23, 22, 40, 43, 42, 41, 60,
                              61, 63, 71, 71, 94, 93, 92])
            f_p_id = s.create_numeric(hf, 'p_id', 'int32')
            f_p_id.data.write(p_id)
            f_p_val = s.create_numeric(hf, 'p_val', 'int32')
            f_p_val.data.write(p_val)
            f_a_pid = s.create_numeric(hf, 'a_pid', 'int32')
            f_a_pid.data.write(a_pid)
            a_to_p = s.create_numeric(hf, 'a_to_p', 'int64')
            f_a_p_val = s.create_numeric(hf, 'a_p_val', 'int32')
            s.ordered_merge_left(f_a_pid, f_p_id, right_field_sources=(f_p_val,), left_field_sinks=(f_a_p_val,),
                                 left_to_right_map=a_to_p, right_unique=True)

            p_to_a_vals_exp = np.asarray([-1, -1, -1, -2, -2, -4, -4, -4, -4, -6, -6, -6, 0, 0, -9, -9, -9], dtype=np.int32)
            actual = s.merge_left(a_pid, p_id, right_fields=(p_val,))
            self.assertTrue(np.array_equal(actual[0], p_to_a_vals_exp))


    def test_ordered_merge_right_2(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            hf = dst.create_dataframe('dst')
            p_id = np.array([100, 200, 300, 400, 500, 600, 800, 900])
            p_val = np.array([-1, -2, -3, -4, -5, -6, -8, -9])
            a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                              600, 600, 700, 700, 900, 900, 900])
            a_val = np.array([10, 11, 12, 23, 22, 40, 43, 42, 41, 60,
                              61, 63, 71, 71, 94, 93, 92])
            f_p_id = s.create_numeric(hf, 'p_id', 'int32')
            f_p_id.data.write(p_id)
            f_p_val = s.create_numeric(hf, 'p_val', 'int32')
            f_p_val.data.write(p_val)
            f_a_pid = s.create_numeric(hf, 'a_pid', 'int32')
            f_a_pid.data.write(a_pid)
            a_to_p = s.create_numeric(hf, 'a_to_p', 'int64')
            f_a_p_val = s.create_numeric(hf, 'a_p_val', 'int32')
            s.ordered_merge_right(f_p_id, f_a_pid,
                                  left_field_sources=(f_p_val,),
                                  right_field_sinks=(f_a_p_val,),
                                  right_to_left_map=a_to_p,
                                  left_unique=True)

            a_to_p_vals_exp = np.asarray([-1, -1, -1, -2, -2, -4, -4, -4, -4, -6, -6, -6, 0, 0, -9, -9, -9], dtype=np.int32)
            actual = s.merge_left(a_pid, p_id, right_fields=(p_val,))
            self.assertTrue(np.array_equal(actual[0], a_to_p_vals_exp))


    def test_merge_right(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 300, 400, 500, 600])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
        r_vals = np.asarray([1000, 2000, 2001, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001])

        s = session.Session()
        res = s.merge_right(l_id, r_id, left_fields=(l_vals,))
        self.assertListEqual([100,   0,   0, 300, 300,   0,   0, 400, 400, 600, 600],
                             res[0].tolist())


    def test_ordered_merge_left(self):
        l_id = np.asarray([b'a', b'b', b'd', b'f', b'g', b'h'])
        l_vals = np.asarray([100, 200, 400, 600, 700, 800])
        l_vals_2 = np.asarray([10000, 20000, 40000, 60000, 70000, 80000])

        r_id = np.asarray([b'a', b'c', b'c', b'd', b'd', b'e', b'e', b'f', b'f', b'h', b'h'])
        r_vals = np.asarray([1000, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001, 8000, 8001])
        r_vals_2 = np.asarray([100000, 300001, 300000, 400001, 400000,
                               500001, 50000, 600001, 600000, 800001, 800000])

        l_vals_exp = np.asarray([100, 0, 0, 400, 400, 0, 0, 600, 600, 800, 800], dtype=np.int32)
        l_vals_2_exp = np.asarray([10000, 0, 0, 40000, 40000, 0, 0, 60000, 60000, 80000, 80000],
                                  dtype=np.int32)

        s = session.Session()
        l_to_r = np.zeros(len(r_id), dtype=np.int64)
        actual = s.ordered_merge_left(r_id, l_id, right_field_sources=(l_vals,), left_to_right_map=l_to_r,
                                      right_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))

        r_to_l = np.zeros(len(l_id), dtype=np.int64)
        actual = s.ordered_merge_right(l_id, r_id, left_field_sources=(l_vals,), right_to_left_map=r_to_l,
                                       left_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))

        l_to_r = np.zeros(len(r_id), dtype=np.int64)
        actual = s.ordered_merge_left(r_id, l_id, right_field_sources=(l_vals, l_vals_2), left_to_right_map=l_to_r,
                                      right_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))
        self.assertTrue(np.array_equal(actual[1], l_vals_2_exp))

        r_to_l = np.zeros(len(l_id), dtype=np.int64)
        actual = s.ordered_merge_right(l_id, r_id, left_field_sources=(l_vals, l_vals_2), right_to_left_map=r_to_l,
                                       left_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))
        self.assertTrue(np.array_equal(actual[1], l_vals_2_exp))


    def test_merge_inner(self):
        l_id = np.asarray([b'a', b'b', b'd', b'f', b'g', b'h'])
        l_vals = np.asarray([100, 200, 400, 600, 700, 800])
        l_vals_2 = np.asarray([10000, 20000, 40000, 60000, 70000, 80000])

        r_id = np.asarray([b'a', b'c', b'c', b'd', b'd', b'e', b'e', b'f', b'f', b'h', b'h'])
        r_vals = np.asarray([1000, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001, 8000, 8001])
        r_vals_2 = np.asarray([100000, 300001, 300000, 400001, 400000,
                               500001, 50000, 600001, 600000, 800001, 800000])

        l_vals_exp = np.asarray([100, 400, 400, 600, 600, 800, 800], dtype=np.int32)
        l_vals_2_exp = np.asarray([10000, 40000, 40000, 60000, 60000, 80000, 80000],
                                  dtype=np.int32)
        r_vals_exp = np.asarray([1000, 4000, 4001, 6000, 6001, 8000, 8001], dtype=np.int32)
        r_vals_2_exp = np.asarray([100000, 400001, 400000, 600001, 600000, 800001, 800000],
                                  dtype=np.int32)
        s = session.Session()
        actual = s.merge_inner(l_id, r_id,
                               left_fields=(l_vals, l_vals_2),
                               right_fields=(r_vals, r_vals_2))
        self.assertTrue(np.array_equal(actual[0][0], l_vals_exp))
        self.assertTrue(np.array_equal(actual[0][1], l_vals_2_exp))
        self.assertTrue(np.array_equal(actual[1][0], r_vals_exp))
        self.assertTrue(np.array_equal(actual[1][1], r_vals_2_exp))


    def test_ordered_merge_inner_fields(self):
        l_id = np.asarray([b'a', b'b', b'd', b'f', b'g', b'h'])
        l_vals = np.asarray([100, 200, 400, 600, 700, 800])
        l_vals_2 = np.asarray([10000, 20000, 40000, 60000, 70000, 80000])

        r_id = np.asarray([b'a', b'c', b'c', b'd', b'd', b'e', b'e', b'f', b'f', b'h', b'h'])
        r_vals = np.asarray([1000, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001, 8000, 8001])
        r_vals_2 = np.asarray([100000, 300001, 300000, 400001, 400000,
                               500001, 50000, 600001, 600000, 800001, 800000])

        l_vals_exp = np.asarray([100, 400, 400, 600, 600, 800, 800], dtype=np.int32)
        l_vals_2_exp = np.asarray([10000, 40000, 40000, 60000, 60000, 80000, 80000],
                                  dtype=np.int32)
        r_vals_exp = np.asarray([1000, 4000, 4001, 6000, 6001, 8000, 8001], dtype=np.int32)
        r_vals_2_exp = np.asarray([100000, 400001, 400000, 600001, 600000, 800001, 800000],
                                  dtype=np.int32)

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio,'w','dst')
            hf=dst.create_dataframe('dst')
            l_id_f = s.create_fixed_string(hf, 'l_id', 1); l_id_f.data.write(l_id)
            l_vals_f = s.create_numeric(hf, 'l_vals_f', 'int32'); l_vals_f.data.write(l_vals)
            l_vals_2_f = s.create_numeric(hf, 'l_vals_2_f', 'int32'); l_vals_2_f.data.write(l_vals_2)
            r_id_f = s.create_fixed_string(hf, 'r_id', 1); r_id_f.data.write(r_id)
            r_vals_f = s.create_numeric(hf, 'r_vals_f', 'int32'); r_vals_f.data.write(r_vals)
            r_vals_2_f = s.create_numeric(hf, 'r_vals_2_f', 'int32'); r_vals_2_f.data.write(r_vals_2)
            i_l_vals_f = s.create_numeric(hf, 'i_l_vals_f', 'int32')
            i_l_vals_2_f = s.create_numeric(hf, 'i_l_vals_2_f', 'int32')
            i_r_vals_f = s.create_numeric(hf, 'i_r_vals_f', 'int32')
            i_r_vals_2_f = s.create_numeric(hf, 'i_r_vals_2_f', 'int32')
            s.ordered_merge_inner(l_id_f, r_id_f,
                                  left_field_sources=(l_vals_f, l_vals_2_f),
                                  right_field_sources=(r_vals_f, r_vals_2_f),
                                  left_unique=True, right_unique=False,
                                  left_field_sinks=(i_l_vals_f, i_l_vals_2_f),
                                  right_field_sinks=(i_r_vals_f, i_r_vals_2_f))
            self.assertTrue(np.array_equal(i_l_vals_f.data[:], l_vals_exp))
            self.assertTrue(np.array_equal(i_l_vals_2_f.data[:], l_vals_2_exp))
            self.assertTrue(np.array_equal(i_r_vals_f.data[:], r_vals_exp))
            self.assertTrue(np.array_equal(i_r_vals_2_f.data[:], r_vals_2_exp))


    def test_ordered_merge_inner(self):
        l_id = np.asarray([b'a', b'b', b'd', b'f', b'g', b'h'], dtype='S1')
        l_vals = np.asarray([100, 200, 400, 600, 700, 800])
        l_vals_2 = np.asarray([10000, 20000, 40000, 60000, 70000, 80000])

        r_id = np.asarray([b'a', b'c', b'c', b'd', b'd', b'e', b'e', b'f', b'f', b'h', b'h'], dtype='S1')
        r_vals = np.asarray([1000, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001, 8000, 8001])
        r_vals_2 = np.asarray([100000, 300001, 300000, 400001, 400000,
                               500001, 50000, 600001, 600000, 800001, 800000])

        l_vals_exp = np.asarray([100, 400, 400, 600, 600, 800, 800], dtype=np.int32)
        l_vals_2_exp = np.asarray([10000, 40000, 40000, 60000, 60000, 80000, 80000],
                                  dtype=np.int32)
        r_vals_exp = np.asarray([1000, 4000, 4001, 6000, 6001, 8000, 8001], dtype=np.int32)
        r_vals_2_exp = np.asarray([100000, 400001, 400000, 600001, 600000, 800001, 800000],
                                  dtype=np.int32)
        s = session.Session()
        actual = s.ordered_merge_inner(l_id, r_id,
                                       left_field_sources=(l_vals, l_vals_2),
                                       right_field_sources=(r_vals, r_vals_2),
                                       left_unique=True, right_unique=False)
        self.assertTrue(np.array_equal(actual[0][0], l_vals_exp))
        self.assertTrue(np.array_equal(actual[0][1], l_vals_2_exp))
        self.assertTrue(np.array_equal(actual[1][0], r_vals_exp))
        self.assertTrue(np.array_equal(actual[1][1], r_vals_2_exp))


class TestSessionJoin(unittest.TestCase):

    def test_session_join(self):

        pk = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32)
        fki = np.asarray([0, 1, 1, 2, 4, 5, 5, 6, 8, 9, 9, 10], dtype=np.int32)
        vals = np.asarray([1, 2, 1, 1, 2, 1, 1, 2, 1], dtype=np.int64)
        s = session.Session()
        self.assertListEqual([1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0],
                             s.join(pk, fki, vals).tolist())


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
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            hf = dst.create_dataframe('dst')

            s.create_fixed_string(hf, 'x', 1).data.write(vx)
            s.create_numeric(hf, 'a', 'int32').data.write(va)
            s.create_numeric(hf, 'b', 'int32').data.write(vb)

            ra = s.get(hf['a'])
            rb = s.get(hf['b'])
            rx = s.get(hf['x'])
            sindex = s.dataset_sort_index((ra, rb), np.arange(5, dtype='uint32'))

            ra.writeable().data[:] = s.apply_index(sindex, ra)
            rb.writeable().data[:] = s.apply_index(sindex, rb)
            rx.writeable().data[:] = s.apply_index(sindex, rx)

            self.assertListEqual([1, 1, 1, 2, 2], s.get(hf['a']).data[:].tolist())
            self.assertListEqual([1, 2, 5, 3, 4], s.get(hf['b']).data[:].tolist())
            self.assertListEqual([b'e', b'd', b'a', b'c', b'b'], s.get(hf['x']).data[:].tolist())


    def test_dataset_sort_index_groups(self):

        s = session.Session(10)
        vx = np.asarray([b'a', b'b', b'c', b'd', b'e'], dtype='S1')
        va = np.asarray([1, 2, 2, 1, 1])
        vb = np.asarray([5, 4, 3, 2, 1])

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            hf = dst.create_dataframe('dst')

            s.create_fixed_string(hf, 'x', 1).data.write(vx)
            s.create_numeric(hf, 'a', 'int32').data.write(va)
            s.create_numeric(hf, 'b', 'int32').data.write(vb)

            sindex = s.dataset_sort_index((hf['a'], hf['b']), np.arange(5, dtype='uint32'))

            s.get(hf['a']).writeable().data[:] = s.apply_index(sindex, hf['a'])
            s.get(hf['b']).writeable().data[:] = s.apply_index(sindex, hf['b'])
            s.get(hf['x']).writeable().data[:] = s.apply_index(sindex, hf['x'])

            self.assertListEqual([1, 1, 1, 2, 2], s.get(hf['a']).data[:].tolist())
            self.assertListEqual([1, 2, 5, 3, 4], s.get(hf['b']).data[:].tolist())
            self.assertListEqual([b'e', b'd', b'a', b'c', b'b'], s.get(hf['x']).data[:].tolist())


    def test_sort_on(self):

        idx = np.asarray([b'a', b'e', b'b', b'd', b'c'], dtype='S1')
        val = np.asarray([10, 20, 30, 40, 50], dtype=np.int32)
        val2 = ['a', 'ee', 'bbb', 'dddd', 'ccccc']

        bio = BytesIO()
        with session.Session(10) as s:
            dst = s.open_dataset(bio, "w", "src")
            src = dst.create_dataframe('ds')
            idx_f = s.create_fixed_string(src, "idx", 1)
            val_f = s.create_numeric(src, "val", "int32")
            val2_f = s.create_indexed_string(src, "val2")
            idx_f.data.write(idx)
            val_f.data.write(val)
            val2_f.data.write(val2)
            s.sort_on(src, src, ("idx",), verbose=False)

            self.assertListEqual([b'a', b'b', b'c', b'd', b'e'], idx_f.data[:].tolist())
            self.assertListEqual([10, 30, 50, 40, 20], val_f.data[:].tolist())
            self.assertListEqual(['a', 'bbb', 'ccccc', 'dddd', 'ee'], val2_f.data[:])


class TestSessionFilter(unittest.TestCase):

    def test_apply_filter(self):

        s = session.Session(10)
        vx = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
        filt = np.asarray([True, True, False, False, True, False, True, False])

        result = s.apply_filter(filt, vx)
        self.assertListEqual([1, 2, 5, 7], result.tolist())


class TestSessionGetSpans(unittest.TestCase):

    def test_get_spans_one_field(self):

        vals = np.asarray([0, 1, 1, 3, 3, 6, 5, 5, 5], dtype=np.int32)
        bio = BytesIO()
        with session.Session() as s:
            self.assertListEqual([0, 1, 3, 5, 6, 9], s.get_spans(vals).tolist())

            dst = s.open_dataset(bio, "w", "src")
            ds = dst.create_dataframe('ds')
            vals_f = s.create_numeric(ds, "vals", "int32")
            vals_f.data.write(vals)
            self.assertListEqual([0, 1, 3, 5, 6, 9], s.get_spans(s.get(ds['vals'])).tolist())

    def test_get_spans_two_fields(self):

        vals_1 = np.asarray(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c'], dtype='S1')
        vals_2 = np.asarray([5, 5, 6, 2, 2, 3, 4, 4, 7, 7, 7, 7], dtype=np.int32)
        bio = BytesIO()
        with session.Session() as s:
            self.assertListEqual([0, 2, 3, 5, 6, 8, 12], s.get_spans(fields=(vals_1, vals_2)).tolist())

            dst = s.open_dataset(bio, "w", "src")
            ds = dst.create_dataframe('ds')
            vals_1_f = s.create_fixed_string(ds, 'vals_1', 1)
            vals_1_f.data.write(vals_1)
            vals_2_f = s.create_numeric(ds, 'vals_2', 'int32')
            vals_2_f.data.write(vals_2)
            self.assertListEqual([0, 2, 3, 5, 6, 8, 12], s.get_spans(fields=(vals_1, vals_2)).tolist())

    def test_get_spans_index_string_field(self):
        bio=BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            ds = dst.create_dataframe('ds')
            idx= s.create_indexed_string(ds,'idx')
            idx.data.write(['aa','bb','bb','c','c','c','d','d','e','f','f','f'])
            self.assertListEqual([0,1,3,6,8,9,12],s.get_spans(idx))



class TestSessionAggregate(unittest.TestCase):

    def test_apply_spans_count(self):

        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            results = s.apply_spans_count(spans)
            self.assertListEqual([1, 2, 3, 4], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.apply_spans_count(spans, dest=s.create_numeric(ds, 'result', 'int32'))
            self.assertListEqual([1, 2, 3, 4], s.get(ds['result']).data[:].tolist())

    def test_apply_spans_first(self):

        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            results = s.apply_spans_first(spans, vals)
            self.assertListEqual([0, 8, 6, 3], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.apply_spans_first(spans, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 8, 6, 3], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.apply_spans_first(spans, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 8, 6, 3], s.get(ds['result2']).data[:].tolist())

    def test_apply_spans_last(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            results = s.apply_spans_last(spans, vals)
            self.assertListEqual([0, 2, 5, 9], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.apply_spans_last(spans, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 2, 5, 9], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.apply_spans_last(spans, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 2, 5, 9], s.get(ds['result2']).data[:].tolist())

    def test_apply_spans_min(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            results = s.apply_spans_min(spans, vals)
            self.assertListEqual([0, 2, 4, 1], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.apply_spans_min(spans, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 2, 4, 1], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.apply_spans_min(spans, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 2, 4, 1], s.get(ds['result2']).data[:].tolist())

    def test_apply_spans_max(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            results = s.apply_spans_max(spans, vals)
            self.assertListEqual([0, 8, 6, 9], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.apply_spans_max(spans, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 8, 6, 9], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.apply_spans_max(spans, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 8, 6, 9], s.get(ds['result2']).data[:].tolist())

    def test_apply_spans_concat(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            self.assertListEqual([0, 1, 3, 6, 10], spans.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.create_indexed_string(ds, 'vals').data.write(vals)
            s.apply_spans_concat(spans, s.get(ds['vals']), dest=s.create_indexed_string(ds, 'result'))
            self.assertListEqual([0, 1, 4, 9, 16], s.get(ds['result']).indices[:].tolist())
            self.assertListEqual(['a', 'b,a', 'b,a,b', 'a,b,a,b'], s.get(ds['result']).data[:])

    def test_apply_spans_concat_2(self):
        idx = np.asarray([0, 0, 1, 2, 2, 3, 4, 4, 4, 4], dtype=np.int32)
        vals = ['a', 'b,c', 'd', 'e,f', 'g', 'h,i', 'j', 'k,l', 'm', 'n,o']
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            self.assertListEqual([0, 2, 3, 5, 6, 10], spans.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.create_indexed_string(ds, 'vals').data.write(vals)
            s.apply_spans_concat(spans, s.get(ds['vals']), dest=s.create_indexed_string(ds, 'result'))
            self.assertListEqual([0, 7, 8, 15, 20, 35], s.get(ds['result']).indices[:].tolist())
            self.assertListEqual(['a,"b,c"', 'd', '"e,f",g', '"h,i"', 'j,"k,l",m,"n,o"'],
                                 s.get(ds['result']).data[:])

    def test_apply_spans_concat_field(self):
        idx = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
        vals = ['a', "'b'", 'what', 'some, information', 'x',
               '', 'foo', 'flop',
               "'dun'", "'mun'", "'race, track?'", '', "for, too", 'z', 'now!']

        # vals = ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']
        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            # results = s.apply_spans_concat(spans, vals)
            # self.assertListEqual([0, 8, 6, 9], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            # s.apply_spans_concat(spans, vals, dest=s.create_indexed_string(ds, 'result'))
            # self.assertListEqual([0, 8, 6, 9], s.get(ds['result']).data[:].tolist())

            s.create_indexed_string(ds, 'vals').data.write(vals)
            s.apply_spans_concat(spans, s.get(ds['vals']), dest=s.create_indexed_string(ds, 'result'))
            self.assertListEqual(['a,\'b\',what,"some, information",x', 'foo,flop',
                                  '\'dun\',\'mun\',"\'race, track?\'","for, too",z,now!'],
                                 s.get(ds['result']).data[:])

    def test_apply_spans_concat_small_chunk_size(self):
        idx = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 3, 3,
                          4, 4, 4, 5, 5, 6, 6, 6, 7, 7,
                          8, 8, 8, 9, 9, 10, 10, 10, 11, 11,
                          12, 12, 12, 13, 13, 14, 14, 14, 15, 15,
                          16, 16, 16, 17, 17, 18, 18, 18, 19, 19])
        vals = ['a', 'b,c', '', 'd', 'e,f', '', 'g', 'h,i', '', 'j',
                'k,l', '', 'm', 'n,o', '', 'p', 'q,r', '', 's', 't,u',
                '', 'v', 'w,x', '', 'y', 'z,aa', '', 'ab', 'ac,ad', '',
                'ae', 'af,ag', '', 'ah', 'ai,aj', '', 'ak', 'al,am', '', 'an',
                'ao,ap', '', 'aq', 'ar,as', '', 'at', 'au,av', '', 'aw', 'ax,ay']

        bio = BytesIO()
        with session.Session() as s:
            spans = s.get_spans(idx)
            self.assertListEqual([0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28,
                                  30, 33, 35, 38, 40, 43, 45, 48, 50], spans.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.create_indexed_string(ds, 'vals').data.write(vals)

            expected_indices = [0,
                                7, 14, 21, 22, 29, 34, 41, 48, 55, 56,
                                65, 72, 82, 92, 102, 104, 114, 121, 131, 141]
            expected_data = ['a,"b,c"', 'd,"e,f"', 'g,"h,i"', 'j', '"k,l",m',
                             '"n,o"', 'p,"q,r"', 's,"t,u"', 'v,"w,x"', 'y',
                             '"z,aa",ab', '"ac,ad"', 'ae,"af,ag"', 'ah,"ai,aj"', 'ak,"al,am"',
                             'an', '"ao,ap",aq', '"ar,as"', 'at,"au,av"', 'aw,"ax,ay"']

            s.apply_spans_concat(spans, s.get(ds['vals']), dest=s.create_indexed_string(ds, 'result'))
            self.assertListEqual(expected_indices, s.get(ds['result']).indices[:].tolist())
            self.assertListEqual(expected_data, s.get(ds['result']).data[:])

            s.apply_spans_concat(spans, s.get(ds['vals']), dest=s.create_indexed_string(ds, 'result2'),
                                 src_chunksize=16, dest_chunksize=16)
            self.assertListEqual(expected_indices, s.get(ds['result2']).indices[:].tolist())
            self.assertListEqual(expected_data, s.get(ds['result2']).data[:])


    def test_aggregate_count(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        bio = BytesIO()
        with session.Session() as s:
            results = s.aggregate_count(idx)
            self.assertListEqual([1, 2, 3, 4], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.aggregate_count(idx, dest=s.create_numeric(ds, 'result', 'int32'))
            self.assertListEqual([1, 2, 3, 4], s.get(ds['result']).data[:].tolist())

    def test_aggregate_first(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            results = s.aggregate_first(idx, vals)
            self.assertListEqual([0, 8, 6, 3], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.aggregate_first(idx, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 8, 6, 3], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.aggregate_first(idx, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 8, 6, 3], s.get(ds['result2']).data[:].tolist())

    def test_aggregate_last(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            results = s.aggregate_last(idx, vals)
            self.assertListEqual([0, 2, 5, 9], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.aggregate_last(idx, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 2, 5, 9], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.aggregate_last(idx, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 2, 5, 9], s.get(ds['result2']).data[:].tolist())

    def test_aggregate_min(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            results = s.aggregate_min(idx, vals)
            self.assertListEqual([0, 2, 4, 1], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.aggregate_min(idx, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 2, 4, 1], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.aggregate_min(idx, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 2, 4, 1], s.get(ds['result2']).data[:].tolist())

    def test_aggregate_max(self):
        idx = np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        vals = np.asarray([0, 8, 2, 6, 4, 5, 3, 7, 1, 9], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            results = s.aggregate_max(idx, vals)
            self.assertListEqual([0, 8, 6, 9], results.tolist())

            dst = s.open_dataset(bio, "w", "ds")
            ds = dst.create_dataframe('ds')
            s.aggregate_max(idx, vals, dest=s.create_numeric(ds, 'result', 'int64'))
            self.assertListEqual([0, 8, 6, 9], s.get(ds['result']).data[:].tolist())

            s.create_numeric(ds, 'vals', 'int64').data.write(vals)
            s.aggregate_max(idx, s.get(ds['vals']), dest=s.create_numeric(ds, 'result2', 'int64'))
            self.assertListEqual([0, 8, 6, 9], s.get(ds['result2']).data[:].tolist())


class TestSessionFields(unittest.TestCase):

    def test_write_then_read_numeric(self):
        from exetera.core.session import Session
        from exetera.core import fields
        from exetera.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=1000000, size=100000000)
            fields.numeric_field_constructor(s, hf, 'a', 'int32')
            a = fields.NumericField(s, hf['a'], write_enabled=True)
            a.data.write(values)

            total = np.sum(a.data[:])
            self.assertEqual(49997540637149, total)

            a.data[:] = a.data[:] * 2
            total = np.sum(a.data[:])
            self.assertEqual(99995081274298, total)

    def test_write_then_read_categorical(self):
        from exetera.core.session import Session
        from exetera.core import fields
        from exetera.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=3, size=100000000)
            fields.categorical_field_constructor(s, hf, 'a', 'int8',
                                                 {'foo': 0, 'bar': 1, 'boo': 2})
            a = fields.CategoricalField(s, hf['a'], write_enabled=True)
            a.data.write(values)

            total = np.sum(a.data[:])
            self.assertEqual(99987985, total)

    def test_write_then_read_fixed_string(self):
        from exetera.core.session import Session
        from exetera.core import fields
        from exetera.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=4, size=1000000)
            svalues = [b''.join([b'x'] * v) for v in values]
            fields.fixed_string_field_constructor(s, hf, 'a', 8)
            a = fields.FixedStringField(s, hf['a'], write_enabled=True)
            a.data.write(svalues)

            total = np.unique(a.data[:])
            self.assertListEqual([b'', b'x', b'xx', b'xxx'], total.tolist())

            a.data[:] = np.core.defchararray.add(a.data[:], b'y')
            self.assertListEqual(
                [b'xxxy', b'xxy', b'xxxy', b'y', b'xy', b'y', b'xxxy', b'xxxy', b'xy', b'y'],
                a.data[:10].tolist())


    def test_write_then_read_indexed_string(self):
        from exetera.core.session import Session
        from exetera.core import fields
        from exetera.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=4, size=200000)
            svalues = [''.join(['x'] * v) for v in values]
            fields.indexed_string_field_constructor(s, hf, 'a', 8)
            a = fields.IndexedStringField(s, hf['a'], write_enabled=True)
            a.data.write(svalues)

            total = np.unique(a.data[:])
            self.assertListEqual(['', 'x', 'xx', 'xxx'], total.tolist())

            strs = a.data[:]
            strs = [s+'y' for s in strs]
            a.data.clear()
            a.data.write(strs)

            # print(strs[:10])
            self.assertListEqual(
                ['xxxy', 'xxy', 'xxxy', 'y', 'xy', 'y', 'xxxy', 'xxxy', 'xy', 'y'], strs[:10])
            # print(a.indices[:10])
            self.assertListEqual([0, 4, 7, 11, 12, 14, 15, 19, 23, 25],
                                 a.indices[:10].tolist())
            # print(a.values[:10])
            self.assertListEqual(
                [120, 120, 120, 121, 120, 120, 121, 120, 120, 120], a.values[:10].tolist())
            # print(a.data[:10])
            self.assertListEqual(
                ['xxxy', 'xxy', 'xxxy', 'y', 'xy', 'y', 'xxxy', 'xxxy', 'xy', 'y'], a.data[:10])


class TestSessionImporters(unittest.TestCase):

    def test_indexed_string_importer(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            hf = dst.create_dataframe('hf')
            values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
            im = fields.IndexedStringImporter(s, hf, 'x')
            im.write(values)
            f = s.get(hf['x'])

            expected = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
             '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
            self.assertListEqual(expected, f.data[:])

            expected = [0, 0, 0, 5, 5, 11, 16, 21, 26, 26, 26, 31, 36, 36, 41, 47, 52, 52]
            self.assertListEqual(expected, f.indices[:].tolist())

            expected = [49, 46, 48, 46, 48, 49, 46, 48, 46, 195,
                        164, 49, 46, 48, 46, 48, 49, 46, 48, 46,
                        48, 49, 46, 48, 46, 48, 49, 46, 48, 46,
                        48, 49, 46, 48, 46, 48, 49, 46, 48, 46,
                        48, 49, 46, 48, 46, 195, 164, 49, 46, 48,
                        46, 48]
            self.assertListEqual(expected, f.values[:].tolist())

    def test_fixed_string_importer(self):
        bio = BytesIO()
        with session.Session() as s:
            dst=s.open_dataset(bio,'r+','dst')
            hf=dst.create_dataframe('hf')
            values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
            bvalues = [v.encode() for v in values]
            im = fields.FixedStringImporter(s, hf, 'x', max(len(b) for b in bvalues))
            im.write(bvalues)
            f = s.get(hf['x'])
            expected = [b'', b'', b'1.0.0', b'', b'1.0.\xc3\xa4', b'1.0.0', b'1.0.0',
                        b'1.0.0', b'', b'', b'1.0.0', b'1.0.0', b'', b'1.0.0',
                        b'1.0.\xc3\xa4', b'1.0.0', b'']

            self.assertListEqual(expected, f.data[:].tolist())

    def test_numeric_importer(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            hf = dst.create_dataframe('hf')
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            im = fields.NumericImporter(s, hf, 'x', 'float32', per.try_str_to_float)
            im.write(values)
            f = s.get(hf['x'])
            expected = [0.0, 0.0, 2.0, 3.0, 40.0, 5.21e-2, 0.0, -6.0, -7.0, -80.0, -9.21e-2,
                        0.0, 0.0, 2.0, 3.0, 40.0, 5.21e-2, 0.0, -6.0, -7.0, -80.0, -9.21e-2]
            actual = f.data[:].tolist()
            self.assertEqual(len(expected), len(actual))
            for i, j in zip(expected, actual):
                self.assertAlmostEqual(i, j)

    def test_date_importer(self):
        from datetime import datetime
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio,'r+','dst')
            hf = dst.create_dataframe('hf')
            values = ['2020-05-10', '2020-05-12', '2020-05-12', '2020-05-15']
            im = fields.DateImporter(s, hf, 'x')
            im.write(values)
            f = s.get(hf['x'])
            self.assertListEqual(
                [datetime(year=int(v[0:4]), month=int(v[5:7]), day=int(v[8:10])
                          ).timestamp() for v in values],
                f.data[:].tolist())