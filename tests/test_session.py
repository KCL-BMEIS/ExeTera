import unittest

import numpy as np
from io import BytesIO

import h5py

from hystore.core import session
from hystore.core import fields
from hystore.core import persistence as per


class TestSessionMerge(unittest.TestCase):

    def test_merge_left(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 300, 400, 500, 600])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
        r_vals = np.asarray([1000, 2000, 2001, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001])

        s = session.Session()
        print(s.merge_left(l_id, r_id, right_fields=(r_vals,)))

        r_vals2 = np.asarray(['a0', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1', 'f0', 'f1', 'h0', 'h1'])

        s = session.Session()
        print(s.merge_left(l_id, r_id, right_fields=(r_vals2,)))


    def test_merge_left_2(self):
        s = session.Session()
        p_id = np.array([100, 200, 300, 400, 500, 600, 800, 900])
        p_val = np.array([-1, -2, -3, -4, -5, -6, -8, -9])
        a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                          600, 600, 700, 700, 900, 900, 900])
        a_val = np.array([10, 11, 12, 23, 22, 40, 43, 42, 41, 60,
                          61, 63, 71, 71, 94, 93, 92])

        print(s.merge_left(p_id, a_pid, right_fields=(a_val,)))
        print(s.merge_left(a_pid, p_id, right_fields=(p_val,)))


    def test_merge_left_3(self):
        s = session.Session()
        p_id = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        d_pid = np.array([10, 30, 40, 60, 80])
        d_counts = np.array([2, 1, 2, 3, 2])
        d_to_p = np.zeros(len(p_id))
        import pandas as pd
        pdf = pd.DataFrame({'id': p_id})
        ddf = pd.DataFrame({'patient_id': d_pid, 'd_counts': d_counts})
        print(pd.merge(left=pdf, right=ddf, left_on='id', right_on='patient_id', how='left'))
        print(s.merge_left(left_on=p_id, right_on=d_pid, right_fields=(d_counts,)))
        print(s.ordered_left_merge(left_on=p_id, right_on=d_pid, left_to_right_map=d_to_p,
                                   left_field_sources=(d_counts,),
                                   left_unique=True, right_unique=True))


    def test_ordered_merge_left_2(self):
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            s = session.Session()
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
            s.ordered_left_merge(f_a_pid, f_p_id, a_to_p,
                                 left_field_sources=(f_p_val,),
                                 left_field_sinks=(f_a_p_val,),
                                 right_unique=True)

            print(f_a_p_val.data[:])

            # print(s.merge_left(p_id, a_pid, left_fields=(p_val,)))
            print(s.merge_left(a_pid, p_id, right_fields=(p_val,)))


    def test_ordered_merge_right_2(self):
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            s = session.Session()
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
            s.ordered_right_merge(f_p_id, f_a_pid, a_to_p,
                                 right_field_sources=(f_p_val,),
                                 right_field_sinks=(f_a_p_val,),
                                 left_unique=True)

            print(f_a_p_val.data[:])

            # print(s.merge_left(p_id, a_pid, left_fields=(p_val,)))
            print(s.merge_left(a_pid, p_id, right_fields=(p_val,)))


    def test_merge_right(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 300, 400, 500, 600])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
        r_vals = np.asarray([1000, 2000, 2001, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001])

        s = session.Session()
        res = s.merge_right(l_id, r_id, left_fields=(l_vals,))
        print(res)

    def test_ordered_merge_left(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 400, 600, 700, 800])
        l_vals_2 = np.asarray([10000, 20000, 40000, 60000, 70000, 80000])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
        r_vals = np.asarray([1000, 3000, 3001, 4000, 4001, 5000, 5001, 6000, 6001, 8000, 8001])
        r_vals_2 = np.asarray([100000, 300001, 300000, 400001, 400000,
                               500001, 50000, 600001, 600000, 800001, 800000])

        l_vals_exp = np.asarray([100, 0, 0, 400, 400, 0, 0, 600, 600, 800, 800], dtype=np.int32)
        l_vals_2_exp = np.asarray([10000, 0, 0, 40000, 40000, 0, 0, 60000, 60000, 80000, 80000],
                                  dtype=np.int32)

        s = session.Session()
        l_to_r = np.zeros(len(r_id), dtype=np.int64)
        actual = s.ordered_left_merge(r_id, l_id, left_to_right_map=l_to_r,
                                      left_field_sources=(l_vals,),
                                      right_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))

        r_to_l = np.zeros(len(l_id), dtype=np.int64)
        actual = s.ordered_right_merge(l_id, r_id, right_to_left_map=r_to_l,
                                       right_field_sources=(l_vals,),
                                       left_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))

        l_to_r = np.zeros(len(r_id), dtype=np.int64)
        actual = s.ordered_left_merge(r_id, l_id, left_to_right_map=l_to_r,
                                      left_field_sources=(l_vals, l_vals_2),
                                      right_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))
        self.assertTrue(np.array_equal(actual[1], l_vals_2_exp))

        r_to_l = np.zeros(len(l_id), dtype=np.int64)
        actual = s.ordered_right_merge(l_id, r_id, right_to_left_map=r_to_l,
                                       right_field_sources=(l_vals, l_vals_2),
                                       left_unique=True)
        self.assertTrue(np.array_equal(actual[0], l_vals_exp))
        self.assertTrue(np.array_equal(actual[1], l_vals_2_exp))


    def test_ordered_merge_inner(self):
        l_id = np.asarray(['a', 'b', 'd', 'f', 'g', 'h'])
        l_vals = np.asarray([100, 200, 400, 600, 700, 800])
        l_vals_2 = np.asarray([10000, 20000, 40000, 60000, 70000, 80000])

        r_id = np.asarray(['a', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f', 'h', 'h'])
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
        actual = s.ordered_inner_merge(l_id, r_id,
                                       left_field_sources=(l_vals, l_vals_2),
                                       right_field_sources=(r_vals, r_vals_2),
                                       left_unique=True, right_unique=False)
        self.assertTrue(np.array_equal(actual[0][0], l_vals_exp))
        self.assertTrue(np.array_equal(actual[0][1], l_vals_2_exp))
        self.assertTrue(np.array_equal(actual[1][0], r_vals_exp))
        self.assertTrue(np.array_equal(actual[1][1], r_vals_2_exp))


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
            s.create_fixed_string(hf, 'x', 1).data.write(vx)
            s.create_numeric(hf, 'a', 'int32').data.write(va)
            s.create_numeric(hf, 'b', 'int32').data.write(vb)

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
            s.create_fixed_string(hf, 'x', 1).data.write(vx)
            s.create_numeric(hf, 'a', 'int32').data.write(va)
            s.create_numeric(hf, 'b', 'int32').data.write(vb)

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


class TestSessionAggregate(unittest.TestCase):

    pass


class TestSessionFields(unittest.TestCase):

    def test_write_then_read_numeric(self):
        from hystore.core.session import Session
        from hystore.core import fields
        from hystore.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=1000000, size=100000000)
            fields.numeric_field_constructor(s, hf, 'a', 'int32')
            a = fields.NumericField(s, hf['a'], write_enabled=True)
            a.data.write(values)

            with Timer("array"):
                total = np.sum(a.data[:])
            print(total)

            with Timer("* 2"):
                a.data[:] = a.data[:] * 2
                total = np.sum(a.data[:])
            print(total)


    def test_write_then_read_categorical(self):
        from hystore.core.session import Session
        from hystore.core import fields
        from hystore.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=3, size=100000000)
            print(values.min(), values.max())
            fields.categorical_field_constructor(s, hf, 'a', 'int8',
                                                 {'foo': 0, 'bar': 1, 'boo': 2})
            a = fields.CategoricalField(s, hf['a'], write_enabled=True)
            a.data.write(values)

            with Timer("array"):
                total = np.sum(a.data[:])
            print(total)

            d = a.data[:]
            a.data[:] = np.where(d == 2, 1, d)


    def test_write_then_read_fixed_string(self):
        from hystore.core.session import Session
        from hystore.core import fields
        from hystore.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=4, size=1000000)
            svalues = [b''.join([b'x'] * v) for v in values]
            fields.fixed_string_field_constructor(s, hf, 'a', 8)
            a = fields.FixedStringField(s, hf['a'], write_enabled=True)
            a.data.write(svalues)

            with Timer("array"):
                total = np.unique(a.data[:])
            print(total)

            with Timer("* 2"):
                a.data[:] = np.core.defchararray.add(a.data[:], b'y')
            print(a.data[:10])


    def test_write_then_read_indexed_string(self):
        from hystore.core.session import Session
        from hystore.core import fields
        from hystore.core.utils import Timer

        s = Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=4, size=200000)
            svalues = [''.join(['x'] * v) for v in values]
            fields.indexed_string_field_constructor(s, hf, 'a', 8)
            a = fields.IndexedStringField(s, hf['a'], write_enabled=True)
            a.data.write(svalues)

            with Timer("array"):
                total = np.unique(a.data[:])
            print(total)

            with Timer("* 2"):
                strs = a.data[:]
                strs = [s+'y' for s in strs]
                a.data.clear()
                a.data.write(strs)
            print(strs[:10])
            print(a.indices[:10])
            print(a.values[:10])
            print(a.data[:10])


class TestSessionImporters(unittest.TestCase):

    def test_indexed_string_importer(self):

        s = session.Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
            im = fields.IndexedStringImporter(s, hf, 'x')
            im.write(values)
            f = s.get(hf['x'])
            print(f, f.data)
            print(f.data[:])
            print(f.indices[:])
            print(f.values[:])

    def test_fixed_string_importer(self):
        s = session.Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
            im = fields.FixedStringImporter(s, hf, 'x',
                                            max(len(v.encode()) for v in values))
            im.write(values)
            f = s.get(hf['x'])
            print(f, f.data)
            print(f.data[:])

    def test_numeric_importer(self):
        from datetime import datetime
        s = session.Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            im = fields.NumericImporter(s, hf, 'x', 'float32', per.try_str_to_float)
            im.write(values)
            f = s.get(hf['x'])
            print(f, f.data)
            print(f.data[:])

    def test_date_importer(self):
        from datetime import datetime
        s = session.Session()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['2020-05-10', '2020-05-12', '2020-05-12', '2020-05-15']
            im = fields.DateImporter(s, hf, 'x')
            im.write(values)
            f = s.get(hf['x'])
            print(f, f.data)
            print([datetime.fromtimestamp(d) for d in f.data[:]])
