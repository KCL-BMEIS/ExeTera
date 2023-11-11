import pandas as pd
from exetera.core.operations import INVALID_INDEX
import unittest
from parameterized import parameterized
from io import BytesIO
import numpy as np
import tempfile
import os

from exetera.core import session
from exetera.core import dataframe
from .utils import SessionTestCase, DEFAULT_FIELD_DATA

class TestDataFrameCreateFields(unittest.TestCase):

    def test_dataframe_init(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            # init
            df = dst.create_dataframe('dst')
            self.assertEqual(len(df), 0)
            self.assertTrue(isinstance(df, dataframe.DataFrame))
            numf = df.create_numeric('numf', 'uint32')
            df2 = dst.create_dataframe('dst2', dataframe=df)
            self.assertTrue(isinstance(df2, dataframe.DataFrame))
            self.assertEqual(len(df2), 1)
            self.assertListEqual([numf], list(df.values()))

            # add & set & contains
            self.assertTrue('numf' in df)
            self.assertTrue('numf' in df2)
            cat = s.create_categorical(df2, 'cat', 'int8', {'a': 1, 'b': 2})
            self.assertFalse('cat' in df)
            self.assertFalse(df.contains_field(cat))
            df['cat'] = cat
            self.assertTrue('cat' in df)
            self.assertEqual(len(df2), 2)
            self.assertEqual(len(df), 2)

            with self.assertRaises(TypeError):
                df[1] = cat
            with self.assertRaises(TypeError):
                df['cat2'] = 'foo'

            num2 = s.create_numeric(df2, 'num2', 'int32')
            df.add(num2)  # add is hard-copy
            self.assertTrue('num2' in df)

            with self.assertRaises(TypeError):
                1 in df
            with self.assertRaises(TypeError):
                df.contains_field(1)
            self.assertTrue(df.contains_field(df['num2']))

            # list & get
            self.assertEqual(id(numf), id(df.get_field('numf')))
            self.assertEqual(id(numf), id(df['numf']))

            with self.assertRaises(TypeError):
                df[1]
            with self.assertRaises(ValueError):
                df['foo']

            # list & iter
            dfit = iter(df)
            self.assertEqual('numf', next(dfit))
            self.assertEqual('cat', next(dfit))

            # del & del by field
            del df['numf']
            self.assertFalse('numf' in df)
            with self.assertRaises(ValueError, msg="This field is owned by a different dataframe"):
                df.delete_field(cat)
            with self.assertRaises(ValueError):
                del df['numf']
            self.assertFalse(df.contains_field(cat))



    def test_dataframe_create_numeric(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            df = dst.create_dataframe('dst')
            num = df.create_numeric('num', 'uint32')
            num.data.write([1, 2, 3, 4])
            self.assertEqual([1, 2, 3, 4], num.data[:].tolist())
            num2 = df.create_numeric('num2', 'uint32')
            num2.data.write([1, 2, 3, 4])

    def test_dataframe_create_numeric(self):
        bio = BytesIO()
        with session.Session() as s:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=1000000, size=100000000)
            dst = s.open_dataset(bio, 'r+', 'dst')
            df = dst.create_dataframe('dst')
            a = df.create_numeric('a','int32')
            a.data.write(values)

            total = np.sum(a.data[:], dtype=np.int64)
            self.assertEqual(49997540637149, total)

            a.data[:] = a.data[:] * 2
            total = np.sum(a.data[:], dtype=np.int64)
            self.assertEqual(99995081274298, total)

    def test_dataframe_create_categorical(self):
        bio = BytesIO()
        with session.Session() as s:
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=3, size=100000000)
            dst = s.open_dataset(bio, 'r+', 'dst')
            hf = dst.create_dataframe('dst')
            a = hf.create_categorical('a', 'int8',
                                                 {'foo': 0, 'bar': 1, 'boo': 2})
            a.data.write(values)

            total = np.sum(a.data[:])
            self.assertEqual(99987985, total)

    def test_dataframe_create_fixed_string(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            hf = dst.create_dataframe('dst')
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=4, size=1000000)
            svalues = [b''.join([b'x'] * v) for v in values]
            a = hf.create_fixed_string('a', 8)
            a.data.write(svalues)

            total = np.unique(a.data[:])
            self.assertListEqual([b'', b'x', b'xx', b'xxx'], total.tolist())

            a.data[:] = np.core.defchararray.add(a.data[:], b'y')
            self.assertListEqual(
                [b'xxxy', b'xxy', b'xxxy', b'y', b'xy', b'y', b'xxxy', b'xxxy', b'xy', b'y'],
                a.data[:10].tolist())


    def test_dataframe_create_indexed_string(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            hf = dst.create_dataframe('dst')
            np.random.seed(12345678)
            values = np.random.randint(low=0, high=4, size=200000)
            svalues = [''.join(['x'] * v) for v in values]
            a = hf.create_indexed_string('a', 8)
            a.data.write(svalues)

            total = np.unique(a.data[:])
            self.assertListEqual(['', 'x', 'xx', 'xxx'], total.tolist())

            strs = a.data[:]
            strs = [s + 'y' for s in strs]
            a.data.clear()
            a.data.write(strs)

            self.assertListEqual(
                ['xxxy', 'xxy', 'xxxy', 'y', 'xy', 'y', 'xxxy', 'xxxy', 'xy', 'y'], strs[:10])
            self.assertListEqual([0, 4, 7, 11, 12, 14, 15, 19, 23, 25],
                                 a.indices[:10].tolist())
            self.assertListEqual(
                [120, 120, 120, 121, 120, 120, 121, 120, 120, 120], a.values[:10].tolist())
            self.assertListEqual(
                ['xxxy', 'xxy', 'xxxy', 'y', 'xy', 'y', 'xxxy', 'xxxy', 'xy', 'y'], a.data[:10])


    def test_dataframe_create_mem_numeric(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            df = dst.create_dataframe('dst')
            num = df.create_numeric('num', 'uint32')
            num.data.write([1, 2, 3, 4])
            self.assertEqual([1, 2, 3, 4], num.data[:].tolist())
            num2 = df.create_numeric('num2', 'uint32')
            num2.data.write([1, 2, 3, 4])

            df['num3'] = num + num2
            self.assertEqual([2, 4, 6, 8], df['num3'].data[:].tolist())
            df['num4'] = num - np.array([1, 2, 3, 4])
            self.assertEqual([0, 0, 0, 0], df['num4'].data[:].tolist())
            df['num5'] = num * np.array([1, 2, 3, 4])
            self.assertEqual([1, 4, 9, 16], df['num5'].data[:].tolist())
            df['num6'] = df['num5'] / np.array([1, 2, 3, 4])
            self.assertEqual([1, 2, 3, 4], df['num6'].data[:].tolist())
            df['num7'] = df['num'] & df['num2']
            self.assertEqual([1, 2, 3, 4], df['num7'].data[:].tolist())
            df['num8'] = df['num'] | df['num2']
            self.assertEqual([1, 2, 3, 4], df['num8'].data[:].tolist())
            df['num9'] = df['num'] ^ df['num2']
            self.assertEqual([0, 0, 0, 0], df['num9'].data[:].tolist())
            df['num10'] = df['num'] % df['num2']
            self.assertEqual([0, 0, 0, 0], df['num10'].data[:].tolist())


    def test_dataframe_create_mem_categorical(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            df = dst.create_dataframe('dst')
            cat1 = df.create_categorical('cat1','uint8',{'foo': 0, 'bar': 1, 'boo': 2})
            cat1.data.write([0, 1, 2, 0, 1, 2])

            cat2 = df.create_categorical('cat2','uint8',{'foo': 0, 'bar': 1, 'boo': 2})
            cat2.data.write([1, 2, 0, 1, 2, 0])

            df['r1'] = cat1 < cat2
            self.assertEqual([True, True, False, True, True, False], df['r1'].data[:].tolist())
            df['r2'] = cat1 <= cat2
            self.assertEqual([True, True, False, True, True, False], df['r2'].data[:].tolist())
            df['r3'] = cat1 == cat2
            self.assertEqual([False, False, False, False, False, False], df['r3'].data[:].tolist())
            df['r4'] = cat1 != cat2
            self.assertEqual([True, True, True, True, True, True], df['r4'].data[:].tolist())
            df['r5'] = cat1 > cat2
            self.assertEqual([False, False, True, False, False, True], df['r5'].data[:].tolist())
            df['r6'] = cat1 >= cat2
            self.assertEqual([False, False, True, False, False, True], df['r6'].data[:].tolist())

    def test_dataframe_static_methods(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('dst')
            numf = s.create_numeric(df, 'numf', 'int32')
            numf.data.write([5, 4, 3, 2, 1])
            idxs = s.create_indexed_string(df, 'idxs')
            idxs.data.write(['aaa', 'b', 'ccc', 'dddd'])

            df2 = dst.create_dataframe('df2')
            dataframe.copy(numf, df2,'numf')
            dataframe.copy(idxs, df2, 'idxs')
            self.assertListEqual([5, 4, 3, 2, 1], df2['numf'].data[:].tolist())
            df.drop('numf')
            self.assertTrue('numf' not in df)
            dataframe.move(df2['numf'], df, 'numf')
            self.assertTrue('numf' not in df2)
            self.assertListEqual([5, 4, 3, 2, 1], df['numf'].data[:].tolist())



    def test_dataframe_ops(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('dst')
            numf = s.create_numeric(df, 'numf', 'int32')
            numf.data.write([5, 4, 3, 2, 1])

            fst = s.create_fixed_string(df, 'fst', 3)
            fst.data.write([b'e', b'd', b'c', b'b', b'a'])

            index = np.array([4, 3, 2, 1, 0])
            ddf = dst.create_dataframe('dst2')
            df.apply_index(index, ddf)
            self.assertEqual([1, 2, 3, 4, 5], ddf['numf'].data[:].tolist())
            self.assertEqual([b'a', b'b', b'c', b'd', b'e'], ddf['fst'].data[:].tolist())
            with self.assertRaises(TypeError):
                df.apply_index(index, 'foo')

            filter_to_apply = np.array([True, True, False, False, True])
            ddf = dst.create_dataframe('dst3')
            df.apply_filter(filter_to_apply, ddf)
            self.assertEqual([5, 4, 1], ddf['numf'].data[:].tolist())
            self.assertEqual([b'e', b'd', b'a'], ddf['fst'].data[:].tolist())





class TestDataFrameRename(unittest.TestCase):

    def test_rename_1(self):

        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        b = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype='int32')

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            fa = df.create_numeric('fa', 'int32').data.write(a)
            fb = df.create_numeric('fb', 'int32').data.write(b)
            df.rename('fa', 'fc')
            self.assertFalse('fa' in df)
            self.assertTrue('fb' in df)
            self.assertTrue('fc' in df)
            with self.assertRaises(ValueError):
                df.rename(123,456)
            with self.assertRaises(ValueError):
                df.rename({'fc': 'fb'}, 'fb')
                df.rename('fc', None)

    def test_rename_should_not_clash(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        b = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype='int32')
        c = np.array([8, 1, 7, 2, 6, 3, 5, 4], dtype='int32')

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            fa = df.create_numeric('fa', 'int32').data.write(a)
            fb = df.create_numeric('fb', 'int32').data.write(b)
            fc = df.create_numeric('fc', 'int32').data.write(c)
            fa = df['fa']
            df.rename({'fa': 'fb', 'fb': 'fc', 'fc': 'fa'})
            self.assertListEqual(['fb', 'fc', 'fa'], list(df.keys()))
            self.assertEqual('fb', fa.name)
            self.assertTrue('fa' in df)
            self.assertTrue('fb' in df)
            self.assertTrue('fc' in df)
            self.assertEqual('fa', df['fa'].name)
            self.assertEqual('fb', df['fb'].name)
            self.assertEqual('fc', df['fc'].name)

    def test_rename_should_clash(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        b = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype='int32')
        c = np.array([8, 1, 7, 2, 6, 3, 5, 4], dtype='int32')

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            fa = df.create_numeric('fa', 'int32').data.write(a)
            fb = df.create_numeric('fb', 'int32').data.write(b)
            fc = df.create_numeric('fc', 'int32').data.write(c)
            fa = df['fa']
            with self.assertRaises(ValueError):
                df.rename({'fa': 'fc'})
            self.assertListEqual(['fa', 'fb', 'fc'], list(df.keys()))
            self.assertTrue('fa' in df)
            self.assertTrue('fb' in df)
            self.assertTrue('fc' in df)
            self.assertEqual('fa', df['fa'].name)
            self.assertEqual('fb', df['fb'].name)
            self.assertEqual('fc', df['fc'].name)

class TestDataFrameCopyMove(unittest.TestCase):

    def test_move_same_dataframe(self):

        sa = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        sb = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype='int32')

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df1 = ds.create_dataframe('df1')
            df1.create_numeric('fa', 'int32').data.write(sa)
            df1.create_numeric('fb', 'int32').data.write(sb)
            fa = df1['fa']
            fc = dataframe.move(df1['fa'], df1, 'fc')
            self.assertEqual('fc', fc.name)
            self.assertEqual('fb', df1['fb'].name)

    def test_move_different_dataframe(self):

        sa = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        sb = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype='int32')

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df1 = ds.create_dataframe('df1')
            df1.create_numeric('fa', 'int32').data.write(sa)
            df1.create_numeric('fb', 'int32').data.write(sb)
            df2 = ds.create_dataframe('df2')
            df2.create_numeric('fb', 'int32').data.write(sb)
            fa = df1['fa']
            fc = dataframe.move(df1['fa'], df2, 'fc')
            with self.assertRaises(ValueError, msg="This field no longer refers to a valid "
                                                   "underlying field object"):
                _ = fa.name
            self.assertEqual('fc', fc.name)
            self.assertEqual('fb', df1['fb'].name)


class TestDataFrameApplyFilter(unittest.TestCase):

    def test_apply_filter(self):

        src = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        filt = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype='bool')
        expected = src[filt].tolist()

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            numf = s.create_numeric(df, 'numf', 'int32')
            numf.data.write(src)
            df2 = dst.create_dataframe('df2')
            df2b = df.apply_filter(filt, df2)
            self.assertListEqual(expected, df2['numf'].data[:].tolist())
            self.assertListEqual(expected, df2b['numf'].data[:].tolist())
            self.assertListEqual(src.tolist(), df['numf'].data[:].tolist())

            df.apply_filter(filt)
            self.assertListEqual(expected, df['numf'].data[:].tolist())

            with self.assertRaises(TypeError):
                df.apply_filter(filt, 123)


    def test_apply_filter_with_numeric_filter(self):

        src = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        filt = np.array([0, 2, 0, 1, 0, 1, 1, 0])
        expected = src[filt!=0].tolist()

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            numf = s.create_numeric(df, 'numf', 'int32')
            numf.data.write(src)
            df2 = dst.create_dataframe('df2')
            df2b = df.apply_filter(filt, df2)
            self.assertListEqual(expected, df2['numf'].data[:].tolist())
            self.assertListEqual(expected, df2b['numf'].data[:].tolist())
            self.assertListEqual(src.tolist(), df['numf'].data[:].tolist())

            df.apply_filter(filt)
            self.assertListEqual(expected, df['numf'].data[:].tolist())

    
    def test_apply_filter_with_equation_filter(self):

        src = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
        expected = src[src > 2].tolist()

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            numf = s.create_numeric(df, 'numf', 'int32')
            numf.data.write(src)
            df2 = dst.create_dataframe('df2')
            df2b = df.apply_filter(df['numf'] > 2, df2)
            self.assertListEqual(expected, df2['numf'].data[:].tolist())
            self.assertListEqual(expected, df2b['numf'].data[:].tolist())
            self.assertListEqual(src.tolist(), df['numf'].data[:].tolist())


class TestDataFrameMerge(unittest.TestCase):

    def tests_merge_left(self):

        l_id = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype='int32')
        r_id = np.asarray([2, 3, 0, 4, 7, 6, 2, 0, 3], dtype='int32')
        r_vals = ['bb1', 'ccc1', '', 'dddd1', 'ggggggg1', 'ffffff1', 'bb2', '', 'ccc2']
        expected = ['', '', '', 'bb1', 'bb2', 'ccc1', 'ccc2', 'dddd1', '', 'ffffff1', 'ggggggg1']

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            ldf = dst.create_dataframe('ldf')
            rdf = dst.create_dataframe('rdf')
            ldf.create_numeric('l_id', 'int32').data.write(l_id)
            rdf.create_numeric('r_id', 'int32').data.write(r_id)
            rdf.create_indexed_string('r_vals').data.write(r_vals)
            ddf = dst.create_dataframe('ddf')
            dataframe.merge(ldf, rdf, ddf, 'l_id', 'r_id', how='left')
            self.assertEqual(expected, ddf['r_vals'].data[:])
            valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:]) | \
                             np.logical_not(ddf['valid_r'].data[:])
            self.assertTrue(np.all(valid_if_equal))

            with self.assertRaises(ValueError):
                dataframe.merge(123, rdf, ddf, 'l_id', 'r_id')
                dataframe.merge(ldf, 123, ddf, 'l_id', 'r_id')
                dataframe.merge(ldf, rdf, ddf, 'l_id', 'r_id', how='foo')



    def tests_merge_sorted(self):
        l_id = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype='int32')
        r_vals = ['bb1', 'ccc1', '', 'dddd1', 'ggggggg1', 'ffffff1', 'bb2', '', 'ccc2']

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            ldf = dst.create_dataframe('ldf')
            rdf = dst.create_dataframe('rdf')
            ldf.create_numeric('l_id', 'int32').data.write(l_id)

            r_sorted = dst.create_dataframe('r_sorted')
            r_sorted.create_numeric('r_id', 'int32').data.write(l_id)
            r_sorted.create_indexed_string('r_vals').data.write(r_vals[0:8])
            ddf = dst.create_dataframe('ddf2')
            dataframe.merge(ldf, r_sorted, ddf, 'l_id', 'r_id', how='left', hint_left_keys_ordered=True,
                            hint_right_keys_ordered=True)
            expected = ['bb1', 'ccc1', '', 'dddd1', 'ggggggg1', 'ffffff1', 'bb2', '']
            self.assertEqual(expected, ddf['r_vals'].data[:])
            valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:])
            self.assertTrue(np.all(valid_if_equal))
            ddf = dst.create_dataframe('ddf3')
            # dataframe.merge(ldf, r_sorted, ddf, 'l_id', 'r_id', how='left', hint_left_keys_ordered=True,
            #                 hint_right_keys_ordered=True, hint_left_keys_unique=True, hint_right_keys_unique=True)
            # self.assertEqual(expected, ddf['r_vals'].data[:])
            # valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:])
            # self.assertTrue(np.all(valid_if_equal))


            ddf = dst.create_dataframe('ddf4')
            dataframe.merge(ldf, r_sorted, ddf, 'l_id', 'r_id', how='right', hint_left_keys_ordered=True,
                            hint_right_keys_ordered=True)
            self.assertEqual(expected, ddf['r_vals'].data[:])
            valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:])
            self.assertTrue(np.all(valid_if_equal))

            ddf = dst.create_dataframe('ddf5')
            # dataframe.merge(ldf, r_sorted, ddf, 'l_id', 'r_id', how='right', hint_left_keys_ordered=True,
            #                 hint_right_keys_ordered=True, hint_left_keys_unique=True, hint_right_keys_unique=True)
            # self.assertEqual(expected, ddf['r_vals'].data[:])
            # valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:])
            # self.assertTrue(np.all(valid_if_equal))

            ddf = dst.create_dataframe('ddf6')
            dataframe.merge(ldf, r_sorted, ddf, 'l_id', 'r_id', how='inner', hint_left_keys_ordered=True,
                            hint_right_keys_ordered=True)
            self.assertEqual(expected, ddf['r_vals'].data[:])
            valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:])
            self.assertTrue(np.all(valid_if_equal))

            ddf = dst.create_dataframe('ddf7')
            # dataframe.merge(ldf, r_sorted, ddf, 'l_id', 'r_id', how='inner', hint_left_keys_ordered=True,
            #                 hint_right_keys_ordered=True, hint_left_keys_unique=True, hint_right_keys_unique=True)
            # self.assertEqual(expected, ddf['r_vals'].data[:])
            # valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:])
            # self.assertTrue(np.all(valid_if_equal))

    def tests_merge_right(self):

        r_id = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype='int32')
        l_id = np.asarray([2, 3, 0, 4, 7, 6, 2, 0, 3], dtype='int32')
        l_vals = ['bb1', 'ccc1', '', 'dddd1', 'ggggggg1', 'ffffff1', 'bb2', '', 'ccc2']
        expected = ['', '', '', 'bb1', 'bb2', 'ccc1', 'ccc2', 'dddd1', '', 'ffffff1', 'ggggggg1']

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            ldf = dst.create_dataframe('ldf')
            rdf = dst.create_dataframe('rdf')
            ldf.create_numeric('l_id', 'int32').data.write(l_id)
            ldf.create_indexed_string('l_vals').data.write(l_vals)
            rdf.create_numeric('r_id', 'int32').data.write(r_id)
            ddf = dst.create_dataframe('ddf')
            dataframe.merge(ldf, rdf, ddf, 'l_id', 'r_id', how='right')
            self.assertEqual(expected, ddf['l_vals'].data[:])
            valid_if_equal = (ddf['l_id'].data[:] == ddf['r_id'].data[:]) | \
                             np.logical_not(ddf['valid_l'].data[:])
            self.assertTrue(np.all(valid_if_equal))

    def tests_merge_inner(self):

        r_id = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype='int32')
        l_id = np.asarray([2, 3, 0, 4, 7, 6, 2, 0, 3], dtype='int32')
        r_vals = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven']
        l_vals = ['bb1', 'ccc1', '', 'dddd1', 'ggggggg1', 'ffffff1', 'bb2', '', 'ccc2']
        expected_left = ['bb1', 'bb2', 'ccc1', 'ccc2', '', '', 'dddd1', 'ggggggg1', 'ffffff1']
        expected_right = ['two', 'two', 'three', 'three', 'zero', 'zero', 'four', 'seven', 'six']

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            ldf = dst.create_dataframe('ldf')
            rdf = dst.create_dataframe('rdf')
            ldf.create_numeric('l_id', 'int32').data.write(l_id)
            ldf.create_indexed_string('l_vals').data.write(l_vals)
            rdf.create_numeric('r_id', 'int32').data.write(r_id)
            rdf.create_indexed_string('r_vals').data.write(r_vals)
            ddf = dst.create_dataframe('ddf')
            dataframe.merge(ldf, rdf, ddf, 'l_id', 'r_id', how='inner')
            self.assertEqual(expected_left, ddf['l_vals'].data[:])
            self.assertEqual(expected_right, ddf['r_vals'].data[:])

    def tests_merge_outer(self):

        r_id = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype='int32')
        l_id = np.asarray([2, 3, 0, 4, 7, 6, 2, 0, 3], dtype='int32')
        r_vals = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven']
        l_vals = ['bb1', 'ccc1', '', 'dddd1', 'ggggggg1', 'ffffff1', 'bb2', '', 'ccc2']
        expected_left = ['bb1', 'bb2', 'ccc1', 'ccc2', '', '', 'dddd1', 'ggggggg1', 'ffffff1',
                         '', '']
        expected_right = ['two', 'two', 'three', 'three', 'zero', 'zero', 'four', 'seven', 'six',
                          'one', 'five']

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            ldf = dst.create_dataframe('ldf')
            rdf = dst.create_dataframe('rdf')
            ldf.create_numeric('l_id', 'int32').data.write(l_id)
            ldf.create_indexed_string('l_vals').data.write(l_vals)
            rdf.create_numeric('r_id', 'int32').data.write(r_id)
            rdf.create_indexed_string('r_vals').data.write(r_vals)
            ddf = dst.create_dataframe('ddf')
            dataframe.merge(ldf, rdf, ddf, 'l_id', 'r_id', how='outer')
            self.assertEqual(expected_left, ddf['l_vals'].data[:])
            self.assertEqual(expected_right, ddf['r_vals'].data[:])


    def tests_merge_left_compound_key(self):

        l_id_1 = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype='int32')
        l_id_2 = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype='int32')
        r_id_1 = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype='int32')
        r_id_2 = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype='int32')
        l_vals = ['00', '01', '02', '03', '10', '11', '12', '13']
        r_vals = ['00', '10', '01', '11', '02', '12', '03', '13']
        expected = ['00', '01', '02', '03', '10', '11', '12', '13']

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            ldf = dst.create_dataframe('ldf')
            rdf = dst.create_dataframe('rdf')
            ldf.create_numeric('l_id_1', 'int32').data.write(l_id_1)
            ldf.create_numeric('l_id_2', 'int32').data.write(l_id_2)
            ldf.create_indexed_string('l_vals').data.write(l_vals)
            rdf.create_numeric('r_id_1', 'int32').data.write(r_id_1)
            rdf.create_numeric('r_id_2', 'int32').data.write(r_id_2)
            rdf.create_indexed_string('r_vals').data.write(r_vals)
            ddf = dst.create_dataframe('ddf')
            dataframe.merge(ldf, rdf, ddf, ('l_id_1', 'l_id_2'), ('r_id_1', 'r_id_2'), how='left')
            self.assertEqual(expected, ddf['l_vals'].data[:])
            self.assertEqual(expected, ddf['r_vals'].data[:])
            self.assertEqual(ddf['l_id_1'].data[:].tolist(), ddf['r_id_1'].data[:].tolist())
            self.assertEqual(ddf['l_id_2'].data[:].tolist(), ddf['r_id_2'].data[:].tolist())



class TestDataFrameGroupBy(unittest.TestCase):

    def test_distinct_single_field(self):
        val = np.asarray([1, 0, 1, 2, 3, 2, 2, 3, 3, 3], dtype=np.int32)
        val2 = np.asarray(['a', 'b', 'a', 'b', 'c', 'b', 'c', 'c', 'd', 'd'], dtype = 'S1')
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.drop_duplicates(by = 'val', ddf = ddf)

            self.assertListEqual([0, 1, 2, 3], ddf['val'].data[:].tolist())        
        

    def test_distinct_multi_fields(self):
        val = np.asarray([1, 0, 1, 2, 3, 2, 2, 3, 3, 3], dtype=np.int32)
        val2 = np.asarray(['a', 'b', 'a', 'b', 'c', 'b', 'c', 'c', 'd', 'd'], dtype = 'S1')
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.drop_duplicates(by = ['val', 'val2'], ddf = ddf)

            self.assertListEqual([0, 1, 2, 2, 3, 3], ddf['val'].data[:].tolist())        
            self.assertListEqual([b'b', b'a', b'b', b'c', b'c', b'd'], ddf['val2'].data[:].tolist())        


    def test_groupby_count_single_field(self):
        val = np.asarray([1, 0, 1, 2, 3, 2, 2, 3, 3, 3], dtype=np.int32)
        val2 = np.asarray(['a', 'b', 'a', 'b', 'c', 'b', 'c', 'c', 'd', 'd'], dtype = 'S1')
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = 'val').count(ddf = ddf)

            self.assertListEqual([0, 1, 2, 3], ddf['val'].data[:].tolist())    
            self.assertListEqual([1, 2, 3, 4], ddf['count'].data[:].tolist())    
        

    def test_groupby_count_multi_fields(self):
        val = np.asarray([1, 0, 1, 2, 3, 2, 2, 3, 3, 3], dtype=np.int32)
        val2 = np.asarray(['a', 'b', 'a', 'b', 'c', 'b', 'c', 'c', 'd', 'd'], dtype = 'S1')
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = ['val', 'val2']).count(ddf = ddf)

            self.assertListEqual([0, 1, 2, 2, 3, 3], ddf['val'].data[:].tolist())        
            self.assertListEqual([b'b', b'a', b'b', b'c', b'c', b'd'], ddf['val2'].data[:].tolist())        
            self.assertListEqual([1, 2, 2, 1, 2, 2], ddf['count'].data[:].tolist())


    def test_groupby_max_single_field(self):
        val = np.asarray([3, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=np.int32)
        val2 = np.asarray([9, 8, 2, 6, 4, 5, 3, 7, 1, 0], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_numeric("val2", "int64").data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = 'val').max(target ='val2', ddf = ddf)
            
            self.assertListEqual([0, 1, 2, 3], ddf['val'].data[:].tolist())
            self.assertListEqual([0, 8, 6, 9], ddf['val2_max'].data[:].tolist())    


    def test_groupby_max_multi_fields(self):
        val = np.asarray([1, 2, 1, 2], dtype=np.int32)
        val2 = np.asarray(['a', 'c', 'a', 'b'], dtype = 'S1')
        val3 = np.asarray([3, 4, 5, 6])
        val4 = np.asarray(['aa', 'ab', 'cd', 'def'])
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)
            df.create_numeric("val3", "int32").data.write(val3)
            df.create_indexed_string("val4").data.write(val4)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = ['val', 'val2']).max(['val3', 'val4'], ddf = ddf)

            self.assertListEqual([1, 2, 2], ddf['val'].data[:].tolist())    
            self.assertListEqual([b'a', b'b', b'c'], ddf['val2'].data[:].tolist())    
            self.assertListEqual([5, 6, 4], ddf['val3_max'].data[:].tolist())    
            self.assertListEqual(['cd', 'def', 'ab'], ddf['val4_max'].data[:])    


    def test_groupby_min_single_field(self):
        val = np.asarray([3, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=np.int32)
        val2 = np.asarray([9, 8, 2, 6, 4, 5, 3, 7, 1, 0], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_numeric("val2", "int64").data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = 'val').min(target ='val2', ddf = ddf)
            
            self.assertListEqual([0, 1, 2, 3], ddf['val'].data[:].tolist())
            self.assertListEqual([0, 2, 4, 1], ddf['val2_min'].data[:].tolist())    


    def test_groupby_min_multi_fields(self):
        val = np.asarray([1, 2, 1, 2], dtype=np.int32)
        val2 = np.asarray(['a', 'c', 'a', 'b'], dtype = 'S1')
        val3 = np.asarray([3, 4, 5, 6])
        val4 = np.asarray(['aa', 'ab', 'cd', 'def'])
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)
            df.create_numeric("val3", "int32").data.write(val3)
            df.create_indexed_string("val4").data.write(val4)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = ['val', 'val2']).min(['val3', 'val4'], ddf = ddf)

            self.assertListEqual([1, 2, 2], ddf['val'].data[:].tolist())    
            self.assertListEqual([b'a', b'b', b'c'], ddf['val2'].data[:].tolist())    
            self.assertListEqual([3, 6, 4], ddf['val3_min'].data[:].tolist())    
            self.assertListEqual(['aa', 'def', 'ab'], ddf['val4_min'].data[:])    


    def test_groupby_first_single_field(self):
        val = np.asarray([3, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=np.int32)
        val2 = np.asarray([9, 8, 2, 6, 4, 5, 3, 7, 1, 0], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_numeric("val2", "int64").data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = 'val').first(target ='val2', ddf = ddf)
            
            self.assertListEqual([0, 1, 2, 3], ddf['val'].data[:].tolist())
            self.assertListEqual([0, 8, 6, 9], ddf['val2_first'].data[:].tolist())    


    def test_groupby_last_single_field(self):
        val = np.asarray([3, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=np.int32)
        val2 = np.asarray([9, 8, 2, 6, 4, 5, 3, 7, 1, 0], dtype=np.int64)
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_numeric("val2", "int64").data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.groupby(by = 'val').last(target ='val2', ddf = ddf)
            
            self.assertListEqual([0, 1, 2, 3], ddf['val'].data[:].tolist())
            self.assertListEqual([0, 2, 5, 1], ddf['val2_last'].data[:].tolist()) 


    def test_groupby_sorted_field(self):
        val = np.asarray([0,0,0,1,1,1,3], dtype=np.int32)
        val2 = np.asarray(['a','b','b','c','d','d','f'], dtype='S1')   
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)
            ddf = dst.create_dataframe('ddf')

            df.groupby(by = 'val').min(target ='val2', ddf = ddf)
            df.groupby(by = 'val').first(target ='val2', ddf = ddf, write_keys=False)

            self.assertListEqual([0, 1, 3], ddf['val'].data[:].tolist())
            self.assertListEqual([b'a', b'c', b'f'], ddf['val2_min'].data[:].tolist())
            self.assertListEqual([b'a', b'c', b'f'], ddf['val2_first'].data[:].tolist())


    def test_groupby_with_hint_keys_is_sorted(self):
        val = np.asarray([0,0,0,1,1,1,3], dtype=np.int32)
        val2 = np.asarray(['a','b','b','c','d','d','f'], dtype='S1')   
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_numeric("val", "int32").data.write(val)
            df.create_fixed_string("val2", 1).data.write(val2)
            ddf = dst.create_dataframe('ddf')

            df.groupby(by = 'val', hint_keys_is_sorted=True).max(target ='val2', ddf = ddf)
            df.groupby(by = 'val', hint_keys_is_sorted=True).last(target ='val2', ddf = ddf, write_keys=False)

            self.assertListEqual([0, 1, 3], ddf['val'].data[:].tolist())
            self.assertListEqual([b'b', b'd', b'f'], ddf['val2_max'].data[:].tolist())
            self.assertListEqual([b'b', b'd', b'f'], ddf['val2_last'].data[:].tolist())


class TestDataFrameSort(unittest.TestCase):

    def test_sort_values_on_original_df(self):
        idx = np.asarray([b'a', b'e', b'b', b'd', b'c'], dtype='S1')
        val = np.asarray([10, 20, 30, 40, 50], dtype=np.int32)
        val2 = ['a', 'ee', 'bbb', 'dddd', 'ccccc']

        bio = BytesIO()
        with session.Session(10) as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_fixed_string("idx", 1).data.write(idx)
            df.create_numeric("val", "int32").data.write(val)
            df.create_indexed_string("val2").data.write(val2)

            df.sort_values(by = 'idx')

            self.assertListEqual([b'a', b'b', b'c', b'd', b'e'], df['idx'].data[:].tolist())
            self.assertListEqual([10, 30, 50, 40, 20], df['val'].data[:].tolist())
            self.assertListEqual(['a', 'bbb', 'ccccc', 'dddd', 'ee'], df['val2'].data[:])


    def test_sort_values_on_other_df(self):
        idx = np.asarray([b'a', b'e', b'b', b'd', b'c'], dtype='S1')
        val = np.asarray([10, 20, 30, 40, 50], dtype=np.int32)
        val2 = ['a', 'ee', 'bbb', 'dddd', 'ccccc']

        bio = BytesIO()
        with session.Session(10) as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_fixed_string("idx", 1).data.write(idx)
            df.create_numeric("val", "int32").data.write(val)
            df.create_indexed_string("val2").data.write(val2)

            ddf = dst.create_dataframe('ddf')

            df.sort_values(by = 'idx', ddf = ddf)

            self.assertListEqual(list(idx), df['idx'].data[:].tolist())
            self.assertListEqual(list(val), df['val'].data[:].tolist())
            self.assertListEqual(list(val2), df['val2'].data[:])


            self.assertListEqual([b'a', b'b', b'c', b'd', b'e'], ddf['idx'].data[:].tolist())
            self.assertListEqual([10, 30, 50, 40, 20], ddf['val'].data[:].tolist())
            self.assertListEqual(['a', 'bbb', 'ccccc', 'dddd', 'ee'], ddf['val2'].data[:])


    def test_sort_values_on_inconsistent_length_df(self):
        idx = np.asarray([b'a', b'e', b'b', b'd', b'c'], dtype='S1')
        val = np.asarray([10, 20, 30, 40], dtype=np.int32)
        val2 = ['a', 'ee', 'bbb', 'dddd']

        bio = BytesIO()
        with session.Session(10) as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_fixed_string("idx", 1).data.write(idx)
            df.create_numeric("val", "int32").data.write(val)
            df.create_indexed_string("val2").data.write(val2)

            with self.assertRaises(ValueError) as context:
                df.sort_values(by = 'idx')

            self.assertEqual(str(context.exception), "There are consistent lengths in dataframe 'ds'. The following length were observed: {4, 5}") 


    def test_sort_values_on_invalid_input(self):
        idx = np.asarray([b'a', b'e', b'b', b'd', b'c'], dtype='S1')
        bio = BytesIO()
        with session.Session(10) as s:
            dst = s.open_dataset(bio, "w", "src")
            df = dst.create_dataframe('ds')
            df.create_fixed_string("idx", 1).data.write(idx)
        
            with self.assertRaises(ValueError) as context:
                df.sort_values(by = 'idx', axis=1)
            
            self.assertEqual(str(context.exception), "Currently sort_values() only supports axis = 0") 

            with self.assertRaises(ValueError) as context:
                df.sort_values(by = 'idx', ascending=False)
            
            self.assertEqual(str(context.exception), "Currently sort_values() only supports ascending = True")     
        
            with self.assertRaises(ValueError) as context:
                df.sort_values(by = 'idx', kind='quicksort')
            
            self.assertEqual(str(context.exception), "Currently sort_values() only supports kind='stable'")  


class TestDataFrameToCSV(unittest.TestCase):

    def test_to_csv_file(self):
        val1 = np.asarray([0, 1, 2, 3], dtype='int32')
        val2 = ['zero', 'one', 'two', 'three']
        bio = BytesIO()

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')

        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            df.create_numeric('val1', 'int32').data.write(val1)
            df.create_indexed_string('val2').data.write(val2)
            df.to_csv(csv_file_name)

        with open(csv_file_name, 'r') as f:
            self.assertEqual(f.readlines(), ['val1,val2\n', '0,zero\n', '1,one\n', '2,two\n', '3,three\n'])

        os.close(fd_csv)


    def test_to_csv_small_chunk_row_size(self):
        val1 = np.asarray([0, 1, 2, 3], dtype='int32')
        val2 = ['zero', 'one', 'two', 'three']
        bio = BytesIO()

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')

        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            df.create_numeric('val1', 'int32').data.write(val1)
            df.create_indexed_string('val2').data.write(val2)
            df.to_csv(csv_file_name, chunk_row_size=2)

        with open(csv_file_name, 'r') as f:
            self.assertEqual(f.readlines(), ['val1,val2\n', '0,zero\n', '1,one\n', '2,two\n', '3,three\n'])

        os.close(fd_csv) 


    def test_to_csv_with_column_filter(self):
        val1 = np.asarray([0, 1, 2, 3], dtype='int32')
        val2 = ['zero', 'one', 'two', 'three']
        bio = BytesIO()

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')

        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            df.create_numeric('val1', 'int32').data.write(val1)
            df.create_indexed_string('val2').data.write(val2)
            df.to_csv(csv_file_name, column_filter=['val1'])

        with open(csv_file_name, 'r') as f:
            self.assertEqual(f.readlines(), ['val1\n', '0\n', '1\n', '2\n', '3\n'])

        os.close(fd_csv)    


    def test_to_csv_with_row_filter_field(self):
        val1 = np.asarray([0, 1, 2, 3], dtype='int32')
        val2 = [True, False, True, False]
        bio = BytesIO()

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')

        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            df.create_numeric('val1', 'int32').data.write(val1)
            df.create_numeric('val2', 'bool').data.write(val2)
            df.to_csv(csv_file_name, row_filter=df['val2'])

        with open(csv_file_name, 'r') as f:
            self.assertEqual(f.readlines(), ['val1\n', '0\n', '2\n'])

        os.close(fd_csv)


class TestdataFrameToPandas(unittest.TestCase):

    def setUp(self):
        numeric_data = [i for i in range(20)]
        fixed_str_data = [b'a' for i in range(20)]
        ts_data = [1632234128 + i for i in range(20)]
        categorical_data = [1 for i in range(20)]
        idx_str_data = ['abc' for i in range(20)]

        bio = BytesIO()
        self.s = session.Session()
        dst = self.s.open_dataset(bio, 'w', 'dst')
        self.df = dst.create_dataframe('df')
        self.df.create_numeric('num', 'int32').data.write(numeric_data)
        self.df.create_fixed_string('fs1', 1).data.write(fixed_str_data)
        self.df.create_timestamp('ts1').data.write(ts_data)
        self.df.create_categorical('c1', 'int32', {'a': 1, 'b': 2}).data.write(categorical_data)
        self.df.create_indexed_string('is1').data.write(idx_str_data)

        self.pddf2 = pd.DataFrame(
            {'num': numeric_data, 'fs1': fixed_str_data, 'ts1': ts_data, 'c1': categorical_data, 'is1': idx_str_data})

    def tearDown(self):
        self.s.close()

    def test_to_pandas_df(self):
        pddf1 = self.df.to_pandas()
        pddf2 = self.pddf2.astype(pddf1.dtypes)
        self.assertTrue(pddf1.equals(pddf2))

        self.df.create_numeric('num2', 'int32').data.write([i for i in range(30)])
        self.assertRaises(ValueError, self.df.to_pandas)
        del self.df['num2']

    def test_to_pandas_df_row_filter(self):
        row_filter = [True if i % 2 == 0 else False for i in range(20)]
        pddf1 = self.df.to_pandas(row_filter=row_filter)
        pddf2 = self.pddf2.astype(pddf1.dtypes).loc[row_filter].reset_index(drop=True)
        self.assertTrue(pddf1.equals(pddf2))

    def test_to_pandas_df_col_filter(self):
        pddf1 = self.df.to_pandas(col_filter='num')
        pddf2 = pd.DataFrame(self.pddf2['num']).astype(pddf1.dtypes)
        self.assertTrue(pddf1.equals(pddf2))

        pddf1 = self.df.to_pandas(col_filter=['num', 'fs1', 'ts1', 'c1'])
        pddf2 = self.pddf2.astype(pddf1.dtypes)[['num', 'fs1', 'ts1', 'c1']]
        self.assertTrue(pddf1.equals(pddf2))

    def test_to_pandas_df_filter(self):
        row_filter = [True if i % 2 == 0 else False for i in range(20)]
        col_filter = ['num', 'fs1', 'ts1', 'c1']
        pddf1 = self.df.to_pandas(row_filter=row_filter, col_filter=col_filter)
        pddf2 = self.pddf2[col_filter].loc[row_filter].astype(pddf1.dtypes).reset_index(drop=True)
        self.assertTrue(pddf1.equals(pddf2))


class TestDataFrameDescribe(unittest.TestCase):

    def test_describe_default(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            df.create_numeric('num', 'int32').data.write([i for i in range(10)])
            df.create_fixed_string('fs1', 1).data.write([b'a' for i in range(20)])
            df.create_timestamp('ts1').data.write([1632234128 + i for i in range(20)])
            df.create_categorical('c1', 'int32', {'a': 1, 'b': 2}).data.write([1 for i in range(20)])
            df.create_indexed_string('is1').data.write(['abc' for i in range(20)])
            result = df.describe(output='None')
            expected = {'fields': ['num', 'ts1'], 'count': [10, 20], 'mean': ['4.50', '1632234137.50'],
                        'std': ['2.87', '5.77'], 'min': ['0.00', '1632234128.00'], '25%': ['0.02', '1632234128.05'],
                        '50%': ['0.04', '1632234128.10'], '75%': ['0.07', '1632234128.14'],
                        'max': ['9.00', '1632234147.00']}
            self.assertEqual(result, expected)

    def test_describe_include(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            df.create_numeric('num', 'int32').data.write([i for i in range(10)])
            df.create_fixed_string('fs1', 1).data.write([b'a' for i in range(20)])
            df.create_timestamp('ts1').data.write([1632234128 + i for i in range(20)])
            df.create_categorical('c1', 'int32', {'a': 1, 'b': 2}).data.write([1 for i in range(20)])
            df.create_indexed_string('is1').data.write(['abc' for i in range(20)])

            result = df.describe(include='all', output='None')
            expected = {'fields': ['num', 'fs1', 'ts1', 'c1', 'is1'], 'count': [10, 20, 20, 20, 20],
                        'mean': ['4.50', 'NaN', '1632234137.50', 'NaN', 'NaN'], 'std': ['2.87', 'NaN', '5.77', 'NaN', 'NaN'],
                        'min': ['0.00', 'NaN', '1632234128.00', 'NaN', 'NaN'], '25%': ['0.02', 'NaN', '1632234128.05', 'NaN', 'NaN'],
                        '50%': ['0.04', 'NaN', '1632234128.10', 'NaN', 'NaN'], '75%': ['0.07', 'NaN', '1632234128.14', 'NaN', 'NaN'],
                        'max': ['9.00', 'NaN', '1632234147.00', 'NaN', 'NaN'], 'unique': ['NaN', 1, 'NaN', 1, 1],
                        'top': ['NaN', b'a', 'NaN', 1, 'abc'], 'freq': ['NaN', 20, 'NaN', 20, 20]}
            self.assertEqual(result, expected)

            result = df.describe(include='num', output='None')
            expected = {'fields': ['num'], 'count': [10], 'mean': ['4.50'], 'std': ['2.87'], 'min': ['0.00'],
                        '25%': ['0.02'], '50%': ['0.04'], '75%': ['0.07'], 'max': ['9.00']}
            self.assertEqual(result, expected)

            result = df.describe(include=['num', 'fs1'], output='None')
            expected = {'fields': ['num', 'fs1'], 'count': [10, 20], 'mean': ['4.50', 'NaN'], 'std': ['2.87', 'NaN'],
                        'min': ['0.00', 'NaN'], '25%': ['0.02', 'NaN'], '50%': ['0.04', 'NaN'], '75%': ['0.07', 'NaN'],
                        'max': ['9.00', 'NaN'], 'unique': ['NaN', 1], 'top': ['NaN', b'a'], 'freq': ['NaN', 20]}
            self.assertEqual(result, expected)

            result = df.describe(include=np.int32, output='None')
            expected = {'fields': ['num', 'c1'], 'count': [10, 20], 'mean': ['4.50', 'NaN'], 'std': ['2.87', 'NaN'],
                        'min': ['0.00', 'NaN'], '25%': ['0.02', 'NaN'], '50%': ['0.04', 'NaN'], '75%': ['0.07', 'NaN'],
                        'max': ['9.00', 'NaN'], 'unique': ['NaN', 1], 'top': ['NaN', 1], 'freq': ['NaN', 20]}
            self.assertEqual(result, expected)

            result = df.describe(include=[np.int32, np.bytes_], output='None')
            expected = {'fields': ['num', 'c1', 'fs1'], 'count': [10, 20, 20], 'mean': ['4.50', 'NaN', 'NaN'],
                        'std': ['2.87', 'NaN', 'NaN'], 'min': ['0.00', 'NaN', 'NaN'], '25%': ['0.02', 'NaN', 'NaN'],
                        '50%': ['0.04', 'NaN', 'NaN'], '75%': ['0.07', 'NaN', 'NaN'], 'max': ['9.00', 'NaN', 'NaN'],
                        'unique': ['NaN', 1, 1], 'top': ['NaN', 1, b'a'], 'freq': ['NaN', 20, 20]}
            self.assertEqual(result, expected)


    def test_describe_exclude(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_numeric('num', 'int32').data.write([i for i in range(10)])
            df.create_numeric('num2', 'int64').data.write([i for i in range(10)])
            df.create_fixed_string('fs1', 1).data.write([b'a' for i in range(20)])
            df.create_timestamp('ts1').data.write([1632234128 + i for i in range(20)])
            df.create_categorical('c1', 'int32', {'a': 1, 'b': 2}).data.write([1 for i in range(20)])
            df.create_indexed_string('is1').data.write(['abc' for i in range(20)])

            result = df.describe(exclude='num', output='None')
            expected = {'fields': ['num2', 'ts1'], 'count': [10, 20], 'mean': ['4.50', '1632234137.50'],
                        'std': ['2.87', '5.77'], 'min': ['0.00', '1632234128.00'], '25%': ['0.02', '1632234128.05'],
                        '50%': ['0.04', '1632234128.10'], '75%': ['0.07', '1632234128.14'],
                        'max': ['9.00', '1632234147.00']}
            self.assertEqual(result, expected)

            result = df.describe(exclude=['num', 'num2'], output='None')
            expected = {'fields': ['ts1'], 'count': [20], 'mean': ['1632234137.50'], 'std': ['5.77'],
                        'min': ['1632234128.00'], '25%': ['1632234128.05'], '50%': ['1632234128.10'],
                        '75%': ['1632234128.14'], 'max': ['1632234147.00']}
            self.assertEqual(result, expected)

            result = df.describe(exclude=np.int32, output='None')
            expected = {'fields': ['num2', 'ts1'], 'count': [10, 20], 'mean': ['4.50', '1632234137.50'],
                        'std': ['2.87', '5.77'], 'min': ['0.00', '1632234128.00'], '25%': ['0.02', '1632234128.05'],
                        '50%': ['0.04', '1632234128.10'], '75%': ['0.07', '1632234128.14'],
                        'max': ['9.00', '1632234147.00']}
            self.assertEqual(result, expected)

            result = df.describe(exclude=[np.int32, np.float64], output='None')
            expected = {'fields': ['num2'], 'count': [10], 'mean': ['4.50'], 'std': ['2.87'], 'min': ['0.00'],
                        '25%': ['0.02'], '50%': ['0.04'], '75%': ['0.07'], 'max': ['9.00']}
            self.assertEqual(result, expected)

    def test_describe_include_and_exclude(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_numeric('num', 'int32').data.write([i for i in range(10)])
            df.create_numeric('num2', 'int64').data.write([i for i in range(10)])
            df.create_fixed_string('fs1', 1).data.write([b'a' for i in range(20)])
            df.create_timestamp('ts1').data.write([1632234128 + i for i in range(20)])
            df.create_categorical('c1', 'int32', {'a': 1, 'b': 2}).data.write([1 for i in range(20)])
            df.create_indexed_string('is1').data.write(['abc' for i in range(20)])

            #str *
            with self.assertRaises(Exception) as context:
                df.describe(include='num', exclude='num', output='None')
            self.assertTrue(isinstance(context.exception, ValueError))

            # list of str , str
            with self.assertRaises(Exception) as context:
                df.describe(include=['num', 'num2'], exclude='num', output='None')
            self.assertTrue(isinstance(context.exception, ValueError))
            # list of str , type
            result = df.describe(include=['num', 'num2'], exclude=np.int32, output='None')
            expected = {'fields': ['num2'], 'count': [10], 'mean': ['4.50'], 'std': ['2.87'], 'min': ['0.00'],
                        '25%': ['0.02'], '50%': ['0.04'], '75%': ['0.07'], 'max': ['9.00']}
            self.assertEqual(result, expected)
            # list of str , list of str
            with self.assertRaises(Exception) as context:
                df.describe(include=['num', 'num2'], exclude=['num', 'num2'], output='None')
            self.assertTrue(isinstance(context.exception, ValueError))
            # list of str , list of type
            result = df.describe(include=['num', 'num2', 'ts1'], exclude=[np.int32, np.int64], output='None')
            expected = {'fields': ['ts1'], 'count': [20], 'mean': ['1632234137.50'], 'std': ['5.77'],
                        'min': ['1632234128.00'], '25%': ['1632234128.05'], '50%': ['1632234128.10'],
                        '75%': ['1632234128.14'], 'max': ['1632234147.00']}
            self.assertEqual(result, expected)

            # type, str
            result = df.describe(include=np.number, exclude='num2', output='None')
            expected = {'fields': ['num', 'ts1', 'c1'], 'count': [10, 20, 20], 'mean': ['4.50', '1632234137.50', 'NaN'],
                       'std': ['2.87', '5.77', 'NaN'], 'min': ['0.00', '1632234128.00', 'NaN'],
                       '25%': ['0.02', '1632234128.05', 'NaN'], '50%': ['0.04', '1632234128.10', 'NaN'],
                       '75%': ['0.07', '1632234128.14', 'NaN'], 'max': ['9.00', '1632234147.00', 'NaN'],
                       'unique': ['NaN', 'NaN', 1], 'top': ['NaN', 'NaN', 1], 'freq': ['NaN', 'NaN', 20]}
            self.assertEqual(result, expected)
            # type, type
            with self.assertRaises(Exception) as context:
                df.describe(include=np.int32, exclude=np.int64)
            self.assertTrue(isinstance(context.exception, ValueError))
            # type, list of str
            result = df.describe(include=np.number, exclude=['num', 'num2'], output='None')
            expected = {'fields': ['ts1', 'c1'], 'count': [20, 20], 'mean': ['1632234137.50', 'NaN'],
                        'std': ['5.77', 'NaN'], 'min': ['1632234128.00', 'NaN'], '25%': ['1632234128.05', 'NaN'],
                        '50%': ['1632234128.10', 'NaN'], '75%': ['1632234128.14', 'NaN'], 'max': ['1632234147.00', 'NaN'],
                        'unique': ['NaN', 1], 'top': ['NaN', 1], 'freq': ['NaN', 20]}
            self.assertEqual(result, expected)
            # type, list of type
            with self.assertRaises(Exception) as context:
                df.describe(include=np.int32, exclude=[np.int64, np.float64], output='None')
            self.assertTrue(isinstance(context.exception, ValueError))

            # list of type, str
            result = df.describe(include=[np.int32, np.int64], exclude='num', output='None')
            expected = {'fields': ['c1', 'num2'], 'count': [20, 10], 'mean': ['NaN', '4.50'], 'std': ['NaN', '2.87'],
                        'min': ['NaN', '0.00'], '25%': ['NaN', '0.02'], '50%': ['NaN', '0.04'], '75%': ['NaN', '0.07'],
                        'max': ['NaN', '9.00'], 'unique': [1, 'NaN'], 'top': [1, 'NaN'], 'freq': [20, 'NaN']}
            self.assertEqual(result, expected)
            # list of type, type
            with self.assertRaises(Exception) as context:
                df.describe(include=[np.int32, np.int64], exclude=np.int64, output='None')
            self.assertTrue(isinstance(context.exception, ValueError))
            # list of type, list of str
            result = df.describe(include=[np.int32, np.int64], exclude=['num', 'num2'], output='None')
            expected = {'fields': ['c1'], 'count': [20], 'mean': ['NaN'], 'std': ['NaN'], 'min': ['NaN'],
                        '25%': ['NaN'], '50%': ['NaN'], '75%': ['NaN'], 'max': ['NaN'], 'unique': [1], 'top': [1],
                        'freq': [20]}
            self.assertEqual(result, expected)
            # list of type, list of type
            with self.assertRaises(Exception) as context:
                df.describe(include=[np.int32, np.int64], exclude=[np.int32, np.int64])
            self.assertTrue(isinstance(context.exception, ValueError))

    def test_raise_errors(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')

            df.create_fixed_string('fs1', 1).data.write([b'a' for i in range(20)])
            df.create_categorical('c1', 'int32', {'a': 1, 'b': 2}).data.write([1 for i in range(20)])
            df.create_indexed_string('is1').data.write(['abc' for i in range(20)])

            with self.assertRaises(Exception) as context:
                df.describe(include='num3')
            self.assertTrue(isinstance(context.exception, ValueError))

            with self.assertRaises(Exception) as context:
                df.describe(include=np.int8)
            self.assertTrue(isinstance(context.exception, ValueError))

            with self.assertRaises(Exception) as context:
                df.describe(include=['num3', 'num4'])
            self.assertTrue(isinstance(context.exception, ValueError))

            with self.assertRaises(Exception) as context:
                df.describe(include=[np.int8, np.uint])
            self.assertTrue(isinstance(context.exception, ValueError))

            with self.assertRaises(Exception) as context:
                df.describe(include=float('3.14159'))
            self.assertTrue(isinstance(context.exception, ValueError))

            with self.assertRaises(Exception) as context:
                df.describe()
            self.assertTrue(isinstance(context.exception, ValueError))

            df.create_numeric('num', 'int32').data.write([i for i in range(10)])
            df.create_numeric('num2', 'int64').data.write([i for i in range(10)])
            df.create_timestamp('ts1').data.write([1632234128 + i for i in range(20)])

            with self.assertRaises(Exception) as context:
                df.describe(exclude=float('3.14159'))
            self.assertTrue(isinstance(context.exception, ValueError))

            with self.assertRaises(Exception) as context:
                df.describe(exclude=['num', 'num2', 'ts1'])
            self.assertTrue(isinstance(context.exception, ValueError))


class TestDataFrameView(SessionTestCase):

    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_get_view(self, creator, name, kwargs, data):
        """
        Test dataframe.view, field.is_view, apply_filter, and apply_index
        """
        f = self.setup_field(self.df, creator, name, (), kwargs, data)
        if "nformat" in kwargs:
            data = np.asarray(data, dtype=kwargs["nformat"])
        else:
            data = np.asarray(data)

        view = self.df.view()
        self.assertTrue(view[name].is_view())
        self.assertListEqual(data[:].tolist(), np.asarray(view[name].data[:]).tolist())

        with self.subTest('All False:'):
            df2 = self.ds.create_dataframe('df2')
            d_filter = np.array([False])
            self.df.apply_filter(d_filter, df2)
            self.assertTrue(df2[name].is_view())
            d_filter = np.nonzero(d_filter)[0]
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())

            d_filter = np.array([True]*len(data))
            self.df.apply_filter(d_filter, df2)
            d_filter = np.nonzero(d_filter)[0]
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())
            self.ds.drop('df2')

        with self.subTest('All True:'):
            df2 = self.ds.create_dataframe('df2')
            d_filter = np.array([True]*len(data))
            self.df.apply_filter(d_filter, df2)
            self.assertTrue(df2[name].is_view())
            d_filter = np.nonzero(d_filter)[0]
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())

            d_filter = np.array([np.random.random()>=0.5 for i in range(len(data))])
            self.df.apply_filter(d_filter, df2)
            d_filter = np.nonzero(d_filter)[0]
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())
            self.ds.drop('df2')

        with self.subTest('Ramdon T/F'):
            df2 = self.ds.create_dataframe('df2')
            d_filter = np.array([np.random.random()>=0.5 for i in range(len(data))])
            self.df.apply_filter(d_filter, df2)
            self.assertTrue(df2[name].is_view())
            d_filter = np.nonzero(d_filter)[0]
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())
            self.ds.drop('df2')

        with self.subTest('All Index:'):
            df2 = self.ds.create_dataframe('df2')
            d_filter = np.array([i for i in range(len(data))])
            self.df.apply_index(d_filter, df2)
            self.assertTrue(df2[name].is_view())
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())
            self.ds.drop('df2')

        with self.subTest('Random Index:'):
            df2 = self.ds.create_dataframe('df2')
            d_filter = []
            for i in range(len(data)):
                if np.random.random() >= 0.5:
                    d_filter.append(i)
            d_filter = np.array(d_filter)
            self.df.apply_index(d_filter, df2)
            self.assertTrue(df2[name].is_view())
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())

            d_filter = np.array([i for i in range(len(data))])
            self.df.apply_index(d_filter, df2)
            self.assertTrue(df2[name].is_view())
            self.assertListEqual(data[d_filter].tolist(), np.asarray(df2[name].data[:]).tolist())
            self.ds.drop('df2')

    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_concrete_field(self, creator, name, kwargs, data):
        """
        Test field.attach, field.detach, field.notify, field.update
        """
        f = self.setup_field(self.df, creator, name, (), kwargs, data)
        if "nformat" in kwargs:
            data = np.asarray(data, dtype=kwargs["nformat"])
        else:
            data = np.asarray(data)
        view = self.df.view()
        self.assertTrue(view[name] in f._view_refs)  # attached
        f.data.clear()
        self.assertListEqual([], np.asarray(f.data[:]).tolist())
        self.assertListEqual(data.tolist(), np.asarray(view[name].data[:]).tolist())  # notify and update
        self.assertFalse(view[name] in f._view_refs)  # detached

    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_view_persistence(self, creator, name, kwargs, data):
        """
        The view should be persistent over sessions.
        """
        bio = BytesIO()
        src = self.s.open_dataset(bio, 'w', 'src')
        df = src.create_dataframe('df')
        f = self.setup_field(df, creator, name, (), kwargs, data)
        df2 = src.create_dataframe('df2')
        d_filter = np.array([np.random.random()>=0.5 for i in range(len(data))])
        df.apply_filter(d_filter, df2)
        self.assertTrue(df2[name].is_view())
        self.s.close()

        src = self.s.open_dataset(bio, 'r+', 'src')
        df = src['df']
        df2 = src['df2']
        self.assertTrue(df2[name].is_view())
        self.assertTrue(df2[name] in df[name]._view_refs)

