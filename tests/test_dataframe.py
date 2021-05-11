import unittest
from io import BytesIO
import numpy as np

from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per
from exetera.core import dataframe


class TestDataFrameCreateFields(unittest.TestCase):

    def test_dataframe_init(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            # init
            df = dst.create_dataframe('dst')
            self.assertTrue(isinstance(df, dataframe.DataFrame))
            numf = df.create_numeric('numf', 'uint32')
            df2 = dst.create_dataframe('dst2', dataframe=df)
            self.assertTrue(isinstance(df2, dataframe.DataFrame))

            # add & set & contains
            self.assertTrue('numf' in df)
            self.assertTrue('numf' in df2)
            cat = s.create_categorical(df2, 'cat', 'int8', {'a': 1, 'b': 2})
            self.assertFalse('cat' in df)
            self.assertFalse(df.contains_field(cat))
            df['cat'] = cat
            self.assertTrue('cat' in df)

            # list & get
            self.assertEqual(id(numf), id(df.get_field('numf')))
            self.assertEqual(id(numf), id(df['numf']))

            # list & iter
            dfit = iter(df)
            self.assertEqual('numf', next(dfit))
            self.assertEqual('cat', next(dfit))

            # del & del by field
            del df['numf']
            self.assertFalse('numf' in df)
            with self.assertRaises(ValueError, msg="This field is owned by a different dataframe"):
                df.delete_field(cat)
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

            total = np.sum(a.data[:])
            self.assertEqual(49997540637149, total)

            a.data[:] = a.data[:] * 2
            total = np.sum(a.data[:])
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

            df2 = dst.create_dataframe('df2')
            dataframe.copy(numf, df2,'numf')
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

            filter_to_apply = np.array([True, True, False, False, True])
            ddf = dst.create_dataframe('dst3')
            df.apply_filter(filter_to_apply, ddf=ddf)
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
            print(fa.name, fa.data[:])
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
        cfilt = ['num2', 'fixed']
        expected = src[filt].tolist()

        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')
            numf = s.create_numeric(df, 'numf', 'int32')
            numf.data.write(src)
            numf = s.create_numeric(df, 'num2', 'int32')
            numf.data.write(src)
            numf = s.create_numeric(df, 'num3', 'int32')
            numf.data.write(src)
            fixed = s.create_fixed_string(df, 'fixed', 1)
            fixed.data.write([b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h'])

            # soft filter
            df.apply_filter(filt)
            df.apply_filter(cfilt)
            self.assertTrue(df['numf'] is None)  # get item masked

            # hard filter other df
            df2 = dst.create_dataframe('df2')
            df2b = df.apply_filter(filt, ddf=df2)
            self.assertListEqual(expected, df2['numf'].data[:].tolist())
            self.assertListEqual(expected, df2b['numf'].data[:].tolist())
            self.assertListEqual(src.tolist(), df['numf'].data[:].tolist())
            # hard filter inline
            df.apply_filter(filt, ddf=df)
            self.assertListEqual(expected, df['numf'].data[:].tolist())
            # hard filter memory
            memdf = df.apply_filter(filt, ddf=None)





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
            self.assertEqual(ddf['r_id_2'].data[:].tolist(), ddf['r_id_2'].data[:].tolist())

