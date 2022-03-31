import unittest

import h5py
import numpy as np

from exetera.core import session, fields
from exetera.core.abstract_types import DataFrame
from io import BytesIO
from exetera.core.dataset import HDF5Dataset, copy, move


class TestDataSet(unittest.TestCase):

    def test_dataset_init(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            df = dst.create_dataframe('df')

            #create field using session api
            num = s.create_numeric(df,'num', 'int32')
            num.data.write([1, 2, 3, 4])
            self.assertEqual([1, 2, 3, 4], num.data[:].tolist())

            cat = s.create_categorical(df, 'cat', 'int8', {'a': 1, 'b': 2})
            cat.data.write([1 , 1, 2, 2])
            self.assertEqual([1, 1, 2, 2], s.get(df['cat']).data[:].tolist())

            #create field using dataframe api
            idsf = df.create_indexed_string('idsf')
            idsf.data.write(['a', 'bb', 'ccc', 'dddd'])
            self.assertEqual(['a', 'bb', 'ccc', 'dddd'], df['idsf'].data[:])

            fsf = df.create_fixed_string('fsf', 3)
            fsf.data.write([b'aaa', b'bbb', b'ccc', b'ddd'])
            self.assertEqual([b'aaa', b'bbb', b'ccc', b'ddd'], df['fsf'].data[:].tolist())

    def test_dataset_init_with_data(self):
        bio = BytesIO()
        with session.Session() as s:
            h5file = h5py.File(bio, 'w')
            hgrp1 = h5file.create_group("grp1")
            num1 = s.create_numeric(hgrp1, 'num1', 'uint32')
            num1.data.write(np.array([0, 1, 2, 3, 4]))
            h5file.close()

            # read existing datafile
            dst = s.open_dataset(bio, 'r+', 'dst')
            self.assertTrue(isinstance(dst['grp1'], DataFrame))
            self.assertEqual(s.get(dst['grp1']['num1']).data[:].tolist(), [0, 1, 2, 3, 4])

            # add dataframe
            bio2 = BytesIO()
            ds2 = s.open_dataset(bio2, 'w', 'ds2')
            df2 = ds2.create_dataframe('df2')
            fs = df2.create_fixed_string('fs', 1)
            fs.data.write([b'a', b'b', b'c', b'd'])

            dst.copy(df2, 'df2')
            self.assertTrue(isinstance(dst['df2'], DataFrame))
            self.assertEqual([b'a', b'b', b'c', b'd'], dst['df2']['fs'].data[:].tolist())

            del dst['df2']
            self.assertTrue(len(dst.keys()) == 1)
            self.assertTrue(len(dst._file.keys()) == 1)

            # set dataframe (this is a copy between datasets
            dst['df3'] = df2
            self.assertTrue(isinstance(dst['df3'], DataFrame))
            self.assertEqual([b'a', b'b', b'c', b'd'], dst['df3']['fs'].data[:].tolist())

            # set dataframe within the same dataset (rename)
            dst['df4'] = dst['df3']
            self.assertTrue(isinstance(dst['df4'], DataFrame))
            self.assertEqual([b'a', b'b', b'c', b'd'], dst['df4']['fs'].data[:].tolist())

    def test_dataset_static_func(self):
        bio = BytesIO()
        bio2 = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            df = dst.create_dataframe('df')
            num1 = df.create_numeric('num', 'uint32')
            num1.data.write([1, 2, 3, 4])

            ds2 = s.open_dataset(bio2, 'r+', 'ds2')
            copy(df, ds2, 'df2')
            self.assertTrue(isinstance(ds2['df2'], DataFrame))
            self.assertTrue(isinstance(ds2['df2']['num'], fields.Field))

            ds2.drop('df2')
            self.assertTrue(len(ds2) == 0)

            df2 = ds2.create_dataframe('df2')
            self.assertTrue(ds2.contains_dataframe(df2))
            self.assertFalse(dst.create_dataframe('foo'))

            dst.delete_dataframe(dst['foo'])
            ds2.delete_dataframe(df2)
            self.assertFalse(ds2.contains_dataframe(df2))

            self.assertListEqual(['df'], list(dst.keys()))
            self.assertListEqual([df], list(dst.values()))
            self.assertDictEqual({'df': df}, {k: v for k, v in dst.items()})

            move(df, ds2, 'df2')
            self.assertTrue(len(dst) == 0)
            self.assertTrue(len(ds2) == 1)

    def test_dataframe_create_with_dataframe(self):

        iscontents1 = ['a', 'bb', 'ccc', 'dddd']
        iscontents2 = ['eeee', 'fff', 'gg', 'h']
        fscontents1 = [s.encode() for s in iscontents1]
        fscontents2 = [s.encode() for s in iscontents2]
        ccontents1 = np.array([1, 2, 2, 1], dtype=np.int8)
        ccontents2 = np.array([2, 1, 1, 2], dtype=np.int8)
        ncontents1 = np.array([1, 2, 3, 4], dtype=np.int32)
        ncontents2 = np.array([5, 6, 7, 8], dtype=np.int32)
        from datetime import datetime as D
        tcontents1 = [D(2020, 1, 1), D(2020, 1, 2), D(2020, 1, 3), D(2020, 1, 4)]
        tcontents1 = np.array([d.timestamp() for d in tcontents1])
        tcontents2 = [D(2021, 1, 1), D(2021, 1, 2), D(2021, 1, 3), D(2021, 1, 4)]
        tcontents2 = np.array([d.timestamp() for d in tcontents2])

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df1 = ds.create_dataframe('df1')
            df1.create_indexed_string('is_foo').data.write(iscontents1)
            df1.create_fixed_string('fs_foo', 4).data.write(fscontents1)
            df1.create_categorical('c_foo', 'int8', {b'a': 1, b'b': 2}).data.write(ccontents1)
            df1.create_numeric('n_foo', 'uint32').data.write(ncontents1)
            df1.create_timestamp('t_foo').data.write(tcontents1)

            df2 = ds.create_dataframe('df2', dataframe=df1)

            self.assertListEqual(iscontents1, df1['is_foo'].data[:])
            self.assertListEqual(iscontents1, df2['is_foo'].data[:])
            df2['is_foo'].data.clear()
            df2['is_foo'].data.write(iscontents2)
            self.assertListEqual(iscontents1, df1['is_foo'].data[:])
            self.assertListEqual(iscontents2, df2['is_foo'].data[:])

            self.assertListEqual(fscontents1, df1['fs_foo'].data[:].tolist())
            self.assertListEqual(fscontents1, df2['fs_foo'].data[:].tolist())
            df2['fs_foo'].data[:] = fscontents2
            self.assertListEqual(fscontents1, df1['fs_foo'].data[:].tolist())
            self.assertListEqual(fscontents2, df2['fs_foo'].data[:].tolist())

            self.assertListEqual(ccontents1.tolist(), df1['c_foo'].data[:].tolist())
            self.assertListEqual(ccontents1.tolist(), df2['c_foo'].data[:].tolist())
            df2['c_foo'].data[:] = ccontents2
            self.assertListEqual(ccontents1.tolist(), df1['c_foo'].data[:].tolist())
            self.assertListEqual(ccontents2.tolist(), df2['c_foo'].data[:].tolist())
            self.assertDictEqual({1: b'a', 2: b'b'}, df1['c_foo'].keys)
            self.assertDictEqual({1: b'a', 2: b'b'}, df2['c_foo'].keys)

            self.assertListEqual(ncontents1.tolist(), df1['n_foo'].data[:].tolist())
            self.assertListEqual(ncontents1.tolist(), df2['n_foo'].data[:].tolist())
            df2['n_foo'].data[:] = np.array(ncontents2, dtype=np.uint32)
            self.assertListEqual(ncontents1.tolist(), df1['n_foo'].data[:].tolist())
            self.assertListEqual(ncontents2.tolist(), df2['n_foo'].data[:].tolist())

            self.assertListEqual(tcontents1.tolist(), df1['t_foo'].data[:].tolist())
            self.assertListEqual(tcontents1.tolist(), df2['t_foo'].data[:].tolist())
            df2['t_foo'].data[:] = np.array(tcontents2, dtype=np.float64)
            self.assertListEqual(tcontents1.tolist(), df1['t_foo'].data[:].tolist())
            self.assertListEqual(tcontents2.tolist(), df2['t_foo'].data[:].tolist())

    def test_dataset_ops(self):
        pass
