import unittest

import h5py
import numpy as np

from exetera.core import session
from exetera.core.abstract_types import DataFrame
from io import BytesIO
from exetera.core import data_writer


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
            cat.data.write([1,1,2,2])
            self.assertEqual([1,1,2,2],s.get(df['cat']).data[:].tolist())

            #create field using dataframe api
            idsf = df.create_indexed_string('idsf')
            idsf.data.write(['a','bb','ccc','dddd'])
            self.assertEqual(['a','bb','ccc','dddd'],df['idsf'].data[:])

            fsf = df.create_fixed_string('fsf',3)
            fsf.data.write([b'aaa',b'bbb',b'ccc',b'ddd'])
            self.assertEqual([b'aaa',b'bbb',b'ccc',b'ddd'],df['fsf'].data[:].tolist())

    def test_dataset_init_with_data(self):
        bio = BytesIO()
        with session.Session() as s:
            h5file = h5py.File(bio, 'w')
            hgrp1 = h5file.create_group("grp1") #dataframe
            num1 = s.create_numeric(hgrp1, 'num1', 'uint32')
            num1.data.write(np.array([0, 1, 2, 3, 4]))
            # num1 = h5file["grp1"].create_group('num1') #field
            # num1.attrs['fieldtype'] = 'numeric,{}'.format('uint32')
            # num1.attrs['nformat'] = 'uint32'
            # ds=num1.create_dataset('values', (5,), dtype='uint32')
            # ds[:]=np.array([0, 1, 2, 3, 4])
            h5file.close()

            #read existing datafile
            dst=s.open_dataset(bio, 'r+', 'dst')
            self.assertTrue(isinstance(dst['grp1'], DataFrame))
            self.assertEqual(s.get(dst['grp1']['num1']).data[:].tolist(), [0, 1, 2, 3, 4])
            #add dataframe
            bio2 = BytesIO()
            ds2 = s.open_dataset(bio2, 'w', 'ds2')
            df2=ds2.create_dataframe('df2')
            fs=df2.create_fixed_string('fs', 1)
            fs.data.write([b'a', b'b', b'c', b'd'])

            dst.add(df2)
            self.assertTrue(isinstance(dst['df2'], DataFrame))
            self.assertEqual([b'a', b'b', b'c', b'd'], dst['df2']['fs'].data[:].tolist())

            #del dataframe
            del dst['df2'] #only 'grp1' left
            self.assertTrue(len(dst.keys()) == 1)
            self.assertTrue(len(dst._file.keys()) == 1)

            #set dataframe
            dst['grp1']=df2
            self.assertTrue(isinstance(dst['grp1'], DataFrame))
            self.assertEqual([b'a', b'b', b'c', b'd'], dst['grp1']['fs'].data[:].tolist())
