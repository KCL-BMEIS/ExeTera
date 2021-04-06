import unittest
from io import BytesIO
import numpy as np

from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per
from exetera.core import dataframe


class TestDataFrame(unittest.TestCase):

    def test_dataframe_init(self):
        bio=BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio,'w','dst')
            #init
            df = dataframe.DataFrame('dst',dst)
            self.assertTrue(isinstance(df, dataframe.DataFrame))
            numf = df.create_numeric(s,'numf','uint32')
            fdf = {'numf',numf}
            df2 = dataframe.DataFrame('dst2',dst,data=fdf)
            self.assertTrue(isinstance(df2,dataframe.DataFrame))
            #add & set & contains
            df.add(numf)
            self.assertTrue('numf' in df)
            self.assertTrue(df.contains_field(numf))
            cat=s.create_categorical(df2,'cat','int8',{'a':1,'b':2})
            self.assertFalse('cat' in df)
            self.assertFalse(df.contains_field(cat))
            df['cat']=cat
            self.assertTrue('cat' in df)
            #list & get
            self.assertEqual(id(numf),id(df.get_field('numf')))
            self.assertEqual(id(numf), id(df['numf']))
            self.assertEqual('numf',df.get_name(numf))
            #list & iter
            dfit = iter(df)
            self.assertEqual('numf',next(dfit))
            self.assertEqual('cat', next(dfit))
            #del & del by field
            del df['numf']
            self.assertFalse('numf' in df)
            df.delete_field(cat)
            self.assertFalse(df.contains_field(cat))
            self.assertIsNone(df.get_name(cat))

    def test_dataframe_init_fromh5(self):
        bio = BytesIO()
        with session.Session() as s:
            ds=s.open_dataset(bio,'w','ds')
            dst = ds.create_dataframe('dst')
            num=s.create_numeric(dst,'num','uint8')
            num.data.write([1,2,3,4,5,6,7])
            df = dataframe.DataFrame('dst',dst,h5group=dst)

    def test_dataframe_create_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'r+', 'dst')
            df = dataframe.DataFrame('dst',dst)
            num = df.create_numeric(s,'num','uint32')
            num.data.write([1,2,3,4])
            self.assertEqual([1,2,3,4],num.data[:].tolist())

    # def test_dataframe_ops(self):
    #     bio = BytesIO()
    #     with session.Session() as s:
    #         dst = s.open_dataset(bio, 'w', 'dst')
    #         df = dataframe.DataFrame('dst',dst)
    #         numf = s.create_numeric(dst, 'numf', 'int32')
    #         numf.data.write([5,4,3,2,1])
    #         df.add(numf)
    #         fst = s.create_fixed_string(dst,'fst',3)
    #         fst.data.write([b'e',b'd',b'c',b'b',b'a'])
    #         df.add(fst)
    #         index=np.array([4,3,2,1,0])
    #         ddf = dataframe.DataFrame('dst2',dst)
    #         df.apply_index(index,ddf)
    #         self.assertEqual([1,2,3,4,5],ddf.get_field('/numf').data[:].tolist())
    #         self.assertEqual([b'a',b'b',b'c',b'd',b'e'],ddf.get_field('/fst').data[:].tolist())
    #
    #         filter= np.array([True,True,False,False,True])
    #         df.apply_filter(filter)
    #         self.assertEqual([1, 2, 5], df.get_field('/numf').data[:].tolist())
    #         self.assertEqual([b'a', b'b', b'e'], df.get_field('/fst').data[:].tolist())


