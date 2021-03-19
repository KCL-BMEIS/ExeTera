import unittest
from io import BytesIO

from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per
from exetera.core import dataframe


class TestDataFrame(unittest.TestCase):

    def test_dataframe_init(self):
        bio=BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio,'w','dst')
            numf = s.create_numeric(dst,'numf','int32')
            #init
            df = dataframe.DataFrame()
            self.assertTrue(isinstance(df, dataframe.DataFrame))
            fdf = {'/numf',numf}
            df2 = dataframe.DataFrame(fdf)
            self.assertTrue(isinstance(df2,dataframe.DataFrame))
            #add & set & contains
            df.add(numf)
            self.assertTrue('/numf' in df)
            self.assertTrue(df.contains_field(numf))
            cat=s.create_categorical(dst,'cat','int8',{'a':1,'b':2})
            self.assertFalse('/cat' in df)
            self.assertFalse(df.contains_field(cat))
            df['/cat']=cat
            self.assertTrue('/cat' in df)
            #list & get
            self.assertEqual(id(numf),id(df.get_field('/numf')))
            self.assertEqual(id(numf), id(df['/numf']))
            self.assertEqual('/numf',df.get_name(numf))
            #list & iter
            dfit = iter(df)
            self.assertEqual('/numf',next(dfit))
            self.assertEqual('/cat', next(dfit))
            #del & del by field
            del df['/numf']
            self.assertFalse('/numf' in df)
            df.delete_field(cat)
            self.assertFalse(df.contains_field(cat))
            self.assertIsNone(df.get_name(cat))


