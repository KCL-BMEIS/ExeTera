import unittest

import numpy as np
from io import BytesIO

import h5py

from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per


class TestFieldExistence(unittest.TestCase):

    def test_field_truthness(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, "w", "src")
            f = s.create_indexed_string(src, "a")
            self.assertTrue(bool(f))
            f = s.create_fixed_string(src, "b", 5)
            self.assertTrue(bool(f))
            f = s.create_numeric(src, "c", "int32")
            self.assertTrue(bool(f))
            f = s.create_categorical(src, "d", "int8", {"no": 0, "yes": 1})
            self.assertTrue(bool(f))

    def test_get_spans(self):
        vals = np.asarray([0, 1, 1, 3, 3, 6, 5, 5, 5], dtype=np.int32)
        bio = BytesIO()
        with session.Session() as s:
            self.assertListEqual([0, 1, 3, 5, 6, 9], s.get_spans(vals).tolist())

            ds = s.open_dataset(bio, "w", "ds")
            vals_f = s.create_numeric(ds, "vals", "int32")
            vals_f.data.write(vals)
            self.assertListEqual([0, 1, 3, 5, 6, 9], vals_f.get_spans().tolist())



class TestIndexedStringFields(unittest.TestCase):

    def test_create_indexed_string(self):
        bio = BytesIO()
        with h5py.File(bio, 'r+') as hf:
            s = session.Session()
            strings = ['a', 'bb', 'ccc', 'dddd']
            f = fields.IndexedStringImporter(s, hf, 'foo')
            f.write(strings)
            f = s.get(hf['foo'])
            # f = s.create_indexed_string(hf, 'foo')
            self.assertListEqual([0, 1, 3, 6, 10], f.indices[:].tolist())

            f2 = s.create_indexed_string(hf, 'bar')
            s.apply_filter(np.asarray([False, True, True, False]), f, f2)
            # print(f2.indices[:])
            self.assertListEqual([0, 2, 5], f2.indices[:].tolist())
            # print(f2.values[:])
            self.assertListEqual([98, 98, 99, 99, 99], f2.values[:].tolist())
            # print(f2.data[:])
            self.assertListEqual(['bb', 'ccc'], f2.data[:])
            # print(f2.data[0])
            self.assertEqual('bb', f2.data[0])
            # print(f2.data[1])
            self.assertEqual('ccc', f2.data[1])


    def test_update_legacy_indexed_string_that_has_uint_values(self):
        bio = BytesIO()
        with h5py.File(bio, 'r+') as hf:
            s = session.Session()
            strings = ['a', 'bb', 'ccc', 'dddd']
            f = fields.IndexedStringImporter(s, hf, 'foo')
            f.write(strings)
            values = hf['foo']['values'][:]
            self.assertListEqual([97, 98, 98, 99, 99, 99, 100, 100, 100, 100], values.tolist())



class TestFieldArray(unittest.TestCase):
    def test_write_part(self):
        bio = BytesIO()
        s = session.Session()
        dst = s.open_dataset(bio, 'w', 'dst')
        num = s.create_numeric(dst, 'num', 'int32')
        num.data.write_part(np.arange(10))
        self.assertListEqual([0,1,2,3,4,5,6,7,8,9],list(num.data[:]))

    def test_clear(self):
        bio = BytesIO()
        s = session.Session()
        dst = s.open_dataset(bio, 'w', 'dst')
        num = s.create_numeric(dst, 'num', 'int32')
        num.data.write_part(np.arange(10))
        num.data.clear()
        self.assertListEqual([], list(num.data[:]))


