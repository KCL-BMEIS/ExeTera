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


class TestIndexedStringFields(unittest.TestCase):

    def test_create_indexed_string(self):
        bio = BytesIO()
        with h5py.File(bio) as hf:
            s = session.Session()
            strings = ['a', 'bb', 'ccc', 'dddd']
            f = fields.IndexedStringImporter(s, hf, 'foo')
            f.write(strings)
            f = s.get(hf['foo'])
            # f = s.create_indexed_string(hf, 'foo')
            print("f.indices:", f.indices[:])

            f2 = s.create_indexed_string(hf, 'bar')
            s.apply_filter(np.asarray([False, True, True, False]), f, f2)
            print(f2.indices[:])
            print(f2.values[:])
            print(f2.data[:])
            print(f2.data[0])
            print(f2.data[1])

    def test_update_legacy_indexed_string_that_has_uint_values(self):
        bio = BytesIO()
        with h5py.File(bio) as hf:
            s = session.Session()
            strings = ['a', 'bb', 'ccc', 'dddd']
            f = fields.IndexedStringImporter(s, hf, 'foo')
            f.write(strings)
            values = hf['foo']['values'][:]
            print(values)


