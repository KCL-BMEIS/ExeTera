import unittest

import numpy as np
from io import BytesIO

import h5py

from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per


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
