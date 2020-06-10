import unittest

from datetime import datetime, timezone
from io import BytesIO

import h5py

import persistence

class TestPersistence(unittest.TestCase):

    def test_indexed_string_importer(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.IndexedStringWriter(hf, 10, 'foo', ts)
            values = ['True', 'False', '', '', 'False', '', 'True',
                      'Stupendous', '', "I really don't know", 'True',
                      'Ambiguous', '', '', '', 'Things', 'Zombie driver',
                      'Perspicacious', 'False', 'Fa,lse', '', '', 'True',
                      '', 'True', 'Troubador', '', 'Calisthenics', 'The',
                      '', 'Quick', 'Brown', '', '', 'Fox', 'Jumped', '',
                      'Over', 'The', '', 'Lazy', 'Dog']
            for v in values:
                foo.append(v)
            foo.flush()
            print(hf['foo']['values'][()])
            index = hf['foo']['index'][()]
            print('index:', index)
            print(hf['foo']['values'])
            print(hf['foo']['values'][0:10])
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            actual = list()
            for i in range(index.size - 1):
                # print(index[i], index[i+1])
                # print(hf['foo']['values'][index[i]:index[i+1]].tostring().decode())
                actual.append(hf['foo']['values'][index[i]:index[i+1]].tostring().decode())

            self.assertListEqual(values, actual)

    def test_indexed_string_importer_2(self):

        ts = str(datetime.datetime.now())
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.IndexedStringWriter(hf, 10, 'foo', ts)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            for v in values:
                foo.append(v)
            foo.flush()
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            index = hf['foo']['index'][()]
            actual = list()
            for i in range(index.size - 1):
                # print(index[i], index[i+1])
                # print(hf['foo']['values'][index[i]:index[i+1]].tostring().decode())
                actual.append(hf['foo']['values'][index[i]:index[i+1]].tostring().decode())

    def test_categorical_string_importer(self):

        ts = str(datetime.datetime.now())
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '']
            # ds = hf.create_dataset('foo', (10,), dtype=h5py.string_dtype(encoding='utf-8'))
            # ds[:] = values
            # print(ds)
            foo = persistence.CategoricalWriter(hf, 10, 'foo', {'':0, 'False':1, 'True':2}, ts)
            for v in values:
                foo.append(v)
            foo.flush()
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

        with h5py.File(bio, 'r') as hf:
            print(hf['foo'].keys())
            print(hf['foo']['values'])

