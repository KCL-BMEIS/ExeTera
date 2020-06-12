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

        ts = str(datetime.now(timezone.utc))
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


    def test_fixed_string_importer_1(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.FixedStringWriter(hf, 10, 'foo', ts, 5)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            for v in values:
                foo.append(v)
            foo.flush()
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            for f in persistence.fixed_string_iterator(hf, 'foo'):
                print(f)


    def test_numeric_importer_float32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'float32')
            for v in values:
                foo.append(v)
            foo.flush()

            for f in persistence.numeric_iterator(hf, 'foo'):
                print(f[0], f[1])


    def test_numeric_importer_int32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'int32')
            for v in values:
                foo.append(v)
            foo.flush()

            for f in persistence.numeric_iterator(hf, 'foo'):
                print(f[0], f[1])


    def test_numeric_importer_uint32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'uint32')
            for v in values:
                foo.append(v)
            foo.flush()

            for f in persistence.numeric_iterator(hf, 'foo'):
                print(f[0], f[1])


    def test_categorical_string_importer(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '']
            # ds = hf.create_dataset('foo', (10,), dtype=h5py.string_dtype(encoding='utf-8'))
            # ds[:] = values
            # print(ds)
            foo = persistence.CategoricalWriter(hf, 10, 'foo', ts, {'':0, 'False':1, 'True':2})
            for v in values:
                foo.append(v)
            foo.flush()
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

        with h5py.File(bio, 'r') as hf:
            print(hf['foo'].keys())
            print(hf['foo']['values'])


    def test_large_dataset_chunk_settings(self):
        import time
        import random
        import numpy as np

        with h5py.File('covid_test.hdf5', 'w') as hf:
            random.seed(12345678)
            count = 1000000
            chunk = 100000
            data = np.zeros(count, dtype=np.uint32)
            for i in range(count):
                data[i] = random.randint(0, 1000)
            ds = hf.create_dataset('foo', (count,), chunks=(chunk,), maxshape=(None,), data=data)
            ds2 = hf.create_dataset('foo2', (count,), data=data)

        with h5py.File('covid_test.hdf5', 'r') as hf:

            ds = hf['foo'][()]
            print('foo parse')
            t0 = time.time()
            total = 0
            for d in ds:
                total += d
            print(f"{total} in {time.time() - t0}")

            ds = hf['foo']
            print('foo parse')
            t0 = time.time()
            total = 0
            for d in ds:
                total += d
            print(f"{total} in {time.time() - t0}")

            ds = hf['foo2'][()]
            print('foo parse')
            t0 = time.time()
            total = 0
            for d in ds:
                total += d
            print(f"{total} in {time.time() - t0}")

            ds = hf['foo2']
            print('foo parse')
            t0 = time.time()
            total = 0
            for d in ds:
                total += d
            print(f"{total} in {time.time() - t0}")
