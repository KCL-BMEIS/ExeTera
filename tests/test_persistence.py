import unittest
import random
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import h5py

import persistence

class TestPersistence(unittest.TestCase):


    def test_slice_for_chunk(self):
        dataset = np.zeros(1050)
        expected = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
                    (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000), (1000, 1050)]
        for i in range(11):
            self.assertEqual(expected[i], persistence._slice_for_chunk(i, dataset, 100))
        expected = [(200, 300), (300, 400), (400, 500), (500, 600), (600, 700), (700, 800),
                    (800, 850)]
        for i in range(7):
            self.assertEqual(expected[i], persistence._slice_for_chunk(i, dataset, 100, 200, 850))


    def test_cached_array_read(self):
        dataset = np.arange(95, dtype=np.uint32)
        arr = persistence.Series(dataset, 10)
        print("length:", len(arr))
        print("chunksize:", arr.chunksize())
        for i in range(dataset.size):
            self.assertEqual(i, arr[i])


    def test_cached_array_write(self):
        dataset = np.zeros(95, dtype=np.uint32)
        arr = persistence.Series(dataset, 10)
        for i in range(dataset.size):
            arr[i] = i
        for i in range(dataset.size):
            self.assertEqual(i, arr[i])


    def test_filter(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            htest = hf.create_group('test')
            random.seed(12345678)
            entries = ['', '', '', 'a', 'b']
            values = [entries[random.randint(0, 4)] for _ in range(95)]
            foo = persistence.FixedStringWriter(htest, 10, 'foo', ts, 1)
            for v in values:
                foo.append(v)
            foo.flush()
            results = persistence.filter(
                htest, htest['foo']['values'], 'non_empty_foo', lambda x: len(x) == 0, ts)
            actual = results['values'][()]
            for i in range(len(values)):
                self.assertEqual(values[i] == '', actual[i])


    def test_distinct(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            htest = hf.create_group('test')
            random.seed(12345678)
            entries = ['', '', '', 'a', 'b']
            values = [entries[random.randint(0, 4)] for _ in range(95)]
            print(values)

            foo = persistence.FixedStringWriter(htest, 10, 'foo', ts, 1)
            for v in values:
                foo.append(v)
            foo.flush()

            non_empty = persistence.filter(
                htest, htest['foo']['values'], 'non_empty_foo', lambda x: len(x) == 0, ts)
            results = persistence.distinct(
                htest, htest['foo']['values'], 'distinct_foo')
            print(results)
            results = persistence.distinct(
                htest, htest['foo']['values'], 'distinct_foo', filter=non_empty['values'])
            print(results)


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
                print(hf['foo']['values'][index[i]:index[i+1]].tostring().decode())
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

            for f in persistence.fixed_string_iterator(hf['foo']):
                print(f)


    def test_fixed_string_reader(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                  '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
        bvalues = [v.encode() for v in values]
        with h5py.File(bio, 'w') as hf:
            foo = persistence.FixedStringWriter(hf, 10, 'foo', ts, 6)
            for v in bvalues:
                foo.append(v)
            foo.flush()

        with h5py.File(bio, 'r') as hf:
            r_bytes = persistence.FixedStringReader(hf['foo'])
            r_strs = persistence.FixedStringReader(hf['foo'], as_string=True)
            for i in range(len(r_bytes)):
                self.assertEqual(bvalues[i], r_bytes[i])
                self.assertEqual(values[i], r_strs[i])


    def test_numeric_importer_bool(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = [True, False, None, False, True, None, None, False, True, False,
                  True, False, None, False, True, None, None, False, True, False,
                  True, False, None, False, True]
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'bool')


    def test_numeric_importer_float32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'float32',
                                            persistence.str_to_float, True)
            for v in values:
                foo.append(v)
            foo.flush()

            for f in persistence.numeric_iterator(hf['foo']):
                print(f[0], f[1])


    def test_numeric_reader_float32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                  '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
        with h5py.File(bio, 'w') as hf:
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'float32',
                                            persistence.str_to_float, True)
            for v in values:
                foo.append(v)
            foo.flush()

        with h5py.File(bio, 'r') as hf:
            foo = persistence.NumericReader(hf['foo'])
            for v in range(len(foo)):
                print(foo[v])


    def test_numeric_importer_int32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'int32',
                                            persistence.str_to_int, True)
            for v in values:
                foo.append(v)
            foo.flush()

            for f in persistence.numeric_iterator(hf['foo']):
                print(f[0], f[1])


    def test_numeric_importer_uint32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericWriter(hf, 10, 'foo', ts, 'uint32',
                                            persistence.str_to_int, True)
            for v in values:
                foo.append(v)
            foo.flush()

            for f in persistence.numeric_iterator(hf['foo']):
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


    def test_categorical_string_reader(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '']
            value_map = {'': 0, 'False': 1, 'True': 2}
            foo = persistence.CategoricalWriter(hf, 10, 'foo', ts, value_map)
            for v in values:
                foo.append(v)
            foo.flush()

        with h5py.File(bio, 'r') as hf:
            foo_int = persistence.CategoricalReader(hf['foo'], False)
            foo_str = persistence.CategoricalReader(hf['foo'], True)
            for i in range(len(foo_int)):
                self.assertEqual(values[i], foo_str[i])
                self.assertEqual(value_map[values[i]], foo_int[i])


class TestLongPersistence(unittest.TestCase):

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
