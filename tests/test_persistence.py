import unittest
import random
from datetime import datetime, timezone, timedelta
import time
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

    def test_chunks(self):
        for c in persistence.chunks(1050, 100):
            print(c)
        for c in persistence.chunks(1000, 100):
            print(c)

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
        values = ['True', 'False', '', '', 'False', '', 'True',
                  'Stupendous', '', "I really don't know", 'True',
                  'Ambiguous', '', '', '', 'Things', 'Zombie driver',
                  'Perspicacious', 'False', 'Fa,lse', '', '', 'True',
                  '', 'True', 'Troubador', '', 'Calisthenics', 'The',
                  '', 'Quick', 'Brown', '', '', 'Fox', 'Jumped', '',
                  'Over', 'The', '', 'Lazy', 'Dog']
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.NewIndexedStringWriter(hf, 10, 'foo', ts)
            foo.write_part(values[0:10])
            foo.write_part(values[10:20])
            foo.write_part(values[20:30])
            foo.write_part(values[30:40])
            foo.write_part(values[40:42])
            foo.flush()
            print(hf['foo']['index'][()])

            index = hf['foo']['index'][()]

            actual = list()
            for i in range(index.size - 1):
                actual.append(hf['foo']['values'][index[i]:index[i+1]].tobytes().decode())

            self.assertListEqual(values, actual)

        with h5py.File(bio, 'r') as hf:
            foo = persistence.NewIndexedStringReader(hf['foo'])
            print(foo[:])

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

    def test_indexed_string_writer_from_reader(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['True', 'False', '', '', 'False', '', 'True',
                  'Stupendous', '', "I really don't know", 'True',
                  'Ambiguous', '', '', '', 'Things', 'Zombie driver',
                  'Perspicacious', 'False', 'Fa,lse', '', '', 'True',
                  '', 'True', 'Troubador', '', 'Calisthenics', 'The',
                  '', 'Quick', 'Brown', '', '', 'Fox', 'Jumped', '',
                  'Over', 'The', '', 'Lazy', 'Dog']
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            persistence.NewIndexedStringWriter(hf, 10, 'foo', ts).write(values)

            reader = persistence.get_reader_from_field(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = persistence.get_reader_from_field(hf['foo2'])
            self.assertListEqual(reader[:], reader2[:])

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


    def test_new_fixed_string_importer_1(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.NewFixedStringWriter(hf, 10, 'foo', ts, 5)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            foo.write_part(np.asarray(values[0:10], dtype="S5"))
            foo.write_part(np.asarray(values[10:20], dtype="S5"))
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

    def test_fixed_string_writer_from_reader(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                  '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
        bvalues = [v.encode() for v in values]
        with h5py.File(bio, 'w') as hf:
            persistence.NewFixedStringWriter(hf, 10, 'foo', ts, 6).write(bvalues)

            reader = persistence.get_reader_from_field(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = persistence.get_reader_from_field(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))

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


    def test_new_numeric_reader_float32(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                  '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
        with h5py.File(bio, 'w') as hf:
            foov = persistence.NewNumericWriter(hf, 10, 'foo', ts, 'float32')
            foof = persistence.NewNumericWriter(hf, 10, 'foo_filter', ts, 'bool')
            out_values = np.zeros(len(values), dtype=np.float32)
            out_filter = np.zeros(len(values), dtype=np.bool)
            for i in range(len(values)):
                try:
                    out_values[i] = float(values[i])
                    out_filter[i] = True
                except:
                    out_values[i] = 0
                    out_filter[i] = False
            foov.write_part(out_values)
            foof.write_part(out_filter)
            foov.flush()
            foof.flush()

        with h5py.File(bio, 'r') as hf:
            foov = persistence.NewNumericReader(hf['foo'])
            foof = persistence.NewNumericReader(hf['foo_filter'])
            print(foov[5:15])
            print(foof[5:15])


    def test_new_numeric_writer_from_reader(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                  '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
        with h5py.File(bio, 'w') as hf:
            out_values = np.zeros(len(values), dtype=np.float32)
            out_filter = np.zeros(len(values), dtype=np.bool)
            for i in range(len(values)):
                try:
                    out_values[i] = float(values[i])
                    out_filter[i] = True
                except:
                    out_values[i] = 0
                    out_filter[i] = False
            persistence.NewNumericWriter(hf, 10, 'foo', ts, 'float32').write(out_values)
            persistence.NewNumericWriter(hf, 10, 'foo_filter', ts, 'bool').write(out_filter)

            reader = persistence.get_reader_from_field(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = persistence.get_reader_from_field(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))


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

    def test_new_numeric_importer_int32(self):

        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      0, 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NewNumericImporter(hf, 10, 'foo', ts, 'int32',
                                                 persistence.try_str_to_float_to_int)
            foo.write_part(values[0:10])
            foo.write_part(values[10:20])
            foo.write_part(values[20:22])
            foo.flush()
            # for f in persistence.numeric_iterator(hf['foo']):
            #     print(f[0], f[1])

        expected = np.asarray([0, 0, 2, 3, 40, 0, 0, -6, -7, -80, 0,
                               0, 0, 2, 3, 40, 0, 0, -6, -7, -80, 0], dtype=np.int32)
        expected_valid =\
            np.asarray([False, False, True, True, True, True, False, True, True, True, True,
                        True, False, True, True, True, True, False, True, True, True, True],
                       dtype=np.bool)
        with h5py.File(bio, 'r') as hf:
            foo = persistence.NewNumericReader(hf['foo'])
            foo_valid = persistence.NewNumericReader(hf['foo_valid'])
            self.assertTrue(np.alltrue(foo[:] == expected))
            self.assertTrue(np.alltrue(foo_valid[:] == expected_valid))


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
            foo = persistence.CategoricalWriter(hf, 10, 'foo', ts, {'': 0, 'False': 1, 'True': 2})
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
            for expected, actual in zip([value_map[v] for v in values], foo_int):
                self.assertEqual(expected, actual)


    def test_categorical_field_writer_from_reader(self):
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '']
            value_map = {'': 0, 'False': 1, 'True': 2}
            persistence.NewCategoricalImporter(hf, 10, 'foo', ts, value_map).write(values)

            reader = persistence.get_reader_from_field(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = persistence.get_reader_from_field(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))


    def test_timestamp_reader(self):
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        random.seed(12345678)
        deltas = [random.randint(10, 1000) for _ in range(95)]
        values = [dt + timedelta(seconds=d) for d in deltas]

        with h5py.File(bio, 'w') as hf:
            foo = persistence.DatetimeWriter(hf, 10, 'foo', ts)
            for v in values:
                foo.append(str(v))
            foo.flush()

        with h5py.File(bio, 'r') as hf:
            foo = persistence.TimestampReader(hf['foo'])
            actual = [f for f in foo]
            self.assertEqual([v.timestamp() for v in values], actual)


    def test_new_timestamp_reader(self):
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        random.seed(12345678)
        deltas = [random.randint(10, 1000) for _ in range(95)]
        values = [dt + timedelta(seconds=d) for d in deltas]
        svalues = [str(v).encode() for v in values]
        tsvalues = np.asarray([v.timestamp() for v in values], dtype=np.float64)

        with h5py.File(bio, 'w') as hf:
            foo = persistence.NewDateTimeWriter(hf, 10, 'foo', ts)
            foo.write_part(svalues[0:20])
            foo.write_part(svalues[20:40])
            foo.write_part(svalues[40:60])
            foo.write_part(svalues[60:80])
            foo.write_part(svalues[80:95])
            foo.flush()

        with h5py.File(bio, 'r') as hf:
            foo = persistence.NewTimestampReader(hf['foo'])
            actual = foo[:]
            self.assertTrue(np.alltrue(tsvalues == actual))


    def test_new_timestamp_writer_from_reader(self):
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        random.seed(12345678)
        deltas = [random.randint(10, 1000) for _ in range(95)]
        values = [dt + timedelta(seconds=d) for d in deltas]
        svalues = [str(v).encode() for v in values]
        tsvalues = np.asarray([v.timestamp() for v in values], dtype=np.float64)

        with h5py.File(bio, 'w') as hf:
            persistence.NewDateTimeWriter(hf, 10, 'foo', ts).write(svalues)

            reader = persistence.get_reader_from_field(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = persistence.get_reader_from_field(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))

    def test_np_filtered_iterator(self):
        values = np.asarray([1.0, 0.0, 2.1], dtype=np.float32)
        filter = np.asarray([True, False, True], dtype=np.bool)
        for v in persistence.filtered_iterator(values, filter):
            print(v)


    def test_predicate(self):
        values = np.random.randint(low=0, high=1000, size=95, dtype=np.uint32)

        def functor(foo, footwo):
            #TODO: handle the output being bigger than the final input
            footwo[:] = foo * 2

        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            foo = persistence.NewNumericWriter(hf, 10, 'foo', ts, 'uint32')
            foo.write_part(values)
            foo.flush()

        with h5py.File(bio, 'w') as hf:
            footwo = persistence.NewNumericWriter(hf, 10, 'twofoo', ts, 'uint32')
            foo = persistence.NewNumericReader(hf['foo'])

            persistence.process({'foo': foo}, {'footwo': footwo}, functor)


    def test_raw_performance(self):
        import time

        testsize = 1 << 20
        a = np.random.randint(low=0, high=1000, size=testsize, dtype=np.uint32)
        b = np.random.randint(low=0, high=1000, size=testsize, dtype=np.uint32)

        t0 = time.time()
        c = a + b
        print(time.time() - t0)
        print(np.sum(c))


    def test_sort(self):

        vx = [b'a', b'b', b'c', b'd', b'e']
        va = [1, 2, 2, 1, 1]
        vb = [5, 4, 3, 2, 1]
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            fva = persistence.NewNumericWriter(hf, 10, 'va', ts, 'uint32')
            fva.write(va)
            fvb = persistence.NewNumericWriter(hf, 10, 'vb', ts, 'uint32')
            fvb.write(vb)
            fvx = persistence.NewFixedStringWriter(hf, 10, 'vx', ts, 1)
            fvx.write(vx)

            rva = persistence.NewNumericReader(hf['va'])
            rvb = persistence.NewNumericReader(hf['vb'])
            rvx = persistence.NewFixedStringReader(hf['vx'])
            sindex = persistence.dataset_sort(np.arange(5, dtype='uint32'), (rva, rvb))

            ava = persistence.apply_sort_to_array(sindex, rva[:])
            avb = persistence.apply_sort_to_array(sindex, rvb[:])
            avx = persistence.apply_sort_to_array(sindex, rvx[:])

            self.assertListEqual([1, 1, 1, 2, 2], ava.tolist())
            self.assertListEqual([1, 2, 5, 3, 4], avb.tolist())
            self.assertListEqual([b'e', b'd', b'a', b'c', b'b'], avx.tolist())

        # sindex = np.argsort(vb, kind='stable')
        # #sindex = sorted(np.arange(len(vb)), key=lambda x: vb[x])
        # print(np.asarray(va)[sindex])
        # print(np.asarray(vb)[sindex])
        # print(np.asarray(vx)[sindex])
        # accindex = np.asarray(sindex)
        # sva = np.asarray(va)[sindex]
        # sindex = np.argsort(sva, kind='stable')
        # #sindex = np.asarray(sorted(np.arange(len(va)), key=lambda x: sva[x]))
        # accindex = accindex[sindex]
        # print(accindex)
        # print(np.asarray(va)[accindex])
        # print(np.asarray(vb)[accindex])
        # print(np.asarray(vx)[accindex])



class TestSorting(unittest.TestCase):

    def test_sorting_indexed_string(self):
        string_vals = (
            ['a', 'bb', 'ccc', 'dddd', 'eeeee'], ['a', 'bb', 'ccc', 'dddd', 'eeeee'],
            ['', 'a', '', 'bb', '', 'c', ''], ['', 'a', '', 'bb', '', 'c', '']
        )
        sorted_indices = (
            [2, 3, 4, 1, 0], [0, 1, 2, 3, 4],
            [1, 2, 5, 0, 6, 3, 4], [2, 1, 5, 0, 6, 4, 3]
        )
        for sv, si in zip(string_vals, sorted_indices):
            dt = datetime.now(timezone.utc)
            ts = str(dt)
            bio = BytesIO()
            with h5py.File(bio, 'w') as hf:
                persistence.NewIndexedStringWriter(hf, 10, 'vals', ts).write(sv)

                vals = persistence.NewIndexedStringReader(hf['vals'])
                wvals = vals.getwriter(hf, 'sorted_vals', ts)
                vals.sort(np.asarray(si, dtype=np.uint32), wvals)
                actual = persistence.NewIndexedStringReader(hf['sorted_vals'])[:]
                expected = [sv[i] for i in si]
                self.assertListEqual(expected, actual)


class TestJittingSort(unittest.TestCase):

    def test_jitting_sort(self):
        from numba import jit
        @jit
        def predicate(i):
            return values[i]

        count = 5000000
        values = np.random.seed(12345678)
        values = np.random.rand(count)
        index = np.arange(count, dtype=np.uint32)
        t0 = time.time()
        s_index = sorted(index, key=lambda x: values[x])
        print(f"sorted in {time.time() - t0}s")

        index = np.arange(count, dtype=np.uint32)
        t0 = time.time()
        s_index = sorted(index, key=predicate)
        print(f"sorted in {time.time() - t0}s")

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