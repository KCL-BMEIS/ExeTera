import unittest
import random
from datetime import datetime, timezone, timedelta
import time
from io import BytesIO

import numpy as np
import h5py

from core import persistence


class TestPersistence(unittest.TestCase):


    # def test_slice_for_chunk(self):
    #     dataset = np.zeros(1050)
    #     expected = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
    #                 (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000), (1000, 1050)]
    #     for i in range(11):
    #         self.assertEqual(expected[i], persistence._slice_for_chunk(i, dataset, 100))
    #     expected = [(200, 300), (300, 400), (400, 500), (500, 600), (600, 700), (700, 800),
    #                 (800, 850)]
    #     for i in range(7):
    #         self.assertEqual(expected[i], persistence._slice_for_chunk(i, dataset, 100, 200, 850))

    # def test_chunks(self):
    #     for c in persistence.chunks(1050, 100):
    #         print(c)
    #     for c in persistence.chunks(1000, 100):
    #         print(c)


    # TODO: reintroduce once filter is reinstated
    # def test_filter(self):
    #     ts = str(datetime.now(timezone.utc))
    #     bio = BytesIO()
    #     with h5py.File(bio, 'w') as hf:
    #         htest = hf.create_group('test')
    #         random.seed(12345678)
    #         entries = ['', '', '', 'a', 'b']
    #         values = [entries[random.randint(0, 4)] for _ in range(95)]
    #         foo = persistence.FixedStringWriter(htest, 10, 'foo', ts, 1)
    #         for v in values:
    #             foo.append(v)
    #         foo.flush()
    #         results = persistence.filter(
    #             htest, htest['foo']['values'], 'non_empty_foo', lambda x: len(x) == 0, ts)
    #         actual = results['values'][()]
    #         for i in range(len(values)):
    #             self.assertEqual(values[i] == '', actual[i])


    # TODO: reintroduce once distinct is reinstated
    def test_distinct(self):
        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            htest = hf.create_group('test')
            random.seed(12345678)
            entries = [b'', b'', b'', b'a', b'b']
            values = [entries[random.randint(0, 4)] for _ in range(95)]
            print(values)

            persistence.FixedStringWriter(datastore, htest, 'foo', ts, 1).write(values)

            # non_empty = datastore.filter(
            #     htest, htest['foo']['values'], 'non_empty_foo', lambda x: len(x) == 0, ts)
            results = datastore.distinct(
                htest, htest['foo']['values'], 'distinct_foo')
            print(results)
            # results = persistence.distinct(
            #     htest, htest['foo']['values'], 'distinct_foo', filter=non_empty['values'])
            # print(results)


    def test_indexed_string_importer(self):

        datastore = persistence.DataStore(10)
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

            foo = persistence.IndexedStringWriter(datastore, hf, 'foo', ts)
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
            foo = persistence.IndexedStringReader(datastore, hf['foo'])
            print(foo[:])

    def test_indexed_string_importer_2(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.IndexedStringWriter(datastore, hf, 'foo', ts)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            foo.write(values)
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            index = hf['foo']['index'][()]
            actual = list()
            for i in range(index.size - 1):
                # print(index[i], index[i+1])
                print(hf['foo']['values'][index[i]:index[i+1]].tobytes().decode())
                actual.append(hf['foo']['values'][index[i]:index[i+1]].tobytes().decode())

    def test_indexed_string_writer_from_reader(self):

        datastore = persistence.DataStore(10)
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

            persistence.IndexedStringWriter(datastore, hf, 'foo', ts).write(values)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertListEqual(reader[:], reader2[:])

    def test_fixed_string_importer_1(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.FixedStringWriter(datastore, hf, 'foo', ts, 5)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            bvalues = [v.encode() for v in values]
            foo.write(bvalues)
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            print(persistence.FixedStringReader(datastore, hf['foo'])[:])


    def test_new_fixed_string_importer_1(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = persistence.FixedStringWriter(datastore, hf, 'foo', ts, 5)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            foo.write_part(np.asarray(values[0:10], dtype="S5"))
            foo.write_part(np.asarray(values[10:20], dtype="S5"))
            foo.flush()
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            print(persistence.FixedStringReader(datastore, hf['foo']))


    def test_fixed_string_reader(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                  '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
        bvalues = [v.encode() for v in values]
        with h5py.File(bio, 'w') as hf:
            foo = persistence.FixedStringWriter(datastore, hf, 'foo', ts, 6)
            foo.write(bvalues)

        with h5py.File(bio, 'r') as hf:
            r_bytes = persistence.FixedStringReader(datastore, hf['foo'])[:]
            for i in range(len(r_bytes)):
                self.assertEqual(bvalues[i], r_bytes[i])


    def test_fixed_string_writer_from_reader(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                  '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
        bvalues = [v.encode() for v in values]
        with h5py.File(bio, 'w') as hf:
            persistence.FixedStringWriter(datastore, hf, 'foo', ts, 6).write(bvalues)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))

    def test_numeric_importer_bool(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = [True, False, None, False, True, None, None, False, True, False,
                  True, False, None, False, True, None, None, False, True, False,
                  True, False, None, False, True]
        arrvalues = np.asarray(values, dtype='bool')
        print(type(values[0]))
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')
            foo = persistence.NumericWriter(datastore, hf, 'foo', ts, 'bool').write(arrvalues)

            foo = persistence.NumericReader(datastore, hf['foo'])[:]
            self.assertTrue(np.array_equal(arrvalues, foo))


    def test_numeric_importer_float32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericImporter(datastore, hf, 'foo', ts, 'float32',
                                              persistence.try_str_to_float)
            foo.write(values)

            print(persistence.NumericReader(datastore, hf['foo'])[:])
            print(persistence.NumericReader(datastore, hf['foo_valid'])[:])


    def test_numeric_reader_float32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                  '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
        with h5py.File(bio, 'w') as hf:
            foo = persistence.NumericImporter(datastore, hf, 'foo', ts, 'float32',
                                              persistence.try_str_to_float)

            foo.write(values)

        with h5py.File(bio, 'r') as hf:
            foo = persistence.NumericReader(datastore, hf['foo'])
            print(foo[:])


    def test_new_numeric_reader_float32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                  '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
        with h5py.File(bio, 'w') as hf:
            foov = persistence.NumericWriter(datastore, hf, 'foo', ts, 'float32')
            foof = persistence.NumericWriter(datastore, hf, 'foo_filter', ts, 'bool')
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
            foov = persistence.NumericReader(datastore, hf['foo'])
            foof = persistence.NumericReader(datastore, hf['foo_filter'])
            print(foov[5:15])
            print(foof[5:15])


    def test_new_numeric_writer_from_reader(self):

        datastore = persistence.DataStore(10)
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
            persistence.NumericWriter(datastore, hf, 'foo', ts, 'float32').write(out_values)
            persistence.NumericWriter(datastore, hf, 'foo_filter', ts, 'bool').write(out_filter)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))


    def test_numeric_importer_int32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericImporter(datastore, hf, 'foo', ts, 'int32',
                                              persistence.try_str_to_int).write(values)

            print(list(zip(persistence.NumericReader(datastore, hf['foo'])[:],
                           persistence.NumericReader(datastore, hf['foo_valid'])[:])))


    def test_new_numeric_importer_int32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      0, 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericImporter(datastore, hf, 'foo', ts, 'int32',
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
            foo = persistence.NumericReader(datastore, hf['foo'])
            foo_valid = persistence.NumericReader(datastore, hf['foo_valid'])
            self.assertTrue(np.alltrue(foo[:] == expected))
            self.assertTrue(np.alltrue(foo_valid[:] == expected_valid))


    def test_numeric_importer_uint32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = persistence.NumericImporter(datastore, hf, 'foo', ts, 'uint32',
                                              persistence.try_str_to_int).write(values)

            print(list(zip(persistence.NumericReader(datastore, hf['foo'])[:],
                           persistence.NumericReader(datastore, hf['foo_valid'])[:])))


    def test_categorical_string_importer(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '']
            # ds = hf.create_dataset('foo', (10,), dtype=h5py.string_dtype(encoding='utf-8'))
            # ds[:] = values
            # print(ds)
            foo = persistence.CategoricalImporter(datastore, hf, 'foo', ts,
                                                  {'': 0, 'False': 1, 'True': 2})
            foo.write(values)
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

        with h5py.File(bio, 'r') as hf:
            print(hf['foo'].keys())
            print(hf['foo']['values'])


    def test_categorical_string_reader(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '']
            value_map = {'': 0, 'False': 1, 'True': 2}
            foo = persistence.CategoricalImporter(datastore, hf, 'foo', ts, value_map)
            foo.write(values)

        with h5py.File(bio, 'r') as hf:
            foo_int = persistence.CategoricalReader(datastore, hf['foo'])
            for i in range(len(foo_int)):
                self.assertEqual(value_map[values[i]], foo_int[i])
            for expected, actual in zip([value_map[v] for v in values], foo_int):
                self.assertEqual(expected, actual)


    def test_categorical_field_writer_from_reader(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '']
            value_map = {'': 0, 'False': 1, 'True': 2}
            persistence.CategoricalImporter(datastore, hf, 'foo', ts, value_map).write(values)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))


    def test_timestamp_reader(self):

        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        random.seed(12345678)
        deltas = [random.randint(10, 1000) for _ in range(95)]
        values = [dt + timedelta(seconds=d) for d in deltas]

        with h5py.File(bio, 'w') as hf:
            foo = persistence.DateTimeWriter(datastore, hf, 'foo', ts)
            # for v in values:
            #     foo.append(str(v))
            # foo.flush()
            foo.write([str(v).encode() for v in values])

        with h5py.File(bio, 'r') as hf:
            foo = persistence.TimestampReader(datastore, hf['foo'])
            actual = [f for f in foo[:]]
            self.assertEqual([v.timestamp() for v in values], actual)


    def test_new_timestamp_reader(self):

        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        random.seed(12345678)
        deltas = [random.randint(10, 1000) for _ in range(95)]
        values = [dt + timedelta(seconds=d) for d in deltas]
        svalues = [str(v).encode() for v in values]
        tsvalues = np.asarray([v.timestamp() for v in values], dtype=np.float64)

        with h5py.File(bio, 'w') as hf:
            foo = persistence.DateTimeWriter(datastore, hf, 'foo', ts)
            foo.write_part(svalues[0:20])
            foo.write_part(svalues[20:40])
            foo.write_part(svalues[40:60])
            foo.write_part(svalues[60:80])
            foo.write_part(svalues[80:95])
            foo.flush()

        with h5py.File(bio, 'r') as hf:
            foo = persistence.TimestampReader(datastore, hf['foo'])
            actual = foo[:]
            self.assertTrue(np.alltrue(tsvalues == actual))


    def test_new_timestamp_writer_from_reader(self):

        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        random.seed(12345678)
        deltas = [random.randint(10, 1000) for _ in range(95)]
        values = [dt + timedelta(seconds=d) for d in deltas]
        svalues = [str(v).encode() for v in values]
        tsvalues = np.asarray([v.timestamp() for v in values], dtype=np.float64)

        with h5py.File(bio, 'w') as hf:
            persistence.DateTimeWriter(datastore, hf, 'foo', ts).write(svalues)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.getwriter(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))

    def test_np_filtered_iterator(self):

        datastore = persistence.DataStore(10)
        values = np.asarray([1.0, 0.0, 2.1], dtype=np.float32)
        filter = np.asarray([True, False, True], dtype=np.bool)
        for v in persistence.filtered_iterator(values, filter):
            print(v)


class TestPersistenceConcat(unittest.TestCase):

    def test_apply_spans_concat_fast(self):

        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()

        src_spans = np.asarray([0, 2, 3, 4, 6, 8], dtype=np.int64)
        src_indices = np.asarray([0, 2, 6, 10, 12, 16, 18, 22, 24], dtype=np.int64)
        src_values = np.frombuffer(b'aabbbbccccddeeeeffgggghh', dtype='S1')

        with h5py.File(bio, 'w') as hf:
            foo = persistence.IndexedStringWriter(datastore, hf, 'foo', ts)
            foo.write_raw(src_indices, src_values)
            foo_r = datastore.get_reader(hf['foo'])
            datastore.apply_spans_concat(src_spans, foo_r, foo_r.getwriter(hf, 'concatfoo', ts))

            expected = ['aabbbb', 'cccc', 'dd', 'eeeeff', 'gggghh']
            actual = datastore.get_reader(hf['concatfoo'])[:]
            self.assertListEqual(expected, actual)


    def test_apply_spans_concat_fast_value_flush_length_is_0(self):

        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()

        src_spans = np.asarray([0, 2, 3, 4, 6, 8], dtype=np.int64)
        src_indices = np.asarray([0, 12, 20, 32, 40, 44, 54, 57, 64], dtype=np.int64)
        src_values = np.frombuffer(
            b'aaaaaaaaaaaabbbbbbbbccccccccccccddddddddeeeeffffffffffggggghhhhh', dtype='S1')
        with h5py.File(bio, 'w') as hf:
            foo = persistence.IndexedStringWriter(datastore, hf, 'foo', ts)
            foo.write_raw(src_indices, src_values)
            foo_r = datastore.get_reader(hf['foo'])
            datastore.apply_spans_concat(src_spans, foo_r, foo_r.getwriter(hf, 'concatfoo', ts))

            expected = ['aaaaaaaaaaaabbbbbbbb', 'cccccccccccc', 'dddddddd',
                        'eeeeffffffffff', 'ggggghhhhh']
            actual = datastore.get_reader(hf['concatfoo'])[:]
            self.assertListEqual(expected, actual)


    def test_apply_spans_concat_fast_value_multiple_iterations(self):

        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()

        src_spans = np.asarray([0, 2, 3, 4, 6, 8, 9], dtype=np.int64)
        src_indices = np.asarray([0, 12, 20, 32, 40, 44, 54, 57, 64, 72], dtype=np.int64)
        src_values = np.frombuffer(
            b'aaaaaaaaaaaabbbbbbbbccccccccccccddddddddeeeeffffffffffggggghhhhhiiiiiiii', dtype='S1')
        with h5py.File(bio, 'w') as hf:
            foo = persistence.IndexedStringWriter(datastore, hf, 'foo', ts)
            foo.write_raw(src_indices, src_values)
            foo_r = datastore.get_reader(hf['foo'])
            datastore.apply_spans_concat(src_spans, foo_r, foo_r.getwriter(hf, 'concatfoo', ts))

            expected = ['aaaaaaaaaaaabbbbbbbb', 'cccccccccccc', 'dddddddd',
                        'eeeeffffffffff', 'ggggghhhhh','iiiiiiii']
            actual = datastore.get_reader(hf['concatfoo'])[:]
            self.assertListEqual(expected, actual)


class TestPersistanceMiscellaneous(unittest.TestCase):

    def test_distinct_multi_field(self):
        datastore = persistence.DataStore(10)
        a = np.asarray([1, 2, 1, 1, 2, 2, 1, 3, 2, 1])
        b = np.asarray(['a', 'a', 'b', 'a', 'b', 'a', 'd', 'c', 'a', 'b'])
        print(datastore.distinct(fields=(a, b)))

    def test_get_spans_single_field(self):
        datastore = persistence.DataStore(10)
        a = np.asarray([1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1])
        print(datastore.get_spans(field=a))
        a = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 1])
        print(datastore.get_spans(field=a))
        a = np.asarray([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1])
        print(datastore.get_spans(field=a))
        a = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        print(datastore.get_spans(field=a))
        a = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
        print(datastore.get_spans(field=a))

    def test_apply_spans_count(self):
        spans = np.asarray([0, 1, 3, 4, 7, 8, 12, 14])
        results = np.zeros(len(spans)-1, dtype=np.int64)
        persistence._apply_spans_count(spans, results)
        print(results)

    def test_apply_spans_first(self):
        spans = np.asarray([0, 1, 3, 4, 7, 8, 12, 14])
        values = np.arange(14)
        results = np.zeros(len(spans)-1, dtype=np.int64)
        persistence._apply_spans_first(spans, values, results)
        print(results)

    def test_apply_spans_last(self):
        spans = np.asarray([0, 1, 3, 4, 7, 8, 12, 14])
        values = np.arange(14)
        results = np.zeros(len(spans)-1, dtype=np.int64)
        persistence._apply_spans_last(spans, values, results)
        print(results)

        spans = np.asarray([0, 20, 40])
        values = np.arange(40)
        results = np.zeros(len(spans)-1, dtype=np.int64)
        persistence._apply_spans_last(spans, values, results)
        print(results)

    def test_apply_spans_max(self):
        spans = np.asarray([0, 1, 3, 4, 7, 8, 12])
        values = np.asarray([1, 2, 3, 4, 5, 6, 12, 11, 10, 9, 8, 7])
        results = np.zeros(len(spans)-1, dtype=values.dtype)
        persistence._apply_spans_max(spans, values, results)
        print(results)

    def test_write_to_existing(self):
        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        values = np.arange(95)

        with h5py.File(bio, 'w') as hf:
            persistence.NumericWriter(datastore, hf, 'foo', ts, 'int32').write(values)

            reader = persistence.NumericReader(datastore, hf['foo'])
            writer = reader.getwriter(hf, 'foo', ts, 'overwrite')
            writer.write(values * 2)
            reader = persistence.NumericReader(datastore, hf['foo'])
            print(reader[:])


    def test_try_create_group(self):
        datastore = persistence.DataStore(10)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            a = datastore.get_or_create_group(hf, 'a')
            b = datastore.get_or_create_group(hf, 'a')
            self.assertEqual(a, b)


    def test_get_trash_folder(self):
        datastore = persistence.DataStore(10)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            a = hf.create_group('a')
            b = a.create_group('b')
            print(datastore.get_trash_group(b))
            print(datastore.get_trash_group(a))
            print(datastore.get_trash_group(hf))


    def test_move_group(self):
        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            a = hf.create_group('a')
            b = hf.create_group('b')
            x = a.create_dataset('x', data=np.asarray([1, 2, 3, 4, 5]))
            a.move('x', '/b/y')
            print(b['y'].name)
            x = a.create_dataset('x', data=np.asarray([6, 7, 8, 9, 10]))
            try:
                a.move('x', '/b/y')
            except Exception as e:
                print(e)
                print(x[:])
            print(hf['b/y'][:])

        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            trash = hf.create_group('/trash/asmts')
            asmts = hf.create_group('asmts')
            foo = persistence.NumericWriter(datastore, asmts, 'foo', ts, 'int32')
            foo.write(np.arange(95, dtype='int32'))
            trash = datastore.get_trash_group(foo.field)
            hf.move('/asmts/foo', trash.name)
            print(hf['/trash/abcd/asmts/foo'])
        del hf

        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            a = hf.create_group('a')
            b = a.create_group('/b')
            print(hf.keys())


    def test_copy_group(self):
        bio1 = BytesIO()
        with h5py.File(bio1, 'w') as hf1:
            a = hf1.create_group('a')
            b = a.create_group('b')
            c = b.create_dataset('c', data=np.random.randint(low=0, high=10, size=100))
            d = b.create_dataset('d', data=np.random.rand(100))

            bio2 = BytesIO()
            with h5py.File(bio2, 'w') as hf2:
                da = hf2.create_group('a')
                for k in a.keys():
                    da.copy(a[k], da)
                print(da.keys())
                print(da['b'].keys())

    def test_predicate(self):
        datastore = persistence.DataStore(10)
        values = np.random.randint(low=0, high=1000, size=95, dtype=np.uint32)

        def functor(foo, footwo):
            #TODO: handle the output being bigger than the final input
            footwo[:] = foo * 2

        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            foo = persistence.NumericWriter(datastore, hf, 'foo', ts, 'uint32')
            foo.write_part(values)
            foo.flush()

        with h5py.File(bio, 'w') as hf:
            footwo = persistence.NumericWriter(datastore, hf, 'twofoo', ts, 'uint32')
            foo = persistence.NumericReader(datastore, hf['foo'])

            datastore.process({'foo': foo}, {'footwo': footwo}, functor)

            footwo = persistence.NumericReader(datastore, hf['twofoo'])
            for i, j in zip(foo[:], footwo[:]):
                self.assertTrue(j == i*2)


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

        datastore = persistence.DataStore(10)
        vx = [b'a', b'b', b'c', b'd', b'e']
        va = [1, 2, 2, 1, 1]
        vb = [5, 4, 3, 2, 1]
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            fva = persistence.NumericWriter(datastore, hf, 'va', ts, 'uint32')
            fva.write(va)
            fvb = persistence.NumericWriter(datastore, hf, 'vb', ts, 'uint32')
            fvb.write(vb)
            fvx = persistence.FixedStringWriter(datastore, hf, 'vx', ts, 1)
            fvx.write(vx)

            rva = persistence.NumericReader(datastore, hf['va'])
            rvb = persistence.NumericReader(datastore, hf['vb'])
            rvx = persistence.FixedStringReader(datastore, hf['vx'])
            sindex = datastore.dataset_sort(np.arange(5, dtype='uint32'), (rva, rvb))

            ava = persistence._apply_sort_to_array(sindex, rva[:])
            avb = persistence._apply_sort_to_array(sindex, rvb[:])
            avx = persistence._apply_sort_to_array(sindex, rvx[:])

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
        datastore = persistence.DataStore(10)
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
                persistence.IndexedStringWriter(datastore, hf, 'vals', ts).write(sv)

                vals = persistence.IndexedStringReader(datastore, hf['vals'])
                wvals = vals.getwriter(hf, 'sorted_vals', ts)
                vals.sort(np.asarray(si, dtype=np.uint32), wvals)
                actual = persistence.IndexedStringReader(datastore, hf['sorted_vals'])[:]
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
