import unittest
import random
from datetime import datetime, timezone, timedelta
import time
from io import BytesIO

import numpy as np
import h5py

from exetera.core import operations, persistence
from exetera.core.session import Session
from exetera.core import readerwriter as rw
from exetera.core import validation as val
from exetera.core import utils


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

            rw.FixedStringWriter(datastore, htest, 'foo', 1, ts).write(values)

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

            foo = rw.IndexedStringWriter(datastore, hf, 'foo', ts)
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
            foo = rw.IndexedStringReader(datastore, hf['foo'])
            print(foo[:])

    def test_indexed_string_importer_2(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = rw.IndexedStringWriter(datastore, hf, 'foo', ts)
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

            rw.IndexedStringWriter(datastore, hf, 'foo', ts).write(values)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.get_writer(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertListEqual(reader[:], reader2[:])

    def test_fixed_string_importer_1(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = rw.FixedStringWriter(datastore, hf, 'foo', 5, ts)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            bvalues = [v.encode() for v in values]
            foo.write(bvalues)
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            print(rw.FixedStringReader(datastore, hf['foo'])[:])


    def test_new_fixed_string_importer_1(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            hf.create_group('test')

            foo = rw.FixedStringWriter(datastore, hf, 'foo', 5, ts)
            values = ['', '', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '1.0.0', '', '',
                      '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '1.0.0', '', '1.0.0', '1.0.0', '']
            foo.write_part(np.asarray(values[0:10], dtype="S5"))
            foo.write_part(np.asarray(values[10:20], dtype="S5"))
            foo.flush()
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

            print(rw.FixedStringReader(datastore, hf['foo']))


    def test_fixed_string_reader(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                  '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
        bvalues = [v.encode() for v in values]
        with h5py.File(bio, 'w') as hf:
            foo = rw.FixedStringWriter(datastore, hf, 'foo', 6, ts)
            foo.write(bvalues)

        with h5py.File(bio, 'r') as hf:
            r_bytes = rw.FixedStringReader(datastore, hf['foo'])[:]
            for i in range(len(r_bytes)):
                self.assertEqual(bvalues[i], r_bytes[i])

    def test_fixed_string_reader_scalar_comparison(self):
        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', '', '1.0.0', '', '1.0.ä', '1.0.0', '1.0.0', '1.0.0', '', '',
                  '1.0.0', '1.0.0', '', '1.0.0', '1.0.ä', '1.0.0', '']
        bvalues = [v.encode() for v in values]
        with h5py.File(bio, 'w') as hf:
            foo = rw.FixedStringWriter(datastore, hf, 'foo', 6, ts)
            foo.write(bvalues)

        with h5py.File(bio, 'r') as hf:
            r_bytes = rw.FixedStringReader(datastore, hf['foo'])[:]
            r_filtered = r_bytes == b'1.0.0'
            print(r_filtered)
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
            rw.FixedStringWriter(datastore, hf, 'foo', 6, ts).write(bvalues)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.get_writer(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))

            raw_reader = reader[:]
            print((raw_reader == b'1.0.0') | (raw_reader == '1.0.ä'.encode()))

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
            foo = rw.NumericWriter(datastore, hf, 'foo', 'bool', ts).write(arrvalues)

            foo = rw.NumericReader(datastore, hf['foo'])[:]
            self.assertTrue(np.array_equal(arrvalues, foo))


    def test_numeric_importer_float32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = rw.NumericImporter(datastore, hf, 'foo', 'float32',
                                              persistence.try_str_to_float, ts)
            foo.write(values)

            print(rw.NumericReader(datastore, hf['foo'])[:])
            print(rw.NumericReader(datastore, hf['foo_valid'])[:])


    def test_numeric_reader_float32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                  '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
        with h5py.File(bio, 'w') as hf:
            foo = rw.NumericImporter(datastore, hf, 'foo', 'float32',
                                              persistence.try_str_to_float, ts)

            foo.write(values)

        with h5py.File(bio, 'r') as hf:
            foo = rw.NumericReader(datastore, hf['foo'])
            print(foo[:])


    def test_new_numeric_reader_float32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                  '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
        with h5py.File(bio, 'w') as hf:
            foov = rw.NumericWriter(datastore, hf, 'foo', 'float32', ts)
            foof = rw.NumericWriter(datastore, hf, 'foo_filter', 'bool', ts)
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
            foov = rw.NumericReader(datastore, hf['foo'])
            foof = rw.NumericReader(datastore, hf['foo_filter'])
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
            rw.NumericWriter(datastore, hf, 'foo', 'float32', ts).write(out_values)
            rw.NumericWriter(datastore, hf, 'foo_filter', 'bool', ts).write(out_filter)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.get_writer(hf, 'foo2', ts)
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
            foo = rw.NumericImporter(datastore, hf, 'foo', 'int32',
                                              persistence.try_str_to_int, ts).write(values)

            print(list(zip(rw.NumericReader(datastore, hf['foo'])[:],
                           rw.NumericReader(datastore, hf['foo_valid'])[:])))


    def test_new_numeric_importer_int32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      0, 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = rw.NumericImporter(datastore, hf, 'foo', 'int32',
                                              persistence.try_str_to_float_to_int, ts)
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
            foo = rw.NumericReader(datastore, hf['foo'])
            foo_valid = rw.NumericReader(datastore, hf['foo_valid'])
            self.assertTrue(np.alltrue(foo[:] == expected))
            self.assertTrue(np.alltrue(foo_valid[:] == expected_valid))


    def test_numeric_importer_uint32(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2',
                      '', 'one', '2', '3.0', '4e1', '5.21e-2', 'foo', '-6', '-7.0', '-8e1', '-9.21e-2']
            foo = rw.NumericImporter(datastore, hf, 'foo', 'uint32',
                                              persistence.try_str_to_int, ts).write(values)

            print(list(zip(rw.NumericReader(datastore, hf['foo'])[:],
                           rw.NumericReader(datastore, hf['foo_valid'])[:])))


    def test_categorical_string_importer(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '']
            # ds = hf.create_dataset('foo', (10,), dtype=h5py.string_dtype(encoding='utf-8'))
            # ds[:] = values
            # print(ds)
            foo = rw.CategoricalImporter(datastore, hf, 'foo',
                                                  {'': 0, 'False': 1, 'True': 2}, ts)
            foo.write(values)
            print('fieldtype:', hf['foo'].attrs['fieldtype'])
            print('timestamp:', hf['foo'].attrs['timestamp'])
            print('completed:', hf['foo'].attrs['completed'])

        with h5py.File(bio, 'r') as hf:
            print(hf['foo'].keys())
            print(hf['foo']['values'])
            print(hf['foo']['key_values'])
            print(hf['foo']['key_names'])


    # def test_categorical_string_writer_with_string_data(self):
    #     datastore = persistence.DataStore(10)
    #     ts = str(datetime.now(timezone.utc))
    #     bio = BytesIO()
    #     with h5py.File(bio, 'w') as hf:
    #         values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '']
    #         # ds = hf.create_dataset('foo', (10,), dtype=h5py.string_dtype(encoding='utf-8'))
    #         # ds[:] = values
    #         # print(ds)
    #         foo = persistence.CategoricalWriter(datastore, hf, 'foo',
    #                                               {'': 0, 'False': 1, 'True': 2}, ts)
    #         foo.write(values)
    #         print('fieldtype:', hf['foo'].attrs['fieldtype'])
    #         print('timestamp:', hf['foo'].attrs['timestamp'])
    #         print('completed:', hf['foo'].attrs['completed'])
    #
    #     with h5py.File(bio, 'r') as hf:
    #         print(hf['foo'].keys())
    #         print(hf['foo']['values'])


    def test_categorical_string_reader(self):

        datastore = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            values = ['', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '', '', 'True', 'False', 'True', '',
                      '', 'True', 'False', 'False', '']
            value_map = {'': 0, 'False': 1, 'True': 2}
            foo = rw.CategoricalImporter(datastore, hf, 'foo', value_map, ts)
            foo.write(values)

        with h5py.File(bio, 'r') as hf:
            foo_int = rw.CategoricalReader(datastore, hf['foo'])
            print(foo_int.keys)
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
            rw.CategoricalImporter(datastore, hf, 'foo', value_map, ts).write(values)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.get_writer(hf, 'foo2', ts)
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
            foo = rw.DateTimeWriter(datastore, hf, 'foo', ts)
            # for v in values:
            #     foo.append(str(v))
            # foo.flush()
            foo.write([str(v).encode() for v in values])

        with h5py.File(bio, 'r') as hf:
            foo = rw.TimestampReader(datastore, hf['foo'])
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
            foo = rw.DateTimeWriter(datastore, hf, 'foo', ts)
            foo.write_part(svalues[0:20])
            foo.write_part(svalues[20:40])
            foo.write_part(svalues[40:60])
            foo.write_part(svalues[60:80])
            foo.write_part(svalues[80:95])
            foo.flush()

        with h5py.File(bio, 'r') as hf:
            foo = rw.TimestampReader(datastore, hf['foo'])
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
            rw.DateTimeWriter(datastore, hf, 'foo', ts).write(svalues)

            reader = datastore.get_reader(hf['foo'])
            writer = reader.get_writer(hf, 'foo2', ts)
            writer.write(reader[:])
            reader2 = datastore.get_reader(hf['foo2'])
            self.assertTrue(np.array_equal(reader[:], reader2[:]))

    def test_np_filtered_iterator(self):

        datastore = persistence.DataStore(10)
        values = np.asarray([1.0, 0.0, 2.1], dtype=np.float32)
        filter = np.asarray([True, False, True], dtype=np.bool)
        for v in persistence.filtered_iterator(values, filter):
            print(v)


    def test_filter_duplicate_fields(self):
        values = ['a', 'b', 'b', 'c', 'd', 'd', 'd', 'e', 'f']
        a = np.asarray(values, dtype='S1')
        f = persistence.filter_duplicate_fields(a)
        print(f)
        ds = persistence.DataStore()
        g = ds.apply_filter(f, a)
        print(g)


class TestPersistenceConcat(unittest.TestCase):

    def test_apply_spans_concat_bug_len_1_entry(self):
        spans = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
        indices = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
        values = np.frombuffer(b'abcd', dtype=np.uint8)
        d_indices = np.zeros(100, dtype=np.int32)
        d_values = np.zeros(100, dtype=np.int32)
        s, ii, iv = 0, 0, 0
        separator = np.frombuffer(b',', dtype='S1')[0][0]
        delimiter = np.frombuffer(b'"', dtype='S1')[0][0]
        while s < len(spans) - 1:
            s, ii, iv = persistence._apply_spans_concat(spans, indices, values,
                                                        d_indices, d_values,
                                                        100, 100, s,
                                                        separator, delimiter)
        print(d_indices[:ii])
        print(d_values[:iv])

    def test_apply_spans_concat_bug_len_1_entry_bstr(self):
        spans = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
        indices = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
        values = np.frombuffer(b'abcd', dtype='S1')
        d_indices = np.zeros(100, dtype=np.int32)
        d_values = np.zeros(100, dtype='S1')
        s, ii, iv = 0, 0, 0
        separator = b','
        delimiter = b'"'
        while s < len(spans) - 1:
            s, ii, iv = persistence._apply_spans_concat(spans, indices, values,
                                                        d_indices, d_values,
                                                        100, 100, s,
                                                        separator, delimiter)
        print(d_indices[:ii])
        print(d_values[:iv])

    def test_apply_spans_concat_fast(self):

        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()

        src_spans = np.asarray([0, 2, 3, 4, 6, 8], dtype=np.int64)
        src_indices = np.asarray([0, 2, 6, 10, 12, 16, 18, 22, 24], dtype=np.int64)
        src_values = np.frombuffer(b'aabbbbccccddeeeeffgggghh', dtype=np.uint8)

        with h5py.File(bio, 'w') as hf:
            foo = rw.IndexedStringWriter(datastore, hf, 'foo', ts)
            foo.write_raw(src_indices, src_values)
            foo_r = datastore.get_reader(hf['foo'])
            datastore.apply_spans_concat(src_spans, foo_r, foo_r.get_writer(hf, 'concatfoo', ts))

            expected = ['aabbbb', 'cccc', 'dd', 'eeeeff', 'gggghh']
            actual = datastore.get_reader(hf['concatfoo'])[:]
            print(actual)
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
            foo = rw.IndexedStringWriter(datastore, hf, 'foo', ts)
            foo.write_raw(src_indices, src_values)
            foo_r = datastore.get_reader(hf['foo'])
            datastore.apply_spans_concat(src_spans, foo_r, foo_r.get_writer(hf, 'concatfoo', ts))

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
            foo = rw.IndexedStringWriter(datastore, hf, 'foo', ts)
            foo.write_raw(src_indices, src_values)
            foo_r = datastore.get_reader(hf['foo'])
            datastore.apply_spans_concat(src_spans, foo_r, foo_r.get_writer(hf, 'concatfoo', ts))

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


    def test_get_spans_single_field_numeric(self):
        datastore = persistence.DataStore(10)
        session = Session()

        for s in (datastore, session):

            a = np.asarray([1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1])
            self.assertTrue(np.array_equal(np.asarray([0, 1, 3, 6, 10, 15]), s.get_spans(field=a)))

            a = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 1])
            self.assertTrue(np.array_equal(np.asarray([0, 5, 9, 12, 14, 15]), s.get_spans(field=a)))

            a = np.asarray([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1])
            self.assertTrue(np.array_equal(np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                           s.get_spans(field=a)))

            a = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            self.assertTrue(np.array_equal(np.asarray([0, 15]), s.get_spans(field=a)))

            a = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
            self.assertTrue(np.array_equal(np.asarray([0, 7, 14, 22]), s.get_spans(field=a)))


    def test_get_spans_single_field_string(self):
        datastore = persistence.DataStore(10)
        session = Session()

        for s in (datastore, session):

            a = np.asarray([b'aa', b'ab', b'ab', b'ac', b'ac', b'ac'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 1, 3, 6]), s.get_spans(field=a)))

            a = np.asarray([b'aa', b'ba', b'ba', b'ca', b'ca', b'ca'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 1, 3, 6]), s.get_spans(field=a)))

            a = np.asarray([b'aa', b'aa', b'aa', b'ab', b'ab', b'ac'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 3, 5, 6]), s.get_spans(field=a)))

            a = np.asarray([b'aa', b'aa', b'aa', b'ba', b'ba', b'ca'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 3, 5, 6]), s.get_spans(field=a)))

            a = np.asarray([b'aa'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 1]), s.get_spans(field=a)))

            a = np.asarray([b'aa', b'aa'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 2]), s.get_spans(field=a)))

            a = np.asarray([b'aa', b'aa', b'aa'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 3]), s.get_spans(field=a)))

            a = np.asarray([b'aa', b'bb', b'cc'], dtype='S2')
            self.assertTrue(np.array_equal(np.asarray([0, 1, 2, 3]), s.get_spans(field=a)))


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
        self.assertTrue(np.array_equal(results, [1, 3, 4, 12, 11, 10]))


    def test_apply_spans_index_of_max(self):
        spans = np.asarray([0, 1, 3, 4, 7, 8, 12])
        values = np.asarray([1, 2, 3, 4, 5, 6, 12, 11, 10, 9, 8, 7])
        results = np.zeros(len(spans)-1, dtype=values.dtype)
        persistence._apply_spans_index_of_max(spans, values, results)
        self.assertTrue(np.array_equal(results, [0, 2, 3, 6, 7, 8]))


    def test_write_to_existing(self):
        datastore = persistence.DataStore(10)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        values = np.arange(95)

        with h5py.File(bio, 'w') as hf:
            rw.NumericWriter(datastore, hf, 'foo', 'int32', ts).write(values)

            reader = rw.NumericReader(datastore, hf['foo'])
            writer = reader.get_writer(hf, 'foo', ts, 'overwrite')
            writer.write(values * 2)
            reader = rw.NumericReader(datastore, hf['foo'])
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
            foo = rw.NumericWriter(datastore, asmts, 'foo', 'int32', ts)
            foo.write(np.arange(95, dtype='int32'))
            trash = datastore.get_trash_group(foo.field)
            hf.move('/asmts/foo', trash.name+'/foo')
            print(hf[trash.name+'/foo'])
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
            foo = rw.NumericWriter(datastore, hf, 'foo', 'uint32', ts)
            foo.write_part(values)
            foo.flush()

        with h5py.File(bio, 'w') as hf:
            footwo = rw.NumericWriter(datastore, hf, 'twofoo', 'uint32', ts)
            foo = rw.NumericReader(datastore, hf['foo'])

            datastore.process({'foo': foo}, {'footwo': footwo}, functor)

            footwo = rw.NumericReader(datastore, hf['twofoo'])
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


    def test_index_spans(self):
        spans = np.asarray([0,2,2,4,5,8,8,10], dtype='int32')
        results = np.zeros(10, dtype='int32')
        persistence._index_spans(spans, results)
        print(results)


    def test_get_shared_index(self):
        datastore = persistence.DataStore(10)
        a = np.asarray(['a', 'a', 'c', 'c', 'd', 'f', 'f', 'g', 'i'])
        b = np.asarray(['a', 'b', 'b', 'c', 'f', 'g', 'h', 'h'])
        c = np.asarray(['b', 'c', 'e', 'e', 'e', 'f', 'f', 'i', 'j'])

        x, y, z = datastore.get_shared_index((a, b, c))
        print(x)
        print(y)
        print(z)

class TestPersistenceOperations(unittest.TestCase):

    def test_filter_non_orphaned_foreign_keys(self):
        pks = np.asarray([1, 2, 4, 5, 7, 8])
        fks = np.asarray([1, 1, 1, 3, 3, 4, 4, 5, 6, 6, 8, 8])
        results = persistence.foreign_key_is_in_primary_key(pks, fks)
        print(results)


    def test_filter(self):

        def filter_framework(name, raw_indices, raw_values, the_filter, expected):
            dest_indices, dest_values =\
                persistence._apply_filter_to_index_values(the_filter, raw_indices, raw_values)
            print(dest_indices)
            print(dest_values)
            w = rw.IndexedStringWriter(datastore, hf, name, ts)
            w.write_raw(dest_indices, dest_values)
            r = datastore.get_reader(hf[name])
            print(r[:])

            self.assertListEqual(r[:], expected)

        datastore = persistence.DataStore(10)
        values = ['True', 'False', '', '', 'False', '', 'True',
                  'Stupendous', '', "I really don't know", 'True',
                  'Ambiguous', '', '', '', 'Things', 'Zombie driver',
                  'Perspicacious', 'False', 'Fa,lse', '', '', 'True',
                  '', 'True', 'Troubador', '', 'Calisthenics', 'The',
                  '', 'Quick', 'Brown', '', '', 'Fox', 'Jumped', '',
                  'Over', 'The', '', 'Lazy', 'Dog']
        bio = BytesIO()
        ts = str(datetime.now(timezone.utc))
        with h5py.File(bio, 'w') as hf:
            rw.IndexedStringWriter(datastore, hf, 'foo', ts).write(values)

            raw_indices = hf['foo']['index'][:]
            raw_values = hf['foo']['values'][:]

            even_filter = np.zeros(len(values), np.bool)
            for i in range(len(even_filter)):
                even_filter[i] = i % 2 == 0
            expected = values[::2]
            filter_framework('even_filter', raw_indices, raw_values,
                             even_filter, expected)

            middle_filter = np.ones(len(values), np.bool)
            middle_filter[0] = False
            middle_filter[-1] = False
            expected = values[1:-1]
            filter_framework('middle_filter', raw_indices, raw_values,
                             middle_filter, expected)

            ends_filter = np.logical_not(middle_filter)
            expected = [values[0]] + [values[-1]]
            filter_framework('end_filter', raw_indices, raw_values,
                             ends_filter, expected)

            all_true_filter = np.ones(len(values), np.bool)
            expected = values
            filter_framework('all_true_filter', raw_indices, raw_values,
                             all_true_filter, expected)

            all_false_filter = np.zeros(len(values), np.bool)
            expected = []
            filter_framework('all_false_filter', raw_indices, raw_values,
                             all_false_filter, expected)

    def test_apply_indices(self):

        def index_framework(name, raw_indices, raw_values, the_indices, expected):
            dest_indices, dest_values = \
                persistence._apply_indices_to_index_values(the_indices, raw_indices, raw_values)
            print(dest_indices)
            print(dest_values)
            w = rw.IndexedStringWriter(datastore, hf, name, ts)
            w.write_raw(dest_indices, dest_values)
            r = datastore.get_reader(hf[name])
            print(r[:])

            self.assertListEqual(r[:], expected)

        datastore = persistence.DataStore(10)
        values = ['True', 'False', '', '', 'False', '', 'True',
                  'Stupendous', '', "I really don't know", 'True',
                  'Ambiguous', '', '', '', 'Things', 'Zombie driver',
                  'Perspicacious', 'False', 'Fa,lse', '', '', 'True',
                  '', 'True', 'Troubador', '', 'Calisthenics', 'The',
                  '', 'Quick', 'Brown', '', '', 'Fox', 'Jumped', '',
                  'Over', 'The', '', 'Lazy', 'Dog']
        bio = BytesIO()
        ts = str(datetime.now(timezone.utc))
        with h5py.File(bio, 'w') as hf:
            rw.IndexedStringWriter(datastore, hf, 'foo', ts).write(values)

            raw_indices = hf['foo']['index'][:]
            raw_values = hf['foo']['values'][:]

            even_indices = np.arange(0, len(values), 2)
            expected = values[::2]
            index_framework('even_filter', raw_indices, raw_values,
                            even_indices, expected)

            # middle_filter = np.ones(len(values), np.bool)
            # middle_filter[0] = False
            # middle_filter[-1] = False
            middle_indices = np.arange(1, len(values)-1)
            expected = values[1:-1]
            index_framework('middle_filter', raw_indices, raw_values,
                            middle_indices, expected)

            # ends_filter = np.logical_not(middle_indices)
            ends_indices = np.asarray([0, len(values)-1])
            expected = [values[0]] + [values[-1]]
            index_framework('end_filter', raw_indices, raw_values,
                            ends_indices, expected)

            # all_true_filter = np.ones(len(values), np.bool)
            all_indices = np.arange(len(values))
            expected = values
            index_framework('all_true_filter', raw_indices, raw_values,
                            all_indices, expected)

            # all_false_filter = np.zeros(len(values), np.bool)
            no_indices = np.asarray([], dtype=np.int64)
            expected = []
            index_framework('all_false_filter', raw_indices, raw_values,
                            no_indices, expected)


    def test_apply_spans_index_of_max(self):
        datastore = persistence.DataStore(10)
        ids = np.asarray(['a', 'a', 'b', 'b', 'b', 'c'], dtype='S1')
        vals = np.asarray([1, 2, 2, 1, 2, 1])
        spans = persistence._get_spans_for_field(ids)
        results = np.zeros(len(spans)-1, dtype=np.int64)
        persistence._apply_spans_index_of_max(spans, vals, results)
        print(results)




    def test_sort(self):

        datastore = persistence.DataStore(10)
        vx = np.asarray([b'a', b'b', b'c', b'd', b'e'], dtype='S1')
        va = np.asarray([1, 2, 2, 1, 1], dtype=np.int32)
        vb = np.asarray([5, 4, 3, 2, 1], dtype=np.int32)
        dt = datetime.now(timezone.utc)
        ts = str(dt)
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            fva = rw.NumericWriter(datastore, hf, 'va', 'uint32', ts)
            fva.write(va)
            fvb = rw.NumericWriter(datastore, hf, 'vb', 'uint32', ts)
            fvb.write(vb)
            fvx = rw.FixedStringWriter(datastore, hf, 'vx', 1, ts)
            fvx.write(vx)

            rva = rw.NumericReader(datastore, hf['va'])
            rvb = rw.NumericReader(datastore, hf['vb'])
            rvx = rw.FixedStringReader(datastore, hf['vx'])
            sindex = datastore.dataset_sort((rva, rvb), np.arange(5, dtype='uint32'))

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

    def test_indexed_string_sort(self):
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

            foo = rw.IndexedStringWriter(datastore, hf, 'foo', ts)
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
            print(len(datastore.get_reader(hf['foo'])))

            self.assertListEqual(values, actual)

        with h5py.File(bio, 'r+') as hf:
            foo = rw.IndexedStringReader(datastore, hf['foo'])
            index = np.asarray(
                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
                 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20,
                 18, 16, 14, 12, 10, 8, 6, 4, 2, 0], dtype=np.int64)
            bar = foo.get_writer(hf, 'bar')
            datastore.apply_sort(index, foo, bar)
            expected = ['False', '', '', 'Stupendous', "I really don't know", 'Ambiguous',
                        '', 'Things', 'Perspicacious', 'Fa,lse', '', '', 'Troubador',
                        'Calisthenics', '', 'Brown', '', 'Jumped', 'Over', '', 'Dog',
                        'Lazy', 'The', '', 'Fox', '', 'Quick', 'The', '', 'True', 'True',
                        '', 'False', 'Zombie driver', '', '', 'True', '', 'True', 'False',
                        '', 'True']
            self.assertListEqual(datastore.get_reader(hf['bar'])[:], expected)


class TestJoining(unittest.TestCase):

    def test_join_fk_to_pk(self):
        ds = persistence.DataStore(10)
        p_id = np.array([100, 200, 300, 400, 500, 600, 800, 900])
        p_val = np.array([-1, -2, -3, -4, -5, -6, -8, -9])
        a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                          600, 600, 700, 700, 900, 900, 900])
        a_val = np.array([10, 11, 12, 23, 22, 43, 40, 41, 41, 60,
                          63, 62, 71, 71, 92, 92, 92])
        index = ds.get_index(p_id, a_pid)
        print(index)
        print(persistence._map_valid_indices(p_val, index, 0))
        index = ds.get_index(a_pid, p_id)
        print(index)
        print(persistence._map_valid_indices(a_val, index, 0))

    def test_join_pk_to_fk(self):
        ds = persistence.DataStore(10)
        p_id = np.array([b'a', b'b', b'c', b'd', b'e'], dtype='S1')
        t_pid = np.array([b'a', b'c', b'd'], dtype='S1')
        index = ds.get_index(t_pid, p_id)
        print(index)


    def test_fk_in_pk(self):
        ds = persistence.DataStore(10)
        p_id = np.array([100, 200, 300, 400, 500, 600, 800, 900])
        a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                          600, 600, 700, 700, 900, 900, 900])
        fk_in_pk = persistence.foreign_key_is_in_primary_key(p_id, a_pid)
        print(fk_in_pk)

    def test_failure_case_from_supplements(self):
        ds = persistence.DataStore(10)
        # p_id = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
        p_id = np.array([900, 800, 700, 600, 500, 400, 300, 200, 100])
        a_pid = np.array([100, 100, 100, 200, 200, 400, 400, 400, 400, 600,
                          600, 600, 700, 700, 900, 900, 900])
        a_val = np.array([10, 11, 12, 23, 22, 43, 40, 41, 41, 60,
                          63, 62, 71, 71, 92, 92, 92])

        a_spans = ds.get_spans(a_pid)
        print("spans:", a_spans)
        indices_of_max = ds.apply_spans_index_of_max(a_spans, a_val)
        print("iom:", indices_of_max)
        p_to_a_indices = ds.get_index(a_pid[indices_of_max], p_id)
        print("ixd_a_pid:", a_pid[indices_of_max])
        print("p_to_a_i:", p_to_a_indices)

        p_to_vals = np.zeros(len(p_to_a_indices), np.int32)
        for i_r in range(len(p_to_a_indices)):
            if p_to_a_indices[i_r] >= operations.INVALID_INDEX:
                p_to_vals[i_r] = -1
            else:
                p_to_vals[i_r] = a_val[indices_of_max[p_to_a_indices[i_r]]]
        print("p_to_vals:", p_to_vals)

        sel_a_val = a_val[indices_of_max]
        print("sel_a_val:", sel_a_val)
        p_to_vals = np.zeros(len(p_to_a_indices), np.int32)
        for i_r in range(len(p_to_a_indices)):
            if p_to_a_indices[i_r] >= operations.INVALID_INDEX:
                p_to_vals[i_r] = -1
            else:
                p_to_vals[i_r] = sel_a_val[p_to_a_indices[i_r]]
        print("p_to_vals:", p_to_vals)


    def test_join_pk_to_fk_2(self):
        ds = persistence.DataStore(10)
        ts = str(datetime.now(timezone.utc))
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            a = hf.create_group('assessments')
            p = hf.create_group('patients')
            ds.get_fixed_string_writer(a, 'id', 2, ts).write(
                [e.encode() for e in [
                    'aa', 'ab', 'ac', 'ba', 'bb', 'bc', 'ca', 'cd', 'ea']]
            )
            ds.get_fixed_string_writer(a, 'pid', 1, ts).write(
                [e.encode() for e in [
                    'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'e']]
            )
            ds.get_numeric_writer(a, 'ill', 'int32', ts).write(
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
            )
            ds.get_fixed_string_writer(p, 'id', 1, ts).write(
                [e.encode() for e in ['a', 'c', 'd', 'e']]
            )
            ds.get_numeric_writer(p, 'age', 'int32', ts).write(
                np.array([18, 90, 45, 60], dtype=np.int32)
            )
            ds.get_index(
                ds.get_reader(a['pid']),
                ds.get_reader(p['id']),
                ds.get_numeric_writer(p, 'pid_to_apid', 'int64', ts),
            )
            print(ds.get_reader(p['pid_to_apid'])[:])
            ds.get_index(
                ds.get_reader(p['id']),
                ds.get_reader(a['pid']),
                ds.get_numeric_writer(a, 'apid_to_pid', 'int64', ts),
            )
            print('fkey:', ds.get_reader(a['apid_to_pid'])[:])

            # print(ds.get_reader(p['age'])[:][ds.get_reader(a['apid_to_pid'])[:]])

            result = persistence._map_valid_indices(ds.get_reader(p['age'])[:],
                                                    ds.get_reader(a['apid_to_pid'])[:],
                                                    -100)
            print(result)
            # TODO: appears to be a bug in h5py that doesn't allow complex numpy indexing
            # print(ds.get_reader(p['age'])[ds.get_reader(a['apid_to_pid'])[:]])

            # aages = ds.get_reader(dest_asmts['age'])[:]
            # apids = ds.get_reader(src_asmts['patient_id'])[:]
            # pages = ds.get_reader(src_ptnts['age'])[:]
            # pids = ds.get_reader(src_ptnts['id'])[:]
            # t0 = time.time()
            # from collections import defaultdict
            # dpages = defaultdict(int)
            # for i_r in range(len(pids)):
            #     dpages[pids[i_r]] = pages[i_r]
            #
            # not_in = 0
            # for i_r in range(len(apids)):
            #     if apids[i_r] in dpages:
            #         if aages[i_r] != dpages[apids[i_r]]:
            #             print("bad_mapping:", i_r, apids[i_r], aages[i_r], dpages[apids[i_r]])
            #     else:
            #         not_in += 1
            # print("not_in:", not_in)
            # print(f"mapping checked in {time.time() - t0}s")

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
                rw.IndexedStringWriter(datastore, hf, 'vals', ts).write(sv)

                vals = rw.IndexedStringReader(datastore, hf['vals'])
                wvals = vals.get_writer(hf, 'sorted_vals', ts)
                vals.sort(np.asarray(si, dtype=np.uint32), wvals)
                actual = rw.IndexedStringReader(datastore, hf['sorted_vals'])[:]
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
        with utils.Timer("sorting values"):
            s_index = sorted(index, key=lambda x: values[x])

        index = np.arange(count, dtype=np.uint32)
        t0 = time.time()
        with utils.Timer("sorting values"):
            s_index = sorted(index, key=predicate)


class TestDataWriter(unittest.TestCase):

    def test_data_writer(self):
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            rw.DataWriter.write(hf, 'x', [], 0, dtype='int32')
            print(hf['x'])


# class TestLongPersistence(unittest.TestCase):
#
#     def test_large_dataset_chunk_settings(self):
#         import time
#         import random
#         import numpy as np
#
#         with h5py.File('covid_test.hdf5', 'w') as hf:
#             random.seed(12345678)
#             count = 1000000
#             chunk = 100000
#             data = np.zeros(count, dtype=np.uint32)
#             for i in range(count):
#                 data[i] = random.randint(0, 1000)
#             ds = hf.create_dataset('foo', (count,), chunks=(chunk,), maxshape=(None,), data=data)
#             ds2 = hf.create_dataset('foo2', (count,), data=data)
#
#         with h5py.File('covid_test.hdf5', 'r') as hf:
#
#             ds = hf['foo'][()]
#             print('foo parse')
#             t0 = time.time()
#             total = 0
#             for d in ds:
#                 total += d
#             print(f"{total} in {time.time() - t0}")
#
#             ds = hf['foo']
#             print('foo parse')
#             t0 = time.time()
#             total = 0
#             for d in ds:
#                 total += d
#             print(f"{total} in {time.time() - t0}")
#
#             ds = hf['foo2'][()]
#             print('foo parse')
#             t0 = time.time()
#             total = 0
#             for d in ds:
#                 total += d
#             print(f"{total} in {time.time() - t0}")
#
#             ds = hf['foo2']
#             print('foo parse')
#             t0 = time.time()
#             total = 0
#             for d in ds:
#                 total += d
#             print(f"{total} in {time.time() - t0}")


class TestValidation(unittest.TestCase):

    def test_check_all_readers_valid_and_same_type(self):
        ds = persistence.DataStore()
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            x = hf.create_group('x')
            ds.get_numeric_writer(x, 'a', 'int32').write(np.asarray([1, 2, 3, 4]))
            ds.get_fixed_string_writer(x, 'b', 1).write(np.asarray([b'a', b'b', b'c', b'd']))
            val._check_all_readers_valid_and_same_type((x['a'], x['b']))
            ra = ds.get_reader(x['a'])
            rb = ds.get_reader(x['b'])
            val._check_all_readers_valid_and_same_type((ra, rb))
            val._check_all_readers_valid_and_same_type((ra[:], rb[:]))

