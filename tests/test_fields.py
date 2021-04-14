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
            dst = s.open_dataset(bio, "w", "src")
            src=dst.create_dataframe('src')
            f = s.create_indexed_string(src, "a")
            self.assertTrue(bool(f))
            f = s.create_fixed_string(src, "b", 5)
            self.assertTrue(bool(f))
            f = s.create_numeric(src, "c", "int32")
            self.assertTrue(bool(f))
            f = s.create_categorical(src, "d", "int8", {"no": 0, "yes": 1})
            self.assertTrue(bool(f))


class TestFieldGetSpans(unittest.TestCase):

    def test_get_spans(self):
        '''
        Here test only the numeric field, categorical field and fixed string field.
        Indexed string see TestIndexedStringFields below
        '''
        vals = np.asarray([0, 1, 1, 3, 3, 6, 5, 5, 5], dtype=np.int32)
        bio = BytesIO()
        with session.Session() as s:
            self.assertListEqual([0, 1, 3, 5, 6, 9], s.get_spans(vals).tolist())

            dst = s.open_dataset(bio, "w", "src")
            ds = dst.create_dataframe('src')
            vals_f = s.create_numeric(ds, "vals", "int32")
            vals_f.data.write(vals)
            self.assertListEqual([0, 1, 3, 5, 6, 9], vals_f.get_spans().tolist())

            fxdstr = s.create_fixed_string(ds, 'fxdstr', 2)
            fxdstr.data.write(np.asarray(['aa', 'bb', 'bb', 'cc', 'cc', 'dd', 'dd', 'dd', 'ee'], dtype='S2'))
            self.assertListEqual([0,1,3,5,8,9],list(fxdstr.get_spans()))

            cat = s.create_categorical(ds, 'cat', 'int8', {'a': 1, 'b': 2})
            cat.data.write([1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2])
            self.assertListEqual([0,2,4,7,10,11,12,13,14],list(cat.get_spans()))


class TestIsSorted(unittest.TestCase):

    def test_indexed_string_is_sorted(self):
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('foo')

            f = df.create_indexed_string('f')
            vals = ['the', 'quick', '', 'brown', 'fox', 'jumps', '', 'over', 'the', 'lazy', '', 'dog']
            f.data.write(vals)
            self.assertFalse(f.is_sorted())

            f2 = df.create_indexed_string('f2')
            svals = sorted(vals)
            f2.data.write(svals)
            self.assertTrue(f2.is_sorted())

    def test_fixed_string_is_sorted(self):
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('foo')

            f = df.create_fixed_string('f', 5)
            vals = ['a', 'ba', 'bb', 'bac', 'de', 'ddddd', 'deff', 'aaaa', 'ccd']
            f.data.write([v.encode() for v in vals])
            self.assertFalse(f.is_sorted())

            f2 = df.create_fixed_string('f2', 5)
            svals = sorted(vals)
            f2.data.write([v.encode() for v in svals])
            self.assertTrue(f2.is_sorted())

    def test_numeric_is_sorted(self):
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('foo')

            f = df.create_numeric('f', 'int32')
            vals = [74, 1897, 298, 0, -100098, 380982340, 8, 6587, 28421, 293878]
            f.data.write(vals)
            self.assertFalse(f.is_sorted())

            f2 = df.create_numeric('f2', 'int32')
            svals = sorted(vals)
            f2.data.write(svals)
            self.assertTrue(f2.is_sorted())

    def test_categorical_is_sorted(self):
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('foo')

            f = df.create_categorical('f', 'int8', {'a': 0, 'c': 1, 'd': 2, 'b': 3})
            vals = [0, 1, 3, 2, 3, 2, 2, 0, 0, 1, 2]
            f.data.write(vals)
            self.assertFalse(f.is_sorted())

            f2 = df.create_categorical('f2', 'int8', {'a': 0, 'c': 1, 'd': 2, 'b': 3})
            svals = sorted(vals)
            f2.data.write(svals)
            self.assertTrue(f2.is_sorted())

    def test_timestamp_is_sorted(self):
        from datetime import datetime as D
        from datetime import timedelta as T
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('foo')

            f = df.create_timestamp('f')
            d = D(2020, 5, 10)
            vals = [d + T(seconds=50000), d - T(days=280), d + T(weeks=2), d + T(weeks=250),
                    d - T(weeks=378), d + T(hours=2897), d - T(days=23), d + T(minutes=39873)]
            vals = [v.timestamp() for v in vals]
            f.data.write(vals)
            self.assertFalse(f.is_sorted())

            f2 = df.create_timestamp('f2')
            svals = sorted(vals)
            f2.data.write(svals)
            self.assertTrue(f2.is_sorted())


class TestIndexedStringFields(unittest.TestCase):

    def test_create_indexed_string(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            hf = dst.create_dataframe('src')
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
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            hf = dst.create_dataframe('src')
            strings = ['a', 'bb', 'ccc', 'dddd']
            f = fields.IndexedStringImporter(s, hf, 'foo')
            f.write(strings)
            values = hf['foo'].values[:]
            self.assertListEqual([97, 98, 98, 99, 99, 99, 100, 100, 100, 100], values.tolist())

    def test_index_string_field_get_span(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            ds = dst.create_dataframe('src')
            idx = s.create_indexed_string(ds, 'idx')
            idx.data.write(['aa', 'bb', 'bb', 'c', 'c', 'c', 'ddd', 'ddd', 'e', 'f', 'f', 'f'])
            self.assertListEqual([0, 1, 3, 6, 8, 9, 12], s.get_spans(idx))


class TestFieldArray(unittest.TestCase):

    def test_write_part(self):
        bio = BytesIO()
        s = session.Session()
        ds = s.open_dataset(bio, "w", "src")
        dst = ds.create_dataframe('src')
        num = s.create_numeric(dst, 'num', 'int32')
        num.data.write_part(np.arange(10))
        self.assertListEqual([0,1,2,3,4,5,6,7,8,9],list(num.data[:]))

    def test_clear(self):
        bio = BytesIO()
        s = session.Session()
        ds = s.open_dataset(bio, "w", "src")
        dst = ds.create_dataframe('src')
        num = s.create_numeric(dst, 'num', 'int32')
        num.data.write_part(np.arange(10))
        num.data.clear()
        self.assertListEqual([], list(num.data[:]))



class TestFieldArray(unittest.TestCase):

    def test_write_part(self):
        bio = BytesIO()
        s = session.Session()
        ds = s.open_dataset(bio, "w", "src")
        dst = ds.create_dataframe('src')
        num = s.create_numeric(dst, 'num', 'int32')
        num.data.write_part(np.arange(10))
        self.assertListEqual([0,1,2,3,4,5,6,7,8,9],list(num.data[:]))

    def test_clear(self):
        bio = BytesIO()
        s = session.Session()
        ds = s.open_dataset(bio, "w", "src")
        dst = ds.create_dataframe('src')
        num = s.create_numeric(dst, 'num', 'int32')
        num.data.write_part(np.arange(10))
        num.data.clear()
        self.assertListEqual([], list(num.data[:]))


class TestMemoryFields(unittest.TestCase):

    def _execute_memory_field_test(self, a1, a2, scalar, function):

        def test_simple(expected, actual):
            self.assertListEqual(expected.tolist(), actual.data[:].tolist())

        def test_tuple(expected, actual):
            self.assertListEqual(expected[0].tolist(), actual[0].data[:].tolist())
            self.assertListEqual(expected[1].tolist(), actual[1].data[:].tolist())

        expected = function(a1, a2)
        expected_scalar = function(a1, scalar)
        expected_rscalar = function(scalar, a2)

        test_equal = test_tuple if isinstance(expected, tuple) else test_simple

        s = session.Session()
        f1 = fields.NumericMemField(s, 'int32')
        f2 = fields.NumericMemField(s, 'int32')
        f1.data.write(a1)
        f2.data.write(a2)

        test_equal(expected, function(f1, f2))
        test_equal(expected, function(f1, a2))
        test_equal(expected, function(fields.as_field(a1), f2))
        test_equal(expected_scalar, function(f1, 1))
        test_equal(expected_rscalar, function(1, f2))


    def _execute_field_test(self, a1, a2, scalar, function):

        def test_simple(expected, actual):
            self.assertListEqual(expected.tolist(), actual.data[:].tolist())

        def test_tuple(expected, actual):
            self.assertListEqual(expected[0].tolist(), actual[0].data[:].tolist())
            self.assertListEqual(expected[1].tolist(), actual[1].data[:].tolist())

        expected = function(a1, a2)
        expected_scalar = function(a1, scalar)
        expected_rscalar = function(scalar, a2)

        test_equal = test_tuple if isinstance(expected, tuple) else test_simple

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')

            m1 = fields.NumericMemField(s, fields.FieldDataOps.dtype_to_str(a1.dtype))
            m2 = fields.NumericMemField(s, fields.FieldDataOps.dtype_to_str(a2.dtype))
            m1.data.write(a1)
            m2.data.write(a2)

            f1 = df.create_numeric('f1', fields.FieldDataOps.dtype_to_str(a1.dtype))
            f2 = df.create_numeric('f2', fields.FieldDataOps.dtype_to_str(a2.dtype))
            f1.data.write(a1)
            f2.data.write(a2)

            # test memory field and field operations
            test_equal(expected, function(f1, f2))
            test_equal(expected, function(f1, m2))
            test_equal(expected, function(m1, f2))
            test_equal(expected_scalar, function(f1, scalar))
            test_equal(expected_rscalar, function(scalar, f2))

            # test that the resulting memory field writes to a non-memory field properly
            r = function(f1, f2)
            if isinstance(r, tuple):
                df.create_numeric(
                    'f3a', fields.FieldDataOps.dtype_to_str(r[0].data.dtype)).data.write(r[0])
                df.create_numeric(
                    'f3b', fields.FieldDataOps.dtype_to_str(r[1].data.dtype)).data.write(r[1])
                test_simple(expected[0], df['f3a'])
                test_simple(expected[1], df['f3b'])
            else:
                df.create_numeric(
                    'f3', fields.FieldDataOps.dtype_to_str(r.data.dtype)).data.write(r)
                test_simple(expected, df['f3'])

    def test_mixed_field_add(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x + y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x + y)

    def test_mixed_field_sub(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x - y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x - y)

    def test_mixed_field_mul(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x * y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x * y)

    def test_mixed_field_div(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x / y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x / y)

    def test_mixed_field_floordiv(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x // y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x // y)

    def test_mixed_field_mod(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x % y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x % y)


    def test_mixed_field_divmod(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: divmod(x, y))
        self._execute_field_test(a1, a2, 1, lambda x, y: divmod(x, y))

    def test_mixed_field_and(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x & y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x & y)

    def test_mixed_field_xor(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x ^ y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x ^ y)

    def test_mixed_field_or(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        a2 = np.array([2, 3, 4, 5], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x | y)
        self._execute_field_test(a1, a2, 1, lambda x, y: x | y)
