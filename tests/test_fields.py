from pickle import FALSE
import unittest

import numpy as np
from io import BytesIO

import h5py
from datetime import datetime
from parameterized import parameterized

from .utils import SessionTestCase, shuffle_randstate, allow_slow_tests, DEFAULT_FIELD_DATA

from exetera.core import session
from exetera.core import fields
from exetera.core import persistence as per
from exetera.io import field_importers as fi
from exetera.core import utils
import itertools

NUMERIC_ONLY = [d for d in DEFAULT_FIELD_DATA if d[0] == "create_numeric"]


class TestDefaultData(SessionTestCase):
    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_fields(self, creator, name, kwargs, data):
        """
        Tests basic creation of every field type, checking it's contents are actually what was put into them.
        """
        f = self.setup_field(self.df, creator, name, (), kwargs, data)
        
        # Convert numeric fields and use Numpy's conversion as an oracle to test overflown values in field. If a value
        # overflows when stored in a field then the field's contents will obviously vary compared to `data`, so change
        # data to match by using Numpy to handle overflow for us.
        if "nformat" in kwargs:
            data = np.asarray(data, dtype=kwargs["nformat"])
        
        self.assertFieldEqual(data, f)
        

class TestFieldExistence(SessionTestCase):
    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_field_truthness(self, creator, name, kwargs, data):
        """Test every field object is considered True."""
        f = self.setup_field(self.df, creator, name, (), kwargs, data)
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
            ds = s.open_dataset(bio, 'w', 'src')
            df = ds.create_dataframe('src')
            f = df.create_indexed_string('f')
            d = f.data[:]
            #print(d)


    def test_filter_indexed_string(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            hf = dst.create_dataframe('src')
            data = ['a', 'bb', 'ccc', 'dddd']
            indices, values, offsets, written_row_count = utils.one_dim_data_to_indexed_for_test(data, 10)               
            foo = fi.IndexedStringImporter(s, hf, 'foo')
            foo.import_part(indices, values, offsets, 0, written_row_count)

            self.assertListEqual([0, 1, 3, 6, 10], hf['foo'].indices[:].tolist())

            f2 = s.create_indexed_string(hf, 'bar')
            s.apply_filter(np.asarray([False, True, True, False]), hf['foo'], f2)
            self.assertListEqual([0, 2, 5], f2.indices[:].tolist())
            self.assertListEqual([98, 98, 99, 99, 99], f2.values[:].tolist())
            self.assertListEqual(['bb', 'ccc'], f2.data[:])
            self.assertEqual('bb', f2.data[0])
            self.assertEqual('ccc', f2.data[1])


    def test_reindex_indexed_string(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            hf = dst.create_dataframe('src')
            data = ['a', 'bb', 'ccc', 'dddd']
            indices, values, offsets, written_row_count = utils.one_dim_data_to_indexed_for_test(data, 10)               
            foo = fi.IndexedStringImporter(s, hf, 'foo')
            foo.import_part(indices, values, offsets, 0, written_row_count)
            self.assertListEqual([0, 1, 3, 6, 10], hf['foo'].indices[:].tolist())

            f2 = s.create_indexed_string(hf, 'bar')
            s.apply_index(np.asarray([3, 0, 2, 1], dtype=np.int64), hf['foo'], f2)
            self.assertListEqual([0, 4, 5, 8, 10], f2.indices[:].tolist())
            self.assertListEqual([100, 100, 100, 100, 97, 99, 99, 99, 98, 98],
                                 f2.values[:].tolist())
            self.assertListEqual(['dddd', 'a', 'ccc', 'bb'], f2.data[:])


    def test_update_legacy_indexed_string_that_has_uint_values(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            hf = dst.create_dataframe('src')
            data = ['a', 'bb', 'ccc', 'dddd']
            indices, values, offsets, written_row_count = utils.one_dim_data_to_indexed_for_test(data, 10)               
            foo = fi.IndexedStringImporter(s, hf, 'foo')
            foo.import_part(indices, values, offsets, 0, written_row_count)
            self.assertListEqual([97, 98, 98, 99, 99, 99, 100, 100, 100, 100], hf['foo'].values[:].tolist())


    def test_index_string_field_get_span(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, "w", "src")
            ds = dst.create_dataframe('src')
            idx = s.create_indexed_string(ds, 'idx')
            idx.data.write(['aa', 'bb', 'bb', 'c', 'c', 'c', 'ddd', 'ddd', 'e', 'f', 'f', 'f'])
            self.assertListEqual([0, 1, 3, 6, 8, 9, 12], s.get_spans(idx))


class TestFieldArray(SessionTestCase):
    @parameterized.expand(NUMERIC_ONLY)
    def test_write_part(self, creator, name, kwargs, data):
        """
        Checks that `write_part` will write the data into each field type.
        """
        f = self.s.create_numeric(self.df, name, **kwargs)
        f.data.write_part(data)
        
        if "nformat" in kwargs:
            data = np.asarray(data, dtype=kwargs["nformat"])
        
        self.assertFieldEqual(data, f)

    @parameterized.expand(NUMERIC_ONLY)
    def test_clear(self, creator, name, kwargs, data):
        """
        Checks that `clear` removes data from every field type.
        """
        f = self.s.create_numeric(self.df, name, **kwargs)
        f.data.write_part(data)
        f.data.clear()
        self.assertFieldEqual([], f)


class TestMemoryFieldCreateLike(unittest.TestCase):


    def test_categorical_create_like(self):
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            foo = df.create_categorical('foo', 'int8', {b'a': 0, b'b': 1})
            foo.data.write(np.array([0, 1, 1, 0]))
            foo2 = foo.create_like(df, 'foo2')
            foo2.data.write(foo)
            self.assertListEqual([0, 1, 1, 0], foo2.data[:].tolist())

    def test_numeric_create_like(self):
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            foo = df.create_numeric('foo', 'int32')
            foo.data.write(np.array([1, 2, 3, 4]))
            mfoo = foo + 1
            foo2 = mfoo.create_like(df, 'foo2')
            foo2.data.write(mfoo)
            self.assertListEqual([2, 3, 4, 5], foo2.data[:].tolist())


    def test_indexed_string_create_like(self):
        data = np.array(['bb', 'a', 'ccc', 'ccd'])
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')

            mf = fields.IndexedStringMemField(s)
            mf.data.write(data)
            self.assertIsInstance(mf, fields.IndexedStringMemField)

            mmmf = mf.create_like(df, 'mmmf')
            mmmf.data.write(data)
            self.assertListEqual(data.tolist(), mmmf.data[:])


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

            m1 = fields.NumericMemField(s, fields.dtype_to_str(a1.dtype))
            m2 = fields.NumericMemField(s, fields.dtype_to_str(a2.dtype))
            m1.data.write(a1)
            m2.data.write(a2)

            f1 = df.create_numeric('f1', fields.dtype_to_str(a1.dtype))
            f2 = df.create_numeric('f2', fields.dtype_to_str(a2.dtype))
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
                    'f3a', fields.dtype_to_str(r[0].data.dtype)).data.write(r[0])
                df.create_numeric(
                    'f3b', fields.dtype_to_str(r[1].data.dtype)).data.write(r[1])
                test_simple(expected[0], df['f3a'])
                test_simple(expected[1], df['f3b'])
            else:
                df.create_numeric(
                    'f3', fields.dtype_to_str(r.data.dtype)).data.write(r)
                test_simple(expected, df['f3'])

    def _execute_unary_field_test(self, a1, function):

        def test_simple(expected, actual):
            self.assertListEqual(expected.tolist(), actual.data[:].tolist())

        def test_tuple(expected, actual):
            self.assertListEqual(expected[0].tolist(), actual[0].data[:].tolist())
            self.assertListEqual(expected[1].tolist(), actual[1].data[:].tolist())

        expected = function(a1)

        test_equal = test_tuple if isinstance(expected, tuple) else test_simple

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')

            m1 = fields.NumericMemField(s, fields.dtype_to_str(a1.dtype))
            m1.data.write(a1)

            f1 = df.create_numeric('f1', fields.dtype_to_str(a1.dtype))
            f1.data.write(a1)

            # test memory field and field operations
            test_equal(expected, function(f1))
            test_equal(expected, function(f1))
            test_equal(expected, function(m1))

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

    def test_mixed_field_invert(self):
        a1 = np.array([0, 0, 1, 1], dtype=np.int32)
        self._execute_unary_field_test(a1, lambda x: ~x)

    def test_logical_not(self):
        a1 = np.array([0, 0, 1, 1], dtype=np.int32)
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            num = df.create_numeric('num', 'uint32')
            num.data.write(a1)
            self.assertListEqual(np.logical_not(a1).tolist(), num.logical_not().data[:].tolist())

    def test_less_than(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        r = 1 < a1
        a2 = np.array([5, 4, 3, 2], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x < y)

    def test_less_than_equal(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        r = 1 < a1
        a2 = np.array([5, 4, 3, 2], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x <= y)

    def test_equal(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        r = 1 < a1
        a2 = np.array([5, 4, 3, 2], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x == y)

    def test_not_equal(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        r = 1 < a1
        a2 = np.array([5, 4, 3, 2], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x != y)

    def test_greater_than_equal(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        r = 1 < a1
        a2 = np.array([5, 4, 3, 2], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x >= y)

    def test_greater_than(self):

        a1 = np.array([1, 2, 3, 4], dtype=np.int32)
        r = 1 < a1
        a2 = np.array([5, 4, 3, 2], dtype=np.int32)
        self._execute_memory_field_test(a1, a2, 1, lambda x, y: x > y)

    def test_categorical_remap(self):

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            foo = df.create_categorical('foo', 'int8', {b'a': 1, b'b': 2})
            foo.data.write(np.array([1, 2, 2, 1], dtype='int8'))
            mbar = foo.remap([(1, 0), (2, 1)], {b'a': 0, b'b': 1})
            self.assertListEqual([0, 1, 1, 0], mbar.data[:].tolist())
            self.assertDictEqual({0: b'a', 1: b'b'}, mbar.keys)
            bar = mbar.create_like(df, 'bar')
            bar.data.write(mbar)
            self.assertListEqual([0, 1, 1, 0], mbar.data[:].tolist())
            self.assertDictEqual({0: b'a', 1: b'b'}, mbar.keys)


class TestFieldApplyFilter(unittest.TestCase):

    def test_indexed_string_apply_filter(self):

        data = ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
        filt = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

        expected_indices = [0, 1, 3, 6, 10, 10, 14, 17, 19, 20]
        expected_values = [97, 98, 98, 99, 99, 99, 100, 100, 100, 100,
                           101, 101, 101, 101, 102, 102, 102, 103, 103, 104]
        expected_filt_indices = [0, 2, 6, 10, 12]
        expected_filt_values = [98, 98, 100, 100, 100, 100, 101, 101, 101, 101, 103, 103]
        expected_filt_data = ['bb', 'dddd', 'eeee', 'gg']

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_indexed_string('f')
            f.data.write(data)
            self.assertListEqual(expected_indices, f.indices[:].tolist())
            self.assertListEqual(expected_values, f.values[:].tolist())
            self.assertListEqual(data, f.data[:])

            ff = f.apply_filter(filt, in_place=True)
            self.assertListEqual(expected_filt_indices, f.indices[:].tolist())
            self.assertListEqual(expected_filt_values, f.values[:].tolist())
            self.assertListEqual(expected_filt_data, f.data[:])
            self.assertListEqual(expected_filt_indices, ff.indices[:].tolist())
            self.assertListEqual(expected_filt_values, ff.values[:].tolist())
            self.assertListEqual(expected_filt_data, ff.data[:])

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_filter(filt, fg)
            self.assertListEqual(expected_filt_indices, fg.indices[:].tolist())
            self.assertListEqual(expected_filt_values, fg.values[:].tolist())
            self.assertListEqual(expected_filt_data, fg.data[:])
            self.assertListEqual(expected_filt_indices, fgr.indices[:].tolist())
            self.assertListEqual(expected_filt_values, fgr.values[:].tolist())
            self.assertListEqual(expected_filt_data, fgr.data[:])
            fh = g.apply_filter(filt)
            self.assertListEqual(expected_filt_indices, fh.indices[:].tolist())
            self.assertListEqual(expected_filt_values, fh.values[:].tolist())
            self.assertListEqual(expected_filt_data, fh.data[:])

            mf = fields.IndexedStringMemField(s)
            mf.data.write(data)
            self.assertListEqual(expected_indices, mf.indices[:].tolist())
            self.assertListEqual(expected_values, mf.values[:].tolist())
            self.assertListEqual(data, mf.data[:])

            mf.apply_filter(filt, in_place=True)
            self.assertListEqual(expected_filt_indices, mf.indices[:].tolist())
            self.assertListEqual(expected_filt_values, mf.values[:].tolist())
            self.assertListEqual(expected_filt_data, mf.data[:])

            b = df.create_indexed_string('bar')
            b.data.write(data)
            self.assertListEqual(expected_indices, b.indices[:].tolist())
            self.assertListEqual(expected_values, b.values[:].tolist())
            self.assertListEqual(data, b.data[:])

            mb = b.apply_filter(filt)
            self.assertListEqual(expected_filt_indices, mb.indices[:].tolist())
            self.assertListEqual(expected_filt_values, mb.values[:].tolist())
            self.assertListEqual(expected_filt_data, mb.data[:])

            df_t = ds.create_dataframe('df_t')
            df_t['bar'] = b.apply_filter(filt)
            self.assertListEqual(expected_filt_indices, df_t['bar'].indices[:].tolist())
            self.assertListEqual(expected_filt_values, df_t['bar'].values[:].tolist())
            self.assertListEqual(expected_filt_data, df_t['bar'].data[:])


    def test_fixed_string_apply_filter(self):
        data = np.array([b'a', b'bb', b'ccc', b'dddd', b'eeee', b'fff', b'gg', b'h'], dtype='S4')
        filt = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        expected = [b'bb', b'dddd', b'fff', b'h']

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_fixed_string('foo', 4)
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_filter(filt, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_filter(filt)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.FixedStringMemField(s, 4)
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_fixed_string('bar', 4)
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_filter(filt)
            self.assertListEqual(expected, mb.data[:].tolist())

    def test_numeric_apply_filter(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        filt = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        expected = [2, 4, 6, 8]

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_numeric('foo', 'int32')
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_filter(filt, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_filter(filt)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.NumericMemField(s, 'int32')
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_numeric('bar', 'int32')
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_filter(filt)
            self.assertListEqual(expected, mb.data[:].tolist())

    def test_categorical_apply_filter(self):
        data = np.array([0, 1, 2, 0, 1, 2, 2, 1, 0], dtype=np.int32)
        keys = {b'a': 0, b'b': 1, b'c': 2}
        filt = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        expected = [1, 0, 2, 1]

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_categorical('foo', 'int8', keys)
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_filter(filt, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_filter(filt)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.CategoricalMemField(s, 'int8', keys)
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_categorical('bar', 'int8', keys)
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_filter(filt)
            self.assertListEqual(expected, mb.data[:].tolist())

    def test_timestamp_apply_filter(self):
        from datetime import datetime as D
        from datetime import timezone
        data = [D(2020, 1, 1, tzinfo=timezone.utc), D(2021, 5, 18, tzinfo=timezone.utc), D(2950, 8, 17, tzinfo=timezone.utc), D(1840, 10, 11, tzinfo=timezone.utc),
                D(2110, 11, 1, tzinfo=timezone.utc), D(2002, 3, 3, tzinfo=timezone.utc), D(2018, 2, 28, tzinfo=timezone.utc), D(2400, 9, 1, tzinfo=timezone.utc)]
        data = np.asarray([d.timestamp() for d in data], dtype=np.float64)
        filt = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
        expected = data[filt].tolist()

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_timestamp('foo')
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_filter(filt, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_filter(filt)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.TimestampMemField(s)
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_filter(filt, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_timestamp('bar')
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_filter(filt)
            self.assertListEqual(expected, mb.data[:].tolist())


class TestFieldApplyIndex(unittest.TestCase):

    def test_indexed_string_apply_index(self):

        data = ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
        inds = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

        expected_indices = [0, 1, 3, 6, 10, 10, 14, 17, 19, 20]
        expected_values = [97, 98, 98, 99, 99, 99, 100, 100, 100, 100,
                           101, 101, 101, 101, 102, 102, 102, 103, 103, 104]
        expected_filt_indices = [0, 1, 2, 4, 6, 9, 12, 16, 20, 20]
        expected_filt_values = [104, 97, 103, 103, 98, 98, 102, 102, 102, 99, 99, 99,
                                101, 101, 101, 101, 100, 100, 100, 100]
        expected_filt_data = ['h', 'a', 'gg', 'bb', 'fff', 'ccc', 'eeee', 'dddd', '']

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_indexed_string('foo')
            f.data.write(data)
            self.assertListEqual(expected_indices, f.indices[:].tolist())
            self.assertListEqual(expected_values, f.values[:].tolist())
            self.assertListEqual(data, f.data[:])

            ff = f.apply_index(inds, in_place=True)
            self.assertListEqual(expected_filt_indices, f.indices[:].tolist())
            self.assertListEqual(expected_filt_values, f.values[:].tolist())
            self.assertListEqual(expected_filt_data, f.data[:])
            self.assertListEqual(expected_filt_indices, ff.indices[:].tolist())
            self.assertListEqual(expected_filt_values, ff.values[:].tolist())
            self.assertListEqual(expected_filt_data, ff.data[:])

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_index(inds, fg)
            self.assertListEqual(expected_filt_indices, fg.indices[:].tolist())
            self.assertListEqual(expected_filt_values, fg.values[:].tolist())
            self.assertListEqual(expected_filt_data, fg.data[:])
            self.assertListEqual(expected_filt_indices, fgr.indices[:].tolist())
            self.assertListEqual(expected_filt_values, fgr.values[:].tolist())
            self.assertListEqual(expected_filt_data, fgr.data[:])

            fh = g.apply_index(inds)
            self.assertListEqual(expected_filt_indices, fh.indices[:].tolist())
            self.assertListEqual(expected_filt_values, fh.values[:].tolist())
            self.assertListEqual(expected_filt_data, fh.data[:])

            mf = fields.IndexedStringMemField(s)
            mf.data.write(data)
            self.assertListEqual(expected_indices, mf.indices[:].tolist())
            self.assertListEqual(expected_values, mf.values[:].tolist())
            self.assertListEqual(data, mf.data[:])

            mf.apply_index(inds, in_place=True)
            self.assertListEqual(expected_filt_indices, mf.indices[:].tolist())
            self.assertListEqual(expected_filt_values, mf.values[:].tolist())
            self.assertListEqual(expected_filt_data, mf.data[:])

            b = df.create_indexed_string('bar')
            b.data.write(data)
            self.assertListEqual(expected_indices, b.indices[:].tolist())
            self.assertListEqual(expected_values, b.values[:].tolist())
            self.assertListEqual(data, b.data[:])

            mb = b.apply_index(inds)
            self.assertListEqual(expected_filt_indices, mb.indices[:].tolist())
            self.assertListEqual(expected_filt_values, mb.values[:].tolist())
            self.assertListEqual(expected_filt_data, mb.data[:])

    def test_fixed_string_apply_index(self):
        data = np.array([b'a', b'bb', b'ccc', b'dddd', b'eeee', b'fff', b'gg', b'h'], dtype='S4')
        indices = np.array([7, 0, 6, 1, 5, 2, 4, 3], dtype=np.int32)
        expected = [b'h', b'a', b'gg', b'bb', b'fff', b'ccc', b'eeee', b'dddd']
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_fixed_string('foo', 4)
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_index(indices, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_index(indices, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_index(indices)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.FixedStringMemField(s, 4)
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_index(indices, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_fixed_string('bar', 4)
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_index(indices)
            self.assertListEqual(expected, mb.data[:].tolist())

    def test_numeric_apply_index(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int32')
        indices = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)
        expected = [9, 1, 8, 2, 7, 3, 6, 4, 5]
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_numeric('foo', 'int32')
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_index(indices, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_index(indices, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_index(indices)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.NumericMemField(s, 'int32')
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_index(indices, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_numeric('bar', 'int32')
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_index(indices)
            self.assertListEqual(expected, mb.data[:].tolist())

    def test_categorical_apply_index(self):
        data = np.array([0, 1, 2, 0, 1, 2, 2, 1, 0], dtype=np.int32)
        keys = {b'a': 0, b'b': 1, b'c': 2}
        indices = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)
        expected = [0, 0, 1, 1, 2, 2, 2, 0, 1]
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_categorical('foo', 'int8', keys)
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_index(indices, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_index(indices, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_index(indices)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.CategoricalMemField(s, 'int8', keys)
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_index(indices, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_categorical('bar', 'int8', keys)
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_index(indices)
            self.assertListEqual(expected, mb.data[:].tolist())

    def test_timestamp_apply_index(self):
        from datetime import datetime as D
        from datetime import timezone
        data = [D(2020, 1, 1, tzinfo=timezone.utc), D(2021, 5, 18, tzinfo=timezone.utc), D(2950, 8, 17, tzinfo=timezone.utc), D(1840, 10, 11, tzinfo=timezone.utc),
                D(2110, 11, 1, tzinfo=timezone.utc), D(2002, 3, 3, tzinfo=timezone.utc), D(2018, 2, 28, tzinfo=timezone.utc), D(2400, 9, 1, tzinfo=timezone.utc)]
        data = np.asarray([d.timestamp() for d in data], dtype=np.float64)
        indices = np.array([7, 0, 6, 1, 5, 2, 4, 3], dtype=np.int32)
        expected = data[indices].tolist()
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_timestamp('foo', 'int32')
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            ff = f.apply_index(indices, in_place=True)
            self.assertListEqual(expected, f.data[:].tolist())
            self.assertListEqual(expected, ff.data[:].tolist())

            g = f.create_like(df, 'g')
            g.data.write(data)
            fg = f.create_like(df, 'fg')
            fgr = g.apply_index(indices, fg)
            self.assertListEqual(expected, fg.data[:].tolist())
            self.assertListEqual(expected, fgr.data[:].tolist())

            fh = g.apply_index(indices)
            self.assertListEqual(expected, fh.data[:].tolist())

            mf = fields.TimestampMemField(s)
            mf.data.write(data)
            self.assertListEqual(data.tolist(), mf.data[:].tolist())

            mf.apply_index(indices, in_place=True)
            self.assertListEqual(expected, mf.data[:].tolist())

            b = df.create_timestamp('bar')
            b.data.write(data)
            self.assertListEqual(data.tolist(), b.data[:].tolist())

            mb = b.apply_index(indices)
            self.assertListEqual(expected, mb.data[:].tolist())


class TestFieldApplySpansCount(unittest.TestCase):

    def _test_apply_spans_src(self, spans, src_data, expected, create_fn, apply_fn):
        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = create_fn(df)
            f.data.write(src_data)

            actual = apply_fn(f, spans, None)
            if actual.indexed:
                self.assertListEqual(expected, actual.data[:])
            else:
                self.assertListEqual(expected, actual.data[:].tolist())

    def test_indexed_string_apply_spans(self):
        spans = np.array([0, 2, 3, 6, 8], dtype=np.int32)
        src_data = ['a', 'bb', 'ccc', 'dddd', 'eeee', 'fff', 'gg', 'h']

        expected = ['a', 'ccc', 'dddd', 'gg']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_indexed_string('foo'),
                                   lambda f, p, d: f.apply_spans_first(p, d))

        expected = ['bb', 'ccc', 'fff', 'h']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_indexed_string('foo'),
                                   lambda f, p, d: f.apply_spans_last(p, d))

        expected = ['a', 'ccc', 'dddd', 'gg']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_indexed_string('foo'),
                                   lambda f, p, d: f.apply_spans_min(p, d))

        expected = ['bb', 'ccc', 'fff', 'h']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_indexed_string('foo'),
                                   lambda f, p, d: f.apply_spans_max(p, d))

    def test_fixed_string_apply_spans(self):
        spans = np.array([0, 2, 3, 6, 8], dtype=np.int32)
        src_data = [b'a1', b'a2', b'b1', b'c1', b'c2', b'c3', b'd1', b'd2']

        expected = [b'a1', b'b1', b'c1', b'd1']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_fixed_string('foo', 2),
                                   lambda f, p, d: f.apply_spans_first(p, d))

        expected = [b'a2', b'b1', b'c3', b'd2']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_fixed_string('foo', 2),
                                   lambda f, p, d: f.apply_spans_last(p, d))

        expected = [b'a1', b'b1', b'c1', b'd1']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_fixed_string('foo', 2),
                                   lambda f, p, d: f.apply_spans_min(p, d))

        expected = [b'a2', b'b1', b'c3', b'd2']
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_fixed_string('foo', 2),
                                   lambda f, p, d: f.apply_spans_max(p, d))

    def test_numeric_apply_spans(self):
        spans = np.array([0, 2, 3, 6, 8], dtype=np.int32)
        src_data = [1, 2, 11, 21, 22, 23, 31, 32]

        expected = [1, 11, 21, 31]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_numeric('foo', 'int32'),
                                   lambda f, p, d: f.apply_spans_first(p, d))

        expected = [2, 11, 23, 32]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_numeric('foo', 'int32'),
                                   lambda f, p, d: f.apply_spans_last(p, d))

        expected = [1, 11, 21, 31]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_numeric('foo', 'int32'),
                                   lambda f, p, d: f.apply_spans_min(p, d))

        expected = [2, 11, 23, 32]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_numeric('foo', 'int32'),
                                   lambda f, p, d: f.apply_spans_max(p, d))

    def test_categorical_apply_spans(self):
        spans = np.array([0, 2, 3, 6, 8], dtype=np.int32)
        src_data = [0, 1, 2, 0, 1, 2, 0, 1]
        keys = {b'a': 0, b'b': 1, b'c': 2}

        expected = [0, 2, 0, 0]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_categorical('foo', 'int8', keys),
                                   lambda f, p, d: f.apply_spans_first(p, d))

        expected = [1, 2, 2, 1]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_categorical('foo', 'int8', keys),
                                   lambda f, p, d: f.apply_spans_last(p, d))

        expected = [0, 2, 0, 0]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_categorical('foo', 'int8', keys),
                                   lambda f, p, d: f.apply_spans_min(p, d))

        expected = [1, 2, 2, 1]
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_categorical('foo', 'int8', keys),
                                   lambda f, p, d: f.apply_spans_max(p, d))

    def test_timestamp_apply_spans(self):
        spans = np.array([0, 2, 3, 6, 8], dtype=np.int32)
        from datetime import datetime as D
        from datetime import timezone
        src_data = [D(2020, 1, 1, tzinfo=timezone.utc), D(2021, 5, 1, tzinfo=timezone.utc), D(2950, 8, 17, tzinfo=timezone.utc), D(1840, 10, 11, tzinfo=timezone.utc),
                    D(2021, 1, 1, tzinfo=timezone.utc), D(2022, 5, 18, tzinfo=timezone.utc), D(2951, 8, 17, tzinfo=timezone.utc), D(1841, 10, 11, tzinfo=timezone.utc)]
        src_data = np.asarray([d.timestamp() for d in src_data], dtype=np.float64)

        expected = src_data[[0, 2, 3, 6]].tolist()
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_timestamp('foo'),
                                   lambda f, p, d: f.apply_spans_first(p, d))

        expected = src_data[[1, 2, 5, 7]].tolist()
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_timestamp('foo'),
                                   lambda f, p, d: f.apply_spans_last(p, d))

        expected = src_data[[0, 2, 3, 7]].tolist()
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_timestamp('foo'),
                                   lambda f, p, d: f.apply_spans_min(p, d))

        expected = src_data[[1, 2, 5, 6]].tolist()
        self._test_apply_spans_src(spans, src_data, expected,
                                   lambda df: df.create_timestamp('foo'),
                                   lambda f, p, d: f.apply_spans_max(p, d))


class TestFieldCreateLike(unittest.TestCase):

    def test_indexed_string_field_create_like(self):
        data = ['a', 'bb', 'ccc', 'ddd']

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_indexed_string('foo')
            f.data.write(data)
            self.assertListEqual(data, f.data[:])

            g = f.create_like()
            self.assertIsInstance(g, fields.IndexedStringMemField)
            self.assertEqual(0, len(g.data))

            h = f.create_like(df, "h")
            self.assertIsInstance(h, fields.IndexedStringField)
            self.assertEqual(0, len(h.data))

    def test_fixed_string_field_create_like(self):
        data = np.asarray([b'a', b'bb', b'ccc', b'dddd'], dtype='S4')

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_fixed_string('foo', 4)
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            g = f.create_like()
            self.assertIsInstance(g, fields.FixedStringMemField)
            self.assertEqual(0, len(g.data))

            h = f.create_like(df, "h")
            self.assertIsInstance(h, fields.FixedStringField)
            self.assertEqual(0, len(h.data))

    def test_numeric_field_create_like(self):
        data = np.asarray([1, 2, 3, 4], dtype=np.int32)

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_numeric('foo', 'int32')
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            g = f.create_like()
            self.assertIsInstance(g, fields.NumericMemField)
            self.assertEqual(0, len(g.data))

            h = f.create_like(df, "h")
            self.assertIsInstance(h, fields.NumericField)
            self.assertEqual(0, len(h.data))

    def test_categorical_field_create_like(self):
        data = np.asarray([0, 1, 1, 0], dtype=np.int8)
        key = {b'a': 0, b'b': 1}

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_categorical('foo', 'int8', key)
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            g = f.create_like()
            self.assertIsInstance(g, fields.CategoricalMemField)
            self.assertEqual(0, len(g.data))

            h = f.create_like(df, "h")
            self.assertIsInstance(h, fields.CategoricalField)
            self.assertEqual(0, len(h.data))

    def test_timestamp_field_create_like(self):
        from datetime import datetime as D
        from datetime import timezone
        data = [D(2020, 1, 1, tzinfo=timezone.utc), D(2021, 5, 18, tzinfo=timezone.utc), D(2950, 8, 17, tzinfo=timezone.utc), D(1840, 10, 11, tzinfo=timezone.utc)]
        data = np.asarray([d.timestamp() for d in data], dtype=np.float64)

        bio = BytesIO()
        with session.Session() as s:
            ds = s.open_dataset(bio, 'w', 'ds')
            df = ds.create_dataframe('df')
            f = df.create_timestamp('foo')
            f.data.write(data)
            self.assertListEqual(data.tolist(), f.data[:].tolist())

            g = f.create_like()
            self.assertIsInstance(g, fields.TimestampMemField)
            self.assertEqual(0, len(g.data))

            h = f.create_like(df, "h")
            self.assertIsInstance(h, fields.TimestampField)
            self.assertEqual(0, len(h.data))


class TestFieldCreateLikeWithGroups(unittest.TestCase):

    def test_indexed_string_field_create_like(self):
        data = ['a', 'bb', 'ccc', 'ddd']

        bio = BytesIO()
        with h5py.File(bio, 'w') as ds:
            with session.Session() as s:
                df = ds.create_group('df')
                f = s.create_indexed_string(df, 'foo')
                f.data.write(data)
                self.assertListEqual(data, f.data[:])

                g = f.create_like(df, "g")
                self.assertIsInstance(g, fields.IndexedStringField)
                self.assertEqual(0, len(g.data))

    def test_fixed_string_field_create_like(self):
        data = np.asarray([b'a', b'bb', b'ccc', b'dddd'], dtype='S4')

        bio = BytesIO()
        with h5py.File(bio, 'w') as ds:
            with session.Session() as s:
                df = ds.create_group('df')
                f = s.create_fixed_string(df, 'foo', 4)
                f.data.write(data)
                self.assertListEqual(data.tolist(), f.data[:].tolist())

                g = f.create_like(df, "g")
                self.assertIsInstance(g, fields.FixedStringField)
                self.assertEqual(0, len(g.data))

    def test_numeric_field_create_like(self):
        expected = [1, 2, 3, 4]
        data = np.asarray(expected, dtype=np.int32)

        bio = BytesIO()
        with h5py.File(bio, 'w') as ds:
            with session.Session() as s:
                df = ds.create_group('df')
                f = s.create_numeric(df, 'foo', 'int32')
                f.data.write(data)
                self.assertListEqual(data.tolist(), f.data[:].tolist())

                g = f.create_like(df, "g")
                self.assertIsInstance(g, fields.NumericField)
                self.assertEqual(0, len(g.data))

    def test_categorical_field_create_like(self):
        data = np.asarray([0, 1, 1, 0], dtype=np.int8)
        key = {b'a': 0, b'b': 1}

        bio = BytesIO()
        with h5py.File(bio, 'w') as ds:
            with session.Session() as s:
                df = ds.create_group('df')
                f = s.create_categorical(df, 'foo', 'int8', key)
                f.data.write(data)
                self.assertListEqual(data.tolist(), f.data[:].tolist())

                g = f.create_like(df, "g")
                self.assertIsInstance(g, fields.CategoricalField)
                self.assertEqual(0, len(g.data))
                self.assertDictEqual({0: b'a', 1: b'b'}, g.keys)

    def test_timestamp_field_create_like(self):
        from datetime import datetime as D
        from datetime import timezone
        data = [D(2020, 1, 1, tzinfo=timezone.utc), D(2021, 5, 18, tzinfo=timezone.utc), D(2950, 8, 17, tzinfo=timezone.utc), D(1840, 10, 11, tzinfo=timezone.utc)]
        data = np.asarray([d.timestamp() for d in data], dtype=np.float64)

        bio = BytesIO()
        with h5py.File(bio, 'w') as ds:
            with session.Session() as s:
                df = ds.create_group('df')
                f = s.create_timestamp(df, 'foo')
                f.data.write(data)
                self.assertListEqual(data.tolist(), f.data[:].tolist())

                g = f.create_like(df, "g")
                self.assertIsInstance(g, fields.TimestampField)
                self.assertEqual(0, len(g.data))


class TestNumericFieldAsType(unittest.TestCase):

    def test_numeric_field_astype(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            num = df.create_numeric('num', 'int16')
            num.data.write([1, 2, 3, 4, 5])

            num = num.astype('int32')
            self.assertEqual(num.data[:].dtype.type, np.int32)
            num = num.astype('int64')
            self.assertEqual(num.data[:].dtype.type, np.int64)
            num = num.astype('float32')
            self.assertEqual(num.data[:].dtype.type, np.float32)
            num = num.astype('float64')
            self.assertEqual(num.data[:].dtype.type, np.float64)
            with self.assertRaises(Exception) as context:
                num.astype('int32', casting='safe')
            self.assertTrue(isinstance(context.exception,TypeError))


class TestFieldUnique(unittest.TestCase):

    def test_unique_numeric_with_input_in_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_numeric('f', 'int16').data.write([1, 2, 3, 1, 2])

            self.assertEqual(df['f'].unique().tolist(), [1,2,3])


    def test_unique_numeric_with_input_out_of_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_numeric('f', 'int16').data.write([3, 2, 1, 2, 1])

            self.assertEqual(df['f'].unique().tolist(), [1,2,3])


    def test_unique_indexed_string_with_input_in_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_indexed_string('foo').data.write(['a','bb','bb', 'ccc', 'a', 'bb'])

            self.assertEqual(df['foo'].unique().tolist(), ['a', 'bb', 'ccc'])


    def test_unique_indexed_string_with_input_out_of_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_indexed_string('foo').data.write(['ccc','bb','a','bb'])

            self.assertEqual(df['foo'].unique().tolist(), ['a', 'bb', 'ccc'])

            
    def test_unique_indexed_string_return_index(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_indexed_string('foo').data.write(['ccc','bb','a','bb'])

            val, indices = df['foo'].unique(return_index=True)
            self.assertEqual(val.tolist(), ['a', 'bb', 'ccc'])
            self.assertEqual(indices.tolist(), [2,1,0])


    def test_unique_indexed_string_return_inverse(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_indexed_string('foo').data.write(['ccc','bb','a','bb','a'])

            val, indices = df['foo'].unique(return_inverse=True)
            self.assertEqual(val.tolist(), ['a', 'bb', 'ccc'])
            self.assertEqual(indices.tolist(), [2,1,0,1,0])


    def test_unique_indexed_string_return_counts(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_indexed_string('foo').data.write(['ccc','bb','a','bb'])

            val, counts = df['foo'].unique(return_counts=True)
            self.assertEqual(val.tolist(), ['a', 'bb', 'ccc'])
            self.assertEqual(counts.tolist(), [1,2,1])


    def test_unique_fixed_stringwith_with_input_in_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_fixed_string('foo', 2).data.write(['aa','aa','bb','cc', 'bb'])

            self.assertEqual(df['foo'].unique().tolist(), [b'aa', b'bb', b'cc'])


    def test_unique_fixed_stringwith_with_input_out_of_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            df.create_fixed_string('foo', 2).data.write(['bb','aa','cc','aa'])

            self.assertEqual(df['foo'].unique().tolist(), [b'aa', b'bb', b'cc'])


    def test_unique_categorical_field_with_input_in_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            f = df.create_categorical('f', 'int8', {'a': 0, 'c': 1, 'd': 2, 'b': 3})
            f.data.write([0, 1, 2, 2, 3, 3, 0, 2, 2, 3])
            self.assertEqual(df['f'].unique().tolist(), [0, 1, 2, 3])


    def test_unique_categorical_field_with_input_out_of_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')
            f = df.create_categorical('f', 'int8', {'a': 0, 'c': 1, 'd': 2, 'b': 3})
            f.data.write([0, 1, 3, 2, 3, 2, 0, 1, 0])
            self.assertEqual(df['f'].unique().tolist(), [0, 1, 2, 3])
        

    def test_unique_timestamp_field_with_input_in_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')

            ts1 = datetime(2021, 12, 1).timestamp()
            ts2 = datetime(2022, 1, 1).timestamp()
            df.create_timestamp('ts').data.write([ts1, ts2, ts1])

            self.assertEqual(df['ts'].unique().tolist(), [ts1, ts2])


    def test_unique_timestamp_field_with_input_out_of_order(self):
        bio = BytesIO()
        with session.Session() as s:
            src = s.open_dataset(bio, 'w', 'src')
            df = src.create_dataframe('df')

            ts1 = datetime(2021, 12, 1).timestamp()
            ts2 = datetime(2022, 1, 1).timestamp()
            df.create_timestamp('ts').data.write([ts2, ts2, ts1])

            self.assertEqual(df['ts'].unique().tolist(), [ts1, ts2])



REALLY_LARGE_LIST = list(range(1_000_000))
NUMERIC_ISIN_TESTS = [
    ("int16", [1, 2, 3, 4, 5], [], None),
    ("int16", [1, 2, 3, 4, 5], [6, 7], None),
    ("int16", [1, 2, 3, 4, 5], [1, 2, 3], None),
    ("int16", [1, 2, 3, 4, 5], [1, 2, 3, 6, 7], None),
    ("int16", [1, 2, 3, 4, 5], [4, 1, 3], None),
    ("int16", [3, 1, 5, 4, 2], [4, 1, 3], None),
    ("int16", [1, 2, 3, 4, 5], 3, None),
    ("int16", [3, 1, 5, 4, 2], 4, None),
    # really large inputs can take a long time to run through oracle to pre-compute result
    ("int32", REALLY_LARGE_LIST, REALLY_LARGE_LIST, [True] * len(REALLY_LARGE_LIST)),
    ("int32", REALLY_LARGE_LIST, shuffle_randstate(REALLY_LARGE_LIST), [True] * len(REALLY_LARGE_LIST)),
]

# test data for index string field, test conditions these define are covered to a degree already by DEFAULT_FIELD_DATA
INDEX_STR_DATA = ["a", "", "apple", "app", "APPLE", "APP", "aaaa", "app/", "apple12", "ip"]
INDEX_STR_ISIN_TESTS = [
    (INDEX_STR_DATA,[],None),
    (INDEX_STR_DATA,None,None),
    (INDEX_STR_DATA,[None],None),
    (INDEX_STR_DATA, ["None"], None),
    (INDEX_STR_DATA, ["a", "APPLE"], None),
    (INDEX_STR_DATA, ["app", "APP"], None),
    (INDEX_STR_DATA, ["app/", "app//"], None),
    (INDEX_STR_DATA, ["apple12", "APPLE12", "apple13"], None),
    (INDEX_STR_DATA, ["ip", "ipd", "id"], None),
    (INDEX_STR_DATA, [""], None),
    (INDEX_STR_DATA, INDEX_STR_DATA, [True] * len(INDEX_STR_DATA)),
]


def isin_oracle(data, isin_values, expected=None):
    """
    Generates the expected membership list of boolean values that should be returned by `f.isin(isin_values)` where `f`
    contains `data`. If `expected` is not None this is returned instead, this allows pre-defined expected values to be
    given for only some test cases so to avoid long running calls to this function.
    """
    if expected is not None:
        return expected

    if isinstance(isin_values, type(data)) and isin_values == data:
        return [True] * len(data)

    if isinstance(isin_values, (tuple, list)):
        return [d in isin_values for d in data]

    return [d == isin_values for d in data]


class TestFieldIsIn(SessionTestCase):
    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_isin_default_fields(self, creator, name, kwargs, data):
        """
        Tests `isin` for the default fields by checking with an empty input list and lists containing every value
        and every pair of values from the field data.
        """
        f = self.setup_field(self.df, creator, name, (), kwargs, data)
        if "nformat" in kwargs:
            data = np.asarray(data, dtype=kwargs["nformat"])

        with self.subTest("Test empty isin parameter"):
            expected = [False] * len(data)
            result = f.isin([])
            np.testing.assert_array_equal(expected, result)

        with self.subTest("Test 1 isin values"):
            for idx in range(len(data)):
                isin_data=[data[idx]]
                expected = isin_oracle(data, isin_data)
                result = f.isin(isin_data)
                np.testing.assert_array_equal(expected, result)

        with self.subTest("Test 2 isin values"):
            for idx1,idx2 in itertools.product(range(len(data)),repeat=2):
                isin_data=[data[idx1],data[idx2]]
                expected = isin_oracle(data, isin_data)
                result = f.isin(isin_data)
                np.testing.assert_array_equal(expected, result)


    @parameterized.expand(NUMERIC_ISIN_TESTS)
    def test_module_field_isin(self, dtype, data, isin_data, expected):
        """
        Test `isin` for the numeric fields using `fields.isin` function and the object's method.
        """
        f = self.setup_field(self.df, "create_numeric", "f", (dtype,), {}, data)

        with self.subTest("Test module function"):
            result = fields.isin(f, isin_data)
            expected = isin_oracle(data, isin_data, expected)

            self.assertIsInstance(result, fields.NumericMemField)
            self.assertFieldEqual(expected, result)

        with self.subTest("Test field method"):
            result = f.isin(isin_data)
            expected = isin_oracle(data, isin_data, expected)

            self.assertIsInstance(result, np.ndarray)
            self.assertIsInstance(expected, list)
            np.testing.assert_array_equal(expected, result)


    @parameterized.expand(INDEX_STR_ISIN_TESTS)
    def test_indexed_string_isin(self, data, isin_data, expected):
        """
        Test `isin` for the fixed string fields using `fields.isin` function and the object's method.
        """
        f = self.setup_field(self.df, "create_indexed_string", "f", (), {}, data)

        if isin_data is None:
            with self.assertRaises(TypeError) as context:
                f.isin(isin_data)

            self.assertEqual(str(context.exception), "only list-like or dict-like objects are allowed to be passed to field.isin(), you passed a 'NoneType'")

        else:

            with self.subTest("Test with given data"):
                expected = isin_oracle(data, isin_data, expected)
                result = f.isin(isin_data)
                self.assertIsInstance(result, np.ndarray)
                np.testing.assert_array_equal(expected, result)

            with self.subTest("Test with duplicate data"):
                isin_data = shuffle_randstate(isin_data * 2)  # duplicate the search items and shuffle using a fixed seed
                # reuse expected data from previous subtest
                result = f.isin(isin_data)
                self.assertIsInstance(result, np.ndarray)
                np.testing.assert_array_equal(expected, result)