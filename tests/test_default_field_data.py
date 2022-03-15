
from itertools import product

from parameterized import parameterized

from exetera.core import fields

import numpy as np

from .utils import SessionTestCase, shuffle_randstate, allow_slow_tests, DEFAULT_FIELD_DATA

NUMERIC_ONLY = [d for d in DEFAULT_FIELD_DATA if d[0] == "create_numeric"]


class TestDefaultData(SessionTestCase):
    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_fields(self, creator, name, kwargs, data):
        """
        Tests basic creation of every field type, checking it's contents are actually what was put into them.
        """
        f = self.setup_field(self.df, creator, name, (), kwargs, data)
        self.assertFieldEqual(data, f)


# replaces TestFieldArray in test_fields.py
class TestFieldArray(SessionTestCase):
    @parameterized.expand(NUMERIC_ONLY)
    def test_write_part(self, creator, name, kwargs, data):
        """
        Checks that `write_part` will write the data into each field type.
        """
        f = self.s.create_numeric(self.df, name, **kwargs)
        f.data.write_part(data)
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


# test data for checking isin for numeric field types, includes out-of-order data not present in DEFAULT_FIELD_DATA
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
    # (INDEX_STR_DATA,[],None), # ERROR: raises exception rather than return all Falses
    # (INDEX_STR_DATA,None,None), # ERROR: raises exception from too far down call stack
    # (INDEX_STR_DATA,[None],None), # ERROR: raises exception from too far down call stack
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

    if isin_values == data:
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

        with self.subTest("Test empty isin parameter"):
            expected = [False] * len(data)
            result = f.isin([])
            np.testing.assert_array_equal(expected, result)
            
        with self.subTest("Test 1 and 2 isin values"):
            for idx in range(len(data)):
                isin_data=[data[idx]]
                expected = isin_oracle(data, isin_data)
                result = f.isin(isin_data)
                np.testing.assert_array_equal(expected, result)
                
        with self.subTest("Test 1 and 2 isin values"):
            for idx1,idx2 in product(range(len(data)),repeat=2):
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

        with self.subTest("Test with given data"):
            expected = isin_oracle(data, isin_data, expected)
            result = f.isin(isin_data)

            self.assertIsInstance(result, list)
            self.assertEqual(expected, result)

        with self.subTest("Test with duplicate data"):
            isin_data = shuffle_randstate(isin_data * 2)  # duplicate the search items and shuffle using a fixed seed
            # reuse expected data from previous subtest
            result = f.isin(isin_data)

            self.assertIsInstance(result, list)
            self.assertEqual(expected, result)
