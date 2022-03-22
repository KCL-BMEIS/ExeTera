import os
import unittest
from datetime import datetime, timezone

from typing import Any, List, Tuple, Dict, Optional, Union, MutableSequence
from io import BytesIO

import numpy as np

from exetera.core import session, abstract_types


ArrayLike = Union[MutableSequence, np.ndarray]

DEFAULT_SEED = 42
RAND_STATE = np.random.RandomState(DEFAULT_SEED)

TEST_TYPE_VAR = "TEST_TYPE"  # name of the global variable containing the test type, eg. "slow"

# set of difficult float values to test for as input or output, not an exhaustive list of values to test
HARD_FLOATS = [
    float("nan"),
    float("-inf"),
    float("+inf"),
    float(np.finfo(np.float32).min),
    float(np.finfo(np.float32).max),
    float(-np.finfo(np.float32).tiny),
    float(np.finfo(np.float32).tiny),
    float(np.finfo(np.float64).min),
    float(np.finfo(np.float64).max),
    float(-np.finfo(np.float64).tiny),
    float(np.finfo(np.float64).tiny),
    float(np.finfo(np.float64).min),
    float(np.finfo(np.float64).max),
    float(-np.finfo(np.float64).tiny),
    float(np.finfo(np.float64).tiny),
]

# set of difficult int values to test for as input or output, not an exhaustive list of values to test
HARD_INTS = [
    0,
    int(np.iinfo(np.int8).min),
    int(np.iinfo(np.int8).max),
    int(np.iinfo(np.int16).min),
    int(np.iinfo(np.int16).max),
    int(np.iinfo(np.int32).min),
    int(np.iinfo(np.int32).max),
    int(np.iinfo(np.uint8).max),
    int(np.iinfo(np.uint16).max),
    int(np.iinfo(np.uint32).max),
    int(np.iinfo(np.int64).min),
    int(np.iinfo(np.int64).max),
    # int(np.iinfo(np.uint64).max)  # not supported by HDF5?
]


def utc_timestamp(year, month, day, hour=0, minute=0, second=0, microsecond=0):
    """
    Returns the timestamp for a UTC datetime value constructed from the given arguments.
    """
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc).timestamp()


def shuffle_randstate(arr: ArrayLike, seed=DEFAULT_SEED) -> ArrayLike:
    """
    Shuffles `arr` based on a random state using seed value `seed`, then returns the shuffled array.
    """
    rs = np.random.RandomState(seed)
    rs.shuffle(arr)
    return arr


# default field initialization values for every field type, format is:
# (creator method, field name, args for method, kwargs for method, data)
DEFAULT_FIELD_DATA = [
    ("create_numeric", "f_i8", {"nformat": "int8"}, shuffle_randstate(list(range(-10, 10)) + HARD_INTS)),
    (
        "create_numeric",
        "f_i32",
        {"nformat": "int32"},
        shuffle_randstate(list(range(-10, 10)) + HARD_INTS),
    ),
    (
        "create_numeric",
        "f_i64",
        {"nformat": "int64"},
        shuffle_randstate(list(range(-10, 10)) + HARD_INTS),
    ),
    ("create_numeric", "f_f32", {"nformat": "float32"}, shuffle_randstate(list(range(-10, 10)) + HARD_FLOATS)),
    ("create_numeric", "f_f64", {"nformat": "float64"}, shuffle_randstate(list(range(-10, 10)) + HARD_FLOATS)),
    (
        "create_categorical",
        "f_cat123",
        {"nformat": "int8", "key": {"a": 1, "b": 2, "c": 3}},
        RAND_STATE.randint(1, 4, 20).tolist(),
    ),
    ("create_indexed_string", "f_istr", {}, ["a", "bb", "eeeee", "ccc", "dddd","", " ",]*2),
    ("create_fixed_string", "f_fstr", {"length": 3}, [b"aaa", b"bbb", b"eee", b"ccc", b"ddd", b"   "]*2),
    (
        "create_timestamp",
        "f_ts",
        {},
        [
            utc_timestamp(2020, 1, 1),
            utc_timestamp(2021, 5, 18),
            utc_timestamp(2950, 8, 17),
            utc_timestamp(1840, 10, 11),
            utc_timestamp(2110, 11, 1),
            utc_timestamp(2002, 3, 3),
            utc_timestamp(1963, 6, 7),
            utc_timestamp(2018, 2, 28),
            utc_timestamp(2400, 9, 1),
            utc_timestamp(1, 1, 1),
        ],
    ),
]


def allow_slow_tests():
    """
    Returns True if slow tests are allowed, that is TEST_TYPE_VAR global variable's value is "slow".
    """
    return os.environ.get(TEST_TYPE_VAR, "").lower() == "slow"


def slow_test(obj):
    """
    This will cause decorated tests to be skipped if the TEST_TYPE_VAR environment variable is not the value "slow".
    """
    slow_tests = allow_slow_tests()
    wrapper = unittest.skipIf(not slow_tests, "Skipping slow tests")

    return wrapper(obj)


class SessionTestCase(unittest.TestCase):
    """
    Test case subclass for testing anything needing a Session object. It will create in `setUp` a session called `s`,
    a dataset called `ds` writing into a BytesIO object `bio`, and an empty dataframe `df`. The `setup_dataframe`
    method will be called passing in `df` as the argument to allow subclasses to define a default initialisation. The
    session will be closed by `tearDown`.
    """

    def setUp(self):
        """
        Creates the Session object `self.s`, dataset object `self.ds` writing nto `self.bio`, and dataframe `self.df`.
        It then calls `self.setup_dataframe` with `self.df` as the argument.
        """
        self.bio = BytesIO()
        self.s = session.Session()
        self.ds = self.s.open_dataset(self.bio, "w", "dst")
        self.df = self.ds.create_dataframe("df")

        self.setup_dataframe(self.df)

    def tearDown(self):
        """
        Closes `self.s`.
        """
        self.s.close()

    def setup_dataframe(self, df):
        """
        Setup the dataframe `df` by creating and filling in its fields.
        """
        pass

    def setup_field(
        self,
        df: abstract_types.DataFrame,
        create_method: str,
        name: str,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        data: ArrayLike,
    ) -> abstract_types.Field:
        """
        Creates a new field for dataframe `df` using a method of `df` named by `create_method`. This field is then
        filled with `data` and returned.

        :param df: dataframe to add the field to
        :param create_method: name of method of `df` to call to create the new field
        :param args: positional arguments for creating method
        :param kwargs: keyword arguments for creating method
        :param data: data to fill into the field
        :return: the newly created field
        """
        creator = getattr(df, create_method)
        field = creator(name, *args, **kwargs)
        field.data.write(data)

        return field

    def assertFieldEqual(self, data: ArrayLike, field: abstract_types.Field, msg: str = "Error comparing field values"):
        """
        Asserts that `field` has contents equal to `data`, raising an exception with message `msg` if not.
        """
        self.assertIsInstance(field, abstract_types.Field)

        fdata = field.data[:]
        if hasattr(fdata, "tolist"):
            fdata = fdata.tolist()

        np.testing.assert_array_equal(data, fdata, msg, True)

    def assertFieldAlmostEqual(
        self,
        data: ArrayLike,
        field: abstract_types.Field,
        msg: str = "Error comparing field values",
        rtol=1e-07,
        atol=0,
        equal_nan=True,
    ):
        """
        Asserts that `field` has contents near-equal to `data`, raising an exception with message `msg` if not.
        """
        self.assertIsInstance(field, abstract_types.Field)

        fdata = field.data[:]
        if hasattr(fdata, "tolist"):
            fdata = fdata.tolist()

        np.testing.assert_allclose(data, fdata, rtol, atol, equal_nan, msg, True)
