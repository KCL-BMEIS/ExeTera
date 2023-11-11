# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from collections import defaultdict

from datetime import datetime

import numpy as np
import numba
from numba import njit
from ctypes import sizeof, c_float, c_double, c_int8, c_uint8, c_int16, c_uint16, c_int32, c_uint32, c_int64

from codecs import BOM_UTF8, BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE


SECONDS_PER_DAY = 86400
PERMITTED_NUMERIC_TYPES = ('float32', 'float64', 'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64')
INT64_INDEX_LENGTH = 2**31-1


# environment variable used to toggle Numba off for testing
USE_NUMBA_VAR = "USE_NUMBA"
USE_NUMBA = os.environ.get(USE_NUMBA_VAR, "true").lower() == "true"


# if the above environment variable is set to true or is unset use Numba's njit, otherwise define a no-op decorator
if USE_NUMBA:
    exetera_njit = njit
else:
    def exetera_njit(func, *_, **__):
        return func
    

numba_bool = numba.types.boolean if USE_NUMBA else bool


def validate_file_exists(file_name):
    import os
    if not os.path.exists(file_name):
        raise FileExistsError(f"{file_name} doesn't exist'")
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"{file_name} is not a file'")


def find_longest_sequence_of(string, char):
    longest = 0
    current = 0
    for s in string:
        if s == char:
            current += 1
        else:
            if current > longest:
                longest = current
            current = 0
    if current > longest:
        longest = current
    return longest


def count_flag_empty(flags):
    count = 0
    for f in flags:
        if f == 0:
            count += 1
    return count


def count_flag_not_set(flags, flag_to_test):
    count = 0
    for f in flags:
        if not (f & flag_to_test):
            count += 1
    return count


def count_flag_set(flags, flag_to_test):
    count = 0
    for f in flags:
        if f & flag_to_test:
            count += 1
    return count


def timestamp_to_day(field):
    if field == '':
        return ''
    return f'{field[0:4]}-{field[5:7]}-{field[8:10]}'


def string_to_datetime(field):
    try:
        ts = datetime.strptime(field, '%Y-%m-%d %H:%M:%S.%f%z')
    except ValueError:
        try:
            ts = datetime.strptime(field, '%Y-%m-%d %H:%M:%S%z')
        except ValueError:
            ts = datetime.strptime(field, '%Y-%m-%d')

    return ts


def build_histogram(dataset, filtered_records=None, tx=None):
    # TODO: memory_efficiency: see build_histogram function
    histogram = defaultdict(int)
    for ir, r in enumerate(dataset):
        if not filtered_records or not filtered_records[ir]:
            if tx is not None:
                value = tx(r)
            else:
                value = r
            histogram[value] += 1
    hlist = list(histogram.items())
    del histogram
    return hlist


def filter_field(fields, filter_list, f_missing, f_bad, is_type_fn, type_fn, valid_fn):
    for ir, r in enumerate(fields):
        if not is_type_fn(r):
            if f_missing != 0:
                filter_list[ir] |= f_missing
        else:
            value = type_fn(r)
            if not valid_fn(value):
                if f_bad != 0:
                    filter_list[ir] |= f_bad


def map_between_categories(first_map, second_map):
    result_map = dict()
    for m in first_map.keys():
        result_map[first_map[m]] = second_map[m]
    return result_map


def check_input_lengths(names, fields):
    assert(len(names) == len(fields))
    assert(isinstance(names, tuple))
    assert(isinstance(fields, tuple))

    length = None
    error = False
    for f in fields:
        if not length:
            length = len(f)
        elif length != len(f):
            error = True

    if error:
        field_name_str = ','.join([f"'{n}'" for n in names])
        length_str = ','.join([len(f) for f in fields])
        raise ValueError(f"Collections {field_name_str} have inconsistent lengths: ({length_str})")


def datetime_to_seconds(dt):
    return f'{dt[0:4]}-{dt[5:7]}-{dt[8:10]} {dt[11:13]}:{dt[14:16]}:{dt[17:19]}'


class Timer:
    def __init__(self, start_msg, new_line=False, end_msg='completed in'):
        print(start_msg, end=': ' if new_line is False else '\n')
        self.end_msg = end_msg

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.end_msg + f' {time.time() - self.t0} seconds')


def get_min_max(value_type):
    mapping = {'float32': c_float, 'float64': c_double, 'int8': c_int8, 'uint8': c_uint8, 
                                    'int16': c_int16, 'uint16': c_uint16, 'int32': c_int32, 'uint32': c_uint32, 'int64': c_int64}
    c_type = mapping[value_type]

    signed = c_type(-1).value < c_type(0).value
    bit_size = sizeof(c_type) * 8
    signed_limit = 2 ** (bit_size - 1)
    return (-signed_limit, signed_limit - 1) if signed else (0, 2 * signed_limit - 1)


def one_dim_data_to_indexed_for_test(data, field_size):
    data = [str(s) for s in data]
    count_row = len(data)
    chunk_row_size = count_row

    indices = np.zeros((1, count_row + 1), dtype = np.int64)
    offsets = np.array([0, field_size], dtype=np.int64) * chunk_row_size
    values = np.zeros(offsets[-1], dtype = np.uint8)

    accumulated = 0
    for i, s in enumerate(data):
        length = 0
        for j, c in enumerate(s):
            encoded_c = np.frombuffer(c.encode(), dtype =np.uint8)
            for e in encoded_c:
                values[accumulated] = e
                accumulated += 1
                length += 1
        indices[0, i + 1] = indices[0, i] + length
         
    return indices, values, offsets, count_row


def guess_encoding(filename):
    """
    Attempt to determine the encodig of the given text file by reading the byte order mark, defaulting to utf-8 if 
    none is found.
    
    :param filename: path to a text file containing possible UTF-8, UTF-16, or UTF-32 text
    :return: encoding name, one of utf-8, utf-8-sig, utf-16, utf-32
    """
    with open(filename,"rb") as o:
        dat=o.read(4)
        
    if BOM_UTF32_BE in dat or BOM_UTF32_LE in dat:
        return "utf-32"
    elif BOM_UTF16_BE in dat or BOM_UTF16_LE in dat:
        return "utf-16"
    elif BOM_UTF8 in dat:
        return "utf-8-sig"
    else:
        return "utf-8"

def is_sorted(array):
    """
    Check if an array is ordered.
    """
    if len(array) < 2:
        return True
    return np.all(array[:-1] <= array[1:])