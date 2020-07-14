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

import time
from collections import defaultdict
import csv
from datetime import datetime
from io import StringIO

import numpy as np
from numba import njit


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

def timestamp_to_datetime(field):
    try:
        ts = datetime.strptime(field, '%Y-%m-%d %H:%M:%S.%f%z')
    except ValueError:
        try:
            ts = datetime.strptime(field, '%Y-%m-%d %H:%M:%S%z')
        except ValueError:
            ts = datetime.strptime(field, '%y-%m-%d')

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


def to_categorical(field, transform):
    results = np.zeros_like(field, dtype=field.dtype)
    for ir, r in enumerate(field):
        results[ir] = transform[r]
    return results


def print_diagnostic_row(preamble, ds, ir, keys, fns=None):
    if fns is None:
        fns = dict()
    # indices = [ds.field_to_index(k) for k in keys]
    # indexed_fns = [None if k not in fns else fns[k] for k in keys]
    values = [None] * len(keys)
    # for ii, i in enumerate(indices):
    for i, k in enumerate(keys):
        if not fns or k not in fns:
            values[i] = ds.value_from_fieldname(ir, k)
        else:
            # values[i] = fns[k](ds.value_from_fieldname(ir, k))
            values[i] = ds.value_from_fieldname(ir, k)
        # if indexed_fns[ii] is None:
        #     values[ii] = fields[ir][i]
        # else:
        #     values[ii] = indexed_fns[ii](fields[ir][i])
    print(f'{preamble}: {values}')


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


def valid_range_fac(f_min, f_max, default_value=''):
    def inner_(x):
        return x == default_value or x > f_min and x < f_max
    return inner_


def valid_range_fac_inc(f_min, f_max, default_value=''):
    def inner_(x):
        return x == default_value or x >= f_min and x <= f_max
    return inner_


def datetime_to_seconds(dt):
    return f'{dt[0:4]}-{dt[5:7]}-{dt[8:10]} {dt[11:13]}:{dt[14:16]}:{dt[17:19]}'


def is_int(value):
    try:
        int(float(value))
        return True
    except:
        return False


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False


def to_int(value):
    try:
        fvalue = float(value)
    except ValueError as e:
        raise ValueError(f'{value} cannot be converted to float')

    try:
        ivalue = int(fvalue)
    except ValueError as e:
        raise ValueError(f'{fvalue} cannot be converted to int')

    return ivalue


def to_float(value):
    try:
        fvalue = float(value)
    except ValueError as e:
        raise ValueError(f'{value} cannot be converted to float')

    return fvalue


def list_to_escaped(strings):
    s = StringIO()
    w = csv.writer(s)
    w.writerow(strings)
    return s.getvalue()


def to_escaped(string, separator=',', delimiter='"'):
    comma = False
    quotes = False
    for c in string:
        if c == separator:
            comma = True
        elif c == delimiter:
            quotes = True

    if comma or quotes:
        s = StringIO()
        s.write(delimiter)
        for c in string:
            if c == delimiter:
                s.write(c)
            s.write(c)
        s.write(delimiter)
        return s.getvalue()
    else:
        return string

@njit
def bytearray_to_escaped(srcbytearray, destbytearray,
                         src_start=np.int64(0), src_end=None, dest_start=np.int64(0),
                         separator=b',', delimiter=b'"'):
    if src_end is None:
        src_end = np.int64(len(srcbytearray))

    comma = False
    quotes = False
    for i_c in range(src_start, src_end):
        c = srcbytearray[i_c]
        if c == separator:
            comma = True
        elif c == delimiter:
            quotes = True

    d_index = 0
    if comma or quotes:
        destbytearray[d_index] = delimiter
        d_index += 1
        for i_c in range(src_start, src_end):
            c = srcbytearray[i_c]
            if c == delimiter:
                destbytearray[d_index] = c
                d_index += 1
            destbytearray[d_index] = c
            d_index += 1
        destbytearray[d_index] = delimiter
        d_index += 1
        return d_index
    else:
        s_len = np.int64(src_end - src_start)
        destbytearray[dest_start:dest_start + s_len] = srcbytearray[src_start:src_end]
        d_index += s_len
        return d_index


def from_escaped(string):
    s = StringIO(string)
    r = csv.reader(s)
    return next(r)


def concatenate_maybe_strs(sequence, value, separator=',', delimiter='"'):
    if sequence in (None, '', 'NA'):
        if value in (None, '', 'NA'):
            return ''
        else:
            return to_escaped(value)
    else:
        if value in (None, '', 'NA'):
            return sequence
        else:
            return f"{sequence}{separator}{to_escaped(value)}"


def replace_if_invalid(replacement):
    def inner_(value):
        if value is '':
            return replacement
        else:
            return float(value)
    return inner_


def clear_set_flag(values, to_clear):
    for v in range(len(values)):
        values[v] &= ~to_clear
    return values


def sort_mixed_list(values, check_fn, sort_fn):
    # pass to find the single entry that fails check_fn
    for iv in range(len(values)):
        checked_item = None
        if not check_fn(values[iv]):
            #swap the current item with the last if it isn't last
            found_checked_item = True
            if iv != len(values) - 1:
                values[iv], values[-1] = values[-1], values[iv]
                checked_item = values.pop()
        break

    list.sort(values, key=sort_fn)
    if found_checked_item:
        values.append(checked_item)

    return values


class Timer:
    def __init__(self, start_msg, new_line=False, end_msg='completed in'):
        print(start_msg, end=': ' if new_line is False else '\n')
        self.end_msg = end_msg

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.end_msg + f' {time.time() - self.t0} seconds')
