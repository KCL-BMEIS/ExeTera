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
import uuid
import types
from contextlib import contextmanager
from datetime import datetime, timezone
import time
from io import BytesIO

import h5py
import numpy as np
from numba import jit, njit

import utils

DEFAULT_CHUNKSIZE = 1 << 18

# TODO: rename this persistence file to hdf5persistence
# TODO: wrap the dataset in a withable so that different underlying
# data stores can be used

# schema
# * schema
#   * import history
#   * schema number
# * patients
# * assessments
# * tests

# groups
# * datetime
#   * year
#   * month
#   * day
#   * hour
#   * minute
#   * second
#   * microsecond

# field
# * key
#   * applied sort
#   * applied filter
# * category_names
# * values

chunk_sizes = {
    'patient': (1 << 18,), 'assessment': (1 << 18,), 'test': (1 << 18,)
}


class DataWriter:

    @staticmethod
    def _write(group, name, field, count, dtype=None):
        if name not in group.keys():
            DataWriter._write_first(group, name, field, count, dtype)
        else:
            DataWriter._write_additional(group, name, field, count)

    @staticmethod
    def _write_first(group, name, field, count, dtype=None):
        if dtype is not None:
            if count == len(field):
                ds = group.create_dataset(
                    name, (count,), maxshape=(None,), dtype=dtype)
                ds[:] = field
            else:
                ds = group.create_dataset(
                    name, (count,), maxshape=(None,), dtype=dtype)
                ds[:] = field[:count]
        else:
            if count == len(field):
                group.create_dataset(name, (count,), maxshape=(None,), data=field)
            else:
                group.create_dataset(name, (count,), maxshape=(None,), data=field[:count])

    @staticmethod
    def _write_additional(group, name, field, count):
        gv = group[name]
        gv.resize((gv.size + count,))
        if count == len(field):
            gv[-count:] = field
        else:
            gv[-count:] = field[:count]


# def str_to_float(value):
#     try:
#         return float(value)
#     except ValueError:
#         return None


# def str_to_int(value):
#     try:
#         return int(value)
#     except ValueError:
#         return None

def try_str_to_float_to_int(value, invalid=0):
    try:
        v = int(float(value))
        return True, v
    except ValueError:
        return False, invalid


def try_str_to_int(value, invalid=0):
    try:
        v = int(value)
        return True, v
    except ValueError:
        return False, invalid


def try_str_to_float(value, invalid=0):
    try:
        v = float(value)
        return True, v
    except ValueError:
        return False, invalid


# Readers
# =======


# def _chunkcount(dataset, chunksize, istart=0, iend=None):
#     if iend is None:
#         iend = dataset.size
#     requested_size = iend - istart
#     chunkmax = int(requested_size / chunksize)
#     if requested_size % chunksize != 0:
#         chunkmax += 1
#     return chunkmax


# def _slice_for_chunk(c, dataset, chunksize, istart=0, iend=None):
#     if iend is None:
#         iend = len(dataset)
#     requested_size = iend - istart
#     # if c == chunkmax - 1:
#     if c >= _chunkcount(dataset, chunksize, istart, iend):
#         raise ValueError("Asking for out of range chunk")
#
#     if istart + (c + 1) * chunksize> iend:
#         length = requested_size % chunksize
#     else:
#         length = chunksize
#     return istart + c * chunksize, istart + c * chunksize + length


def dataset_sort(index, readers):
    r_readers = reversed(readers)

    acc_index = index[:]
    first = True
    for f in r_readers:
        if first:
            first = False
            fdata = f[:]
        else:
            fdata = f[:][acc_index]

        index = np.argsort(fdata, kind='stable')
        acc_index = acc_index[index]
    return acc_index


def apply_sort_to_array(index, values):
    return values[index]


from numba import njit
@njit
def apply_sort_to_index_values(index, indices, values):

    s_indices = np.zeros_like(indices)
    s_values = np.zeros_like(values)
    accumulated = np.uint64(0)
    s_indices[0] = 0
    for di, si in enumerate(index):
        src_field_start = indices[si]
        src_field_end = indices[si + 1]
        length = np.uint64(src_field_end - src_field_start)
        if length > 0:
            s_values[accumulated:accumulated + length] =\
                values[src_field_start:src_field_end]
        accumulated += length
        if s_indices[di + 1] != 0:
            print('non-zero index!')
        s_indices[di + 1] = accumulated

    return s_indices, s_values


def apply_sort(index, reader, writer):
    if isinstance(reader, NewIndexedStringReader):
        src_indices = reader.field['index'][:]
        src_values = reader.field.get('values', np.zeros(0, dtype='S1'))[:]
        indices, values = apply_sort_to_index_values(index, src_indices, src_values)
        writer.write_raw(indices, values)
    else:
        result = apply_sort_to_array(index, reader[:])
        writer.write(result)

# TODO: merge implementation may still be required in the medium term
def dataset_merge_sort(group, index, fields):
    raise NotImplementedError()
    # def sort_comparison(*args):
    #     if len(args) == 1:
    #         a0 = args[0]
    #         def _inner(r):
    #             return a0[r]
    #         return _inner
    #     if len(args) == 2:
    #         a0 = args[0]
    #         a1 = args[1]
    #         def _inner(r):
    #             return a0[r], a1[r]
    #         return _inner
    #     if len(args) == 3:
    #         a0 = args[0]
    #         a1 = args[1]
    #         a2 = args[2]
    #         def _inner(r):
    #             return a0[r], a1[r], a2[r]
    #         return _inner
    #     if len(args) > 3:
    #         def _inner(r):
    #             return tuple(a[r] for a in args)
    #         return _inner
    #
    # def sort_function(index, fields):
    #     sort_group = temp_dataset()
    #
    #     # sort each chunk individually
    #     chunksize = 1 << 24
    #     chunkcount = _chunkcount(index, chunksize)
    #     for c in range(chunkcount):
    #         istart, iend = _slice_for_chunk(c, index, chunksize)
    #         length = iend - istart
    #         fieldchunks = [None] * len(fields)
    #         indexchunk = index[istart:iend]
    #         for i_f, f in enumerate(fields):
    #             fc = reader(f, istart, iend)
    #             fieldchunks[i_f] = fc
    #         sfn = sort_comparison(*fieldchunks)
    #         sindexchunk = sorted(indexchunk, key=sfn)
    #         sort_group.create_dataset(f'chunk{c}', (length,), data=sindexchunk)
    #
    # sort_function(index, fields)


def temp_filename():
    uid = str(uuid.uuid4())
    while os.path.exists(uid + '.hdf5'):
        uid = str(uuid.uuid4())
    return uid + '.hdf5'


@contextmanager
def temp_dataset():
    try:
        uid = str(uuid.uuid4())
        while os.path.exists(uid + '.hdf5'):
            uid = str(uuid.uuid4())
        hd = h5py.File(uid, 'w')
        yield hd
    finally:
        hd.flush()
        hd.close()


# TODO: write filter with new readers / writers rather than deleting this
def filter(dataset, field, name, predicate, timestamp=datetime.now(timezone.utc)):
    raise NotImplementedError()

#     c = Series(field)
#     writer = BooleanWriter(dataset, DEFAULT_CHUNKSIZE, name, timestamp)
#     for r in c:
#         writer.append(predicate(r))
#     writer.flush()
#     return dataset[name]


# TODO: write distinct with new readers / writers rather than deleting this
def distinct(field=None, fields=None, filter=None):
    if field is None and fields is None:
        return ValueError("One of 'field' and 'fields' must be set")
    if field is not None and fields is not None:
        return ValueError("Only one of 'field' and 'fields' may be set")

    if field is not None:
        return np.unique(field)

    entries = [(f'{i}', f.dtype) for i, f in enumerate(fields)]
    unified = np.empty_like(fields[0], dtype=np.dtype(entries))
    for i, f in enumerate(fields):
        unified[f'{i}'] = f

    uniques = np.unique(unified)
    results = [uniques[f'{i}'] for i in range(len(fields))]
    return results

# @njit
# def _not_equals(a, c):
#     a_len = len(a)
#     a0 = a[:-1]
#     a1 = a[1:]
#     c0 = c[1:-1]
#     for i_r in range(a_len):
#         c0[i_r] = a0[i_r] == a1[i_r]

@njit
def _not_equals(a, b, c):
    a_len = len(a)
    for i_r in range(a_len):
        c[i_r] = a[i_r] != b[i_r]

def _get_spans_for_field(field0):
    # count = 0
    # spans = np.empty(len(field0)+1, dtype=np.uint32)
    # spans[0] = 0
    # field_count = len(field0)
    # for i in range(1, field_count):
    #     if field0[i] != field0[i-1]:
    #         count += 1
    #         spans[count] = i
    # spans[count+1] = len(field0)
    # return spans[:count+2]


    results = np.zeros(len(field0) + 1, dtype=np.bool)
    # _not_equals(field0, results)
    t0 = time.time()
    a = field0[:-1]
    print(f"    {time.time() - t0}")
    t0 = time.time()
    b = field0[1:]
    print(f"    {time.time() - t0}")
    t0 = time.time()
    _not_equals(field0[:-1], field0[1:], results[1:])
    print(f"    {time.time() - t0}")
    results[0] = True
    results[-1] = True
    return np.nonzero(results)[0]
    # return results

# def find_runs(x):
#     """Find runs of consecutive items in an array."""
#
#     # ensure array
#     x = np.asanyarray(x)
#     if x.ndim != 1:
#         raise ValueError('only 1D array supported')
#     n = x.shape[0]
#
#     # handle empty array
#     if n == 0:
#         return np.array([]), np.array([]), np.array([])
#
#     else:
#         # find run starts
#         loc_run_start = np.empty(n, dtype=bool)
#         loc_run_start[0] = True
#         np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
#         run_starts = np.nonzero(loc_run_start)[0]
#
#         # find run values
#         run_values = x[loc_run_start]
#
#         # find run lengths
#         run_lengths = np.diff(np.append(run_starts, n))
#
#         return run_values, run_starts, run_lengths

@njit
def _get_spans_for_2_fields(field0, field1):
    count = 0
    spans = np.zeros(len(field0)+1, dtype=np.uint32)
    spans[0] = 0
    for i in np.arange(1, len(field0)):
        if field0[i] != field0[i-1] or field1[i] != field1[i-1]:
            count += 1
            spans[count] = i
    spans[count+1] = len(field0)
    return spans[:count+2]


def get_spans(field=None, fields=None):
    if field is None and fields is None:
        return ValueError("One of 'field' and 'fields' must be set")
    if field is not None and fields is not None:
        return ValueError("Only one of 'field' and 'fields' may be set")

    if field is not None:
        #return _get_spans_for_field(field)
        return _get_spans_for_field(field)
    elif len(fields) == 1:
        return _get_spans_for_field(fields[0])
    elif len(fields) == 2:
        return _get_spans_for_2_fields(*fields)
    else:
        raise NotImplementedError("This operation does not support more than two fields at present")

@njit
def _apply_spans_count(spans, dest_array):
    for i in range(len(spans)-1):
        dest_array[i] = np.uint64(spans[i+1] - spans[i])
        # if spans[i+1] - spans[i] < 0:
        #     print(spans[i+1] - spans[i], spans[i+1], spans[i], i)
    print(dest_array.max())


def apply_spans_count(spans, writer):
    if isinstance(writer, NewWriter):
        dest_values = writer.chunk_factory(len(spans) - 1)
        _apply_spans_count(spans, dest_values)
        writer.write(dest_values)
    elif isinstance(writer, np.ndarray):
        _apply_spans_count(spans, writer)
    else:
        raise ValueError(f"'writer' must be one of 'NewWriter' or 'ndarray' but is {type(writer)}")


@njit
def _apply_spans_first(spans, src_array, dest_array):
    dest_array[:] = src_array[spans[:-1]]


def apply_spans_first(spans, reader, writer):
    dest_values = writer.chunk_factory(len(spans) - 1)
    _apply_spans_first(spans, reader[:], dest_values)
    writer.write(dest_values)


@njit
def _apply_spans_last(spans, src_array, dest_array):
    spans = spans[1:]-1
    dest_array[:] = src_array[spans]


def apply_spans_last(spans, reader, writer):
    dest_values = writer.chunk_factory(len(spans) - 1)
    _apply_spans_last(spans, reader[:], dest_values)
    writer.write(dest_values)

@njit
def _apply_spans_max(spans, src_array, dest_array):

    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]
        if next - cur == 1:
            dest_array[i] = src_array[cur]
        else:
            dest_array[i] = src_array[cur:next].max()


def apply_spans_max(spans, reader, writer):
    dest_values = writer.chunk_factory(len(spans) - 1)
    _apply_spans_max(spans, reader[:], dest_values)
    writer.write(dest_values)


def _apply_spans_concat(spans, src_field):
    dest_values = [None] * (len(spans)-1)
    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]
        if next - cur == 1:
            dest_values[i] = src_field[cur]
        else:
            src = [s for s in src_field[cur:next] if len(s) > 0]
            if len(src) > 0:
                dest_values[i] = ','.join(utils.to_escaped(src))
            else:
                dest_values[i] = ''
            # if len(dest_values[i]) > 0:
            #     print(dest_values[i])
    return dest_values

# 0, 2, 3, 4, 6, 8
# 0, 2, 6, 10, 12, 16, 18, 22, 24
# aa bbbb cccc dd eeee ff gggghh

# 0, 6, 10, 12, 18, 24
# aabbbb cccc dd eeeeff gggghh

@njit
def _apply_spans_concat(spans, src_index, src_values, dest_index, dest_values,
                        max_index_i, max_value_i, s_start):#, separator, delimiter):
    separator = np.frombuffer(b',', dtype=np.uint8)[0]
    delimiter = np.frombuffer(b'"', dtype=np.uint8)[0]
    if s_start == 0:
        index_i = np.uint32(1)
        index_v = np.uint64(0)
        dest_index[0] = spans[0]
    else:
        index_i = np.uint32(0)
        index_v = np.uint64(0)

    s_end = len(spans)-1
    for s in range(s_start, s_end):
        cur = spans[s]
        next = spans[s+1]
        cur_src_i = src_index[cur]
        next_src_i = src_index[next]

        dest_index[index_i] = next_src_i
        index_i += 1

        if next_src_i - cur_src_i > 1:
            if next - cur == 1:
                # only one entry to be copied, so commas not required
                next_index_v = next_src_i - cur_src_i + np.uint64(index_v)
                dest_values[index_v:next_index_v] = src_values[cur_src_i:next_src_i]
                index_v = next_index_v
            else:
                # check to see how many non-zero-length entries there are; >1 means we must
                # separate them by commas
                non_empties = 0
                for e in range(cur, next):
                   if src_index[e] < src_index[e+1]:
                       non_empties += 1
                if non_empties == 1:
                    # only one non-empty entry to be copied, so commas not required
                    next_index_v = next_src_i - cur_src_i + np.uint64(index_v)
                    dest_values[index_v:next_index_v] = src_values[cur_src_i:next_src_i]
                    index_v = next_index_v
                else:
                    # the outer conditional already determines that we have a non-empty entry
                    # so there must be multiple non-empty entries and commas are required
                    for e in range(cur, next):
                        # separator = b','
                        # delimiter = b'"'
                        # delta = utils.bytearray_to_escaped(src_values,
                        #                                    dest_values,
                        #                                    src_start=src_index[e],
                        #                                    src_end=src_index[e+1],
                        #                                    dest_start=index_v)
                        # index_v += np.uint64(d_index)
                        src_start = src_index[e]
                        src_end = src_index[e+1]
                        comma = False
                        quotes = False
                        for i_c in range(src_start, src_end):
                            # c = src_values[i_c]
                            # if c == separator[0]:
                            if src_values[i_c] == separator:
                                comma = True
                            # elif c == delimiter[0]:
                            elif src_values[i_c] == delimiter:
                                quotes = True

                        d_index = np.uint64(0)
                        if comma or quotes:
                            dest_values[d_index] = delimiter
                            d_index += 1
                            for i_c in range(src_start, src_end):
                                # c = src_values[i_c]
                                # if c == delimiter[0]:
                                if src_values[i_c] == delimiter:
                                    # dest_values[d_index] = c
                                    dest_values[d_index] = src_values[i_c]
                                    d_index += 1
                                # dest_values[d_index] = c
                                dest_values[d_index] = src_values[i_c]
                                d_index += 1
                            dest_values[d_index] = delimiter
                            d_index += 1
                        else:
                            s_len = np.uint64(src_end - src_start)
                            dest_values[index_v:index_v + s_len] = src_values[src_start:src_end]
                            d_index += s_len
                        index_v += np.uint64(d_index)

        # if either the index or values are past the threshold, write them
        if index_i >= max_index_i or index_v >= max_value_i:
            break
            # return s+1, index_i, index_v
    return s+1, index_i, index_v


def apply_spans_concat(spans, reader, writer):
    separator = np.frombuffer(b',', '|S1')
    delimiter = np.frombuffer(b'"', '|S1')
    src_index = reader.field['index'][:]
    src_values = reader.field['values'][:]
    # write chunks to the writer
    # . read from a given span
    #   . calculate extra characters required
    dest_index = np.zeros(reader.chunksize, src_index.dtype)
    dest_values = np.zeros(reader.chunksize * 16, src_values.dtype)

    max_index_i = reader.chunksize
    max_value_i = reader.chunksize * 8
    s = 0
    while s < len(spans)-1:
        s, index_i, index_v = _apply_spans_concat(spans, src_index, src_values,
                                                  dest_index, dest_values,
                                                  max_index_i, max_value_i, s)
                                                  # separator[0], delimiter[0])

        if index_i > 0 or index_v > 0:
            writer.write_raw(dest_index[:index_i], dest_values[:index_v])
    writer.flush()
    # index_i = 1
    # index_v = 0
    # max_index_i = reader.chunksize
    # max_value_i = reader.chunksize * 8
    # dest_index[0] = spans[0]
    # for s in range(len(spans)-1):
    #     cur = spans[s]
    #     next = spans[s+1]
    #     cur_src_i = src_index[cur]
    #     next_src_i = src_index[next]
    #
    #     dest_index[index_i] = next_src_i
    #     index_i += 1
    #
    #     next_index_v = next_src_i - cur_src_i + np.uint64(index_v)
    #     dest_values[index_v:next_index_v] = src_values[cur_src_i:next_src_i]
    #     index_v = next_index_v
    #     # if either the index or values are past the threshold, write them
    #     if index_i >= max_index_i or index_v >= max_value_i:
    #         writer.write_raw(dest_index[:index_i], dest_values[:index_v])
    #         index_i = 0
    #         index_v = 0
    #
    # if index_i > 0 or index_v > 0:
    #     writer.write_raw(dest_index[:index_i], dest_values[:index_v])
    #
    # writer.flush()




def timestamp_to_date(values):
    results = np.zeros(len(values), dtype='|S10')
    template = "{:04d}-{:02d}-{:02d}"
    for i_r in range(len(values)):
        dt = datetime.fromtimestamp(values[i_r])
        results[i_r] = template.format(dt.year, dt.month, dt.day).encode()
        # results[i_r] = dt.strftime("YYYY-MM-DD").encode()
    return results


def get_reader_from_field(field):
    if 'fieldtype' not in field.attrs.keys():
        raise ValueError(f"'{field_name}' is not a well-formed field")

    fieldtype_map = {
        'indexedstring': NewIndexedStringReader,
        'fixedstring': NewFixedStringReader,
        'categorical': NewCategoricalReader,
        'boolean': NewNumericReader,
        'numeric': NewNumericReader,
        'datetime': NewTimestampReader,
        'date': NewTimestampReader,
        'timestamp': NewTimestampReader
    }

    fieldtype = field.attrs['fieldtype'].split(',')[0]
    return fieldtype_map[fieldtype](field)


def get_writer_from_field(field, dest_group, dest_name):
    reader = get_reader_from_field(field)
    return reader.get_writer(dest_group, dest_name)


@jit
def filtered_iterator(values, filter, default=np.nan):
    for i in range(len(values)):
        if filter[i]:
            yield default
        else:
            yield values[i]


def get_or_create_group(group, name):
    if name in group:
        return group[name]
    return group.create_group(name)

# Newest
# ======

def chunks(length, chunksize):
    cur = 0
    while cur < length:
        next = min(length, cur + chunksize)
        yield cur, next
        cur = next


def process(inputs, outputs, predicate):

    # TODO: modifying the dictionaries in place is not great
    input_readers = dict()
    for k, v in inputs.items():
        if isinstance(v, NewReader):
            input_readers[k] = v
        else:
            input_readers[k] = get_reader_from_field(v)
    output_writers = dict()
    output_arrays = dict()
    for k, v in outputs.items():
        if isinstance(v, NewWriter):
            output_writers[k] = v
        else:
            outputs[k] = get_writer_from_field(v)

    reader = next(iter(input_readers.values()))
    input_length = len(reader)
    writer = next(iter(output_writers.values()))
    chunksize = writer.chunksize
    required_chunksize = min(input_length, chunksize)
    for k, v in outputs.items():
        output_arrays[k] = output_writers[k].chunk_factory(required_chunksize)

    for c in chunks(input_length, chunksize):
        kwargs = dict()

        for k, v in inputs.items():
            kwargs[k] = v[c[0]:c[1]]
        for k, v in output_arrays.items():
            kwargs[k] = v
        predicate(**kwargs)

        # TODO: write back to the writer


def get_index(target, foreign_key, destination):
    print('  building patient_id index')
    t0 = time.time()
    target_lookup = dict()
    for i, v in enumerate(target[:]):
        target_lookup[v] = i
    print(f'  target lookup built in {time.time() - t0}s')

    print('perform initial index')
    t0 = time.time()
    foreign_key_elems = foreign_key[:]
    # foreign_key_index = np.asarray([target_lookup.get(i, -1) for i in foreign_key_elems],
    #                                    dtype=np.int64)
    foreign_key_index = np.zeros(len(foreign_key_elems), dtype=np.int64)

    current_invalid = -1
    for i_k, k in enumerate(foreign_key_elems):
        index = target_lookup.get(k, current_invalid)
        if index < 0:
            current_invalid -= 1
            target_lookup[k] = index
        foreign_key_index[i_k] = index
    print(f'initial index performed in {time.time() - t0}s')

    print(f'fixing up negative index order')
    t0 = time.time()
    # negative values are in the opposite of the key order, so reverse them
    foreign_key_index = np.where(foreign_key_index < 0,
                                 current_invalid - foreign_key_index,
                                 foreign_key_index)
    print(f'negative index order fix performed in {time.time() - t0}s')
    destination.write(foreign_key_index)



def get_trash_group(group):
    # parent = group.parent
    # while (parent != group):
    #     parent, group = parent.parent, group.parent
    # return group
    group_names = group.name[1:].split('/')

    while True:
        id = str(uuid.uuid4())
        try:
            result = group.create_group(f"/trash/{'/'.join(group_names[:-1])}/{id}")
            return result
        except KeyError:
            pass


class NewReader:
    def __init__(self, field):
        self.field = field


class NewIndexedStringReader(NewReader):
    def __init__(self, field):
        NewReader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype']
        if fieldtype != 'indexedstring':
            error = "'fieldtype of '{} should be 'indexedstring' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else len(self.field['index']) - 1
            step = item.step
            #TODO: validate slice
            index = self.field['index'][start:stop+1]
            bytestr = self.field['values'][index[0]:index[-1]]
            results = [None] * (len(index)-1)
            startindex = start
            for ir in range(len(results)):
                results[ir] =\
                    bytestr[index[ir]-np.uint64(startindex):
                            index[ir+1]-np.uint64(startindex)].tobytes().decode()
            return results

    def __len__(self):
        return len(self.field['index']) - 1

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return NewIndexedStringWriter(dest_group, self.chunksize, dest_name, timestamp, write_mode)

    def dtype(self):
        return self.field['index'].dtype, self.field['values'].dtype

    """
    0 1 3 6 10 15
    abbcccddddeeeee

    2 3 1 4 0
    0 3 7 9 14 15
    cccddddbbeeeeea
    """
    def sort(self, index, writer):
        field_index = self.field['index'][:]
        field_values = self.field['values'][:]
        r_field_index, r_field_values = apply_sort_to_index_values(index, field_index, field_values)
        writer.write_raw(r_field_index, r_field_values)


class NewNumericReader(NewReader):
    def __init__(self, field):
        NewReader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype'].split(',')
        if fieldtype[0] != 'numeric':
            error = "'fieldtype of '{} should be 'numeric' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return NewNumericWriter(dest_group, self.chunksize, dest_name, timestamp,
                                self.field.attrs['fieldtype'].split(',')[1], write_mode)

    def dtype(self):
        return self.field['values'].dtype


class NewCategoricalReader(NewReader):
    def __init__(self, field):
        NewReader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype']
        if fieldtype != 'categorical':
            error = "'fieldtype of '{} should be 'categorical' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']
        self.keys = self.field['keys'][()]

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return NewCategoricalWriter(dest_group, self.chunksize, dest_name, timestamp,
                                    {v: k for k, v in enumerate(self.field['keys'])}, write_mode)

    def dtype(self):
        return self.field['values'].dtype


class NewFixedStringReader(NewReader):
    def __init__(self, field):
        NewReader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype'].split(',')
        if fieldtype[0] != 'fixedstring':
            error = "'fieldtype of '{} should be 'numeric' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return NewFixedStringWriter(dest_group, self.chunksize, dest_name, timestamp,
                                    self.field.attrs['fieldtype'].split(',')[1], write_mode)

    def dtype(self):
        return self.field['values'].dtype


class NewTimestampReader(NewReader):
    def __init__(self, field):
        NewReader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype'].split(',')
        if fieldtype[0] not in ('datetime', 'date', 'timestamp'):
            error = "'fieldtype of '{} should be 'datetime' or 'date' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return NewTimestampWriter(dest_group, self.chunksize, dest_name, timestamp, write_mode)

    def dtype(self):
        return self.field['values'].dtype


write_modes = {'write', 'overwrite'}


class NewWriter:
    def __init__(self, group, name, write_mode):
        self.trash_field = None
        if write_mode not in write_modes:
            raise ValueError(f"'write_mode' must be one of {write_modes}")
        if name in group:
            if write_mode == 'overwrite':
                field = group[name]
                trash = get_trash_group(field)
                dest_name = trash.name + f"/{name.split('/')[-1]}"
                group.move(field.name, dest_name)
                self.trash_field = trash[name]
                field = group.create_group(name)
            else:
                error = (f"Field '{name}' already exists. Set 'write_mode' to 'overwrite' "
                         "if you want to overwrite the existing contents")
                raise KeyError(error)
        else:
            field = group.create_group(name)
        self.field = field

    def flush(self):
        if self.trash_field is not None:
            del self.trash_field


class NewIndexedStringWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp, write_mode='write'):
        NewWriter.__init__(self, group, name, write_mode)
        self.fieldtype = f'indexedstring'
        self.timestamp = timestamp
        self.chunksize = chunksize

        self.values = np.zeros(chunksize, dtype=np.uint8)
        self.indices = np.zeros(chunksize, dtype=np.uint64)
        self.ever_written = False
        self.accumulated = 0
        # self.indices[0] = self.accumulated
        # self.value_index = 0
        # self.index_index = 1
        self.value_index = 0
        self.index_index = 0

    def chunk_factory(self, length):
        return [None] * length

    def write_part(self, values):
        """Writes a list of strings in indexed string form to a field
        Args:
            values: a list of utf8 strings
        """
        if not self.ever_written:
            self.indices[0] = self.accumulated
            self.index_index = 1
            self.ever_written = True

        for s in values:
            evalue = s.encode()
            for v in evalue:
                self.values[self.value_index] = v
                self.value_index += 1
                if self.value_index == self.chunksize:
                    DataWriter._write(self.field, 'values', self.values, self.value_index)
                    self.value_index = 0
                self.accumulated += 1
            self.indices[self.index_index] = self.accumulated
            self.index_index += 1
            if self.index_index == self.chunksize:
                DataWriter._write(self.field, 'index', self.indices, self.index_index)
                self.index_index = 0

    def flush(self):
        if self.value_index != 0:
            DataWriter._write(self.field, 'values', self.values, self.value_index)
            self.value_index = 0
        if self.index_index != 0:
            DataWriter._write(self.field, 'index', self.indices, self.index_index)
            self.index_index = 0
        self.field.attrs['fieldtype'] = self.fieldtype
        self.field.attrs['timestamp'] = self.timestamp
        self.field.attrs['chunksize'] = self.chunksize
        self.field.attrs['completed'] = True
        NewWriter.flush(self)

    def write(self, values):
        self.write_part(values)
        self.flush()

    def write_part_raw(self, index, values):
        if index.dtype != np.uint64:
            raise ValueError(f"'index' must be an ndarray of '{np.uint64}'")
        if values.dtype != np.uint8:
            raise ValueError(f"'values' must be an ndarray of '{np.uint8}'")
        DataWriter._write(self.field, 'index', index, len(index))
        DataWriter._write(self.field, 'values', values, len(values))

    def write_raw(self, index, values):
        self.write_part_raw(index, values)
        self.flush()


class NewCategoricalImporter:
    def __init__(self, group, chunksize, name, timestamp, categories, write_mode='write'):
        self.writer =\
            NewCategoricalWriter(group, chunksize, name, timestamp, categories, write_mode)
        self.field_size = max([len(k) for k in categories.keys()])

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'U{self.field_size}')

    def write_part(self, values):
        results = np.zeros(len(values), dtype='uint8')
        keys = self.writer.keys
        for i in range(len(values)):
            results[i] = keys[values[i]]
        self.writer.write_part(results)

    def flush(self):
        self.writer.flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


class NewCategoricalWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp, categories, write_mode='write'):
        NewWriter.__init__(self, group, name, write_mode)
        self.fieldtype = f'categorical'
        self.timestamp = timestamp
        self.chunksize = chunksize
        self.keys = categories

    def chunk_factory(self, length):
        return np.zeros(length, dtype='uint8')

    def write_part(self, values):
        DataWriter._write(self.field, 'values', values, len(values))

    def flush(self):
        key_indices = [None] * len(self.keys)
        for k, v in self.keys.items():
            key_indices[v] = k
        DataWriter._write(self.field, 'keys', key_indices, len(self.keys),
                          dtype=h5py.string_dtype())
        self.field.attrs['fieldtype'] = self.fieldtype
        self.field.attrs['timestamp'] = self.timestamp
        self.field.attrs['chunksize'] = self.chunksize
        self.field.attrs['completed'] = True

    def write(self, values):
        self.write_part(values)
        self.flush()


class NewNumericImporter:
    def __init__(self, group, chunksize, name, timestamp, nformat, parser, write_mode='write'):
        self.data_writer =\
            NewNumericWriter(group, chunksize, name, timestamp, nformat, write_mode)
        self.flag_writer =\
            NewNumericWriter(group, chunksize, f"{name}_valid", timestamp, 'bool', write_mode)
        self.parser = parser

    def chunk_factory(self, length):
        return [None] * length

    def write_part(self, values):
        """
        Given a list of strings, parse the strings and write the parsed values. Values that
        cannot be parsed are written out as zero for the values, and zero for the flags to
        indicate that that entry is not valid.
        Args:
            values: a list of strings to be parsed
        """
        elements = np.zeros(len(values), dtype=self.data_writer.nformat)
        validity = np.zeros(len(values), dtype='bool')
        for i in range(len(values)):
            valid, value = self.parser(values[i])
            elements[i] = value
            validity[i] = valid
        self.data_writer.write_part(elements)
        self.flag_writer.write_part(validity)

    def flush(self):
        self.data_writer.flush()
        self.flag_writer.flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


class NewNumericWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp, nformat, write_mode='write'):
        NewWriter.__init__(self, group, name, write_mode)
        self.fieldtype = f'numeric,{nformat}'
        self.nformat = nformat
        self.timestamp = timestamp
        self.chunksize = chunksize

    def chunk_factory(self, length):
        nformat = self.fieldtype.split(',')[1]
        return np.zeros(length, dtype=nformat)

    def write_part(self, values):
        DataWriter._write(self.field, 'values', values, len(values))

    def flush(self):
        self.field.attrs['fieldtype'] = self.fieldtype
        self.field.attrs['timestamp'] = self.timestamp
        self.field.attrs['chunksize'] = self.chunksize
        self.field.attrs['nformat'] = self.nformat
        self.field.attrs['completed'] = True

    def write(self, values):
        self.write_part(values)
        self.flush()


class NewFixedStringWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp, strlen, write_mode='write'):
        NewWriter.__init__(self, group, name, write_mode)
        self.strlen = strlen
        self.fieldtype = f'fixedstring,{strlen}'
        self.timestamp = timestamp
        self.chunksize = chunksize

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'S{self.strlen}')

    def write_part(self, values):
        DataWriter._write(self.field, 'values', values, len(values))

    def flush(self):
        self.field.attrs['fieldtype'] = self.fieldtype
        self.field.attrs['timestamp'] = self.timestamp
        self.field.attrs['chunksize'] = self.chunksize
        self.field.attrs['completed'] = True

    def write(self, values):
        self.write_part(values)
        self.flush()


class OptionalDateTimeImporter:
    def __init__(self, group, chunksize, name, timestamp, optional=True, write_mode='write'):
        self.datetime = NewDateTimeWriter(group, chunksize, name, timestamp, write_mode)
        self.datestr = NewFixedStringWriter(group, chunksize, f"{name}_day", timestamp, '10',
                                            write_mode)
        self.datetimeset = None
        if optional:
            self.datetimeset =\
                NewNumericWriter(group, chunksize, f"{name}_set", timestamp, 'bool', write_mode)

    def chunk_factory(self, length):
        return self.datetime.chunk_factory(length)

    def write_part(self, values):
        # TODO: use a timestamp writer instead of a datetime writer and do the conversion here

        days = self.datestr.chunk_factory(len(values))
        flags = None
        if self.datetimeset is not None:
            flags = self.datetimeset.chunk_factory(len(values))
            for i in range(len(values)):
                flags[i] = values[i] != b''
                days[i] = values[i][:10]
        else:
            for i in range(len(values)):
                days[i] = values[i][:10]

        self.datetime.write_part(values)
        self.datestr.write_part(days)
        if self.datetimeset is not None:
            self.datetimeset.write_part(flags)

    def flush(self):
        self.datetime.flush()
        self.datestr.flush()
        if self.datetimeset is not None:
            self.datetimeset.flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


# TODO writers can write out more than one field; offset could be done this way
class NewDateTimeWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp, write_mode='write'):
        NewWriter.__init__(self, group, name, write_mode)
        self.fieldtype = f'datetime'
        self.timestamp = timestamp
        self.chunksize = chunksize
        self.name = name

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'S32')

    def write_part(self, values):
        timestamps = np.zeros(len(values), dtype=np.float64)
        for i in range(len(values)):
            value = values[i]
            if value == b'':
                timestamps[i] = 0
            else:
                if len(value) == 32:
                    ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S.%f%z')
                elif len(value) == 25:
                    ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S%z')
                else:
                    raise ValueError(f"Date field '{self.field}' has unexpected format '{value}'")
                timestamps[i] = ts.timestamp()
        DataWriter._write(self.field, 'values', timestamps, len(timestamps))

    def flush(self):
        self.field.attrs['fieldtype'] = self.fieldtype
        self.field.attrs['timestamp'] = self.timestamp
        self.field.attrs['chunksize'] = self.chunksize
        self.field.attrs['completed'] = True

    def write(self, values):
        self.write_part(values)
        self.flush()


class NewDateWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp, write_mode='writer'):
        NewWriter.__init__(self, group, name, write_mode)
        self.fieldtype = f'datetime'
        self.timestamp = timestamp
        self.chunksize = chunksize
        self.name = name

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'S32')

    def write_part(self, values):

        timestamps = np.zeros(len(values), dtype=np.float64)
        for i in range(len(values)):
            value = values[i]
            if value == b'':
                timestamps[i] = 0
            else:
                ts = datetime.strptime(value.decode(), '%Y-%m-%d')
                timestamps[i] = ts.timestamp()
        DataWriter._write(self.field, 'values', timestamps, len(timestamps))

    def flush(self):
        self.field.attrs['fieldtype'] = self.fieldtype
        self.field.attrs['timestamp'] = self.timestamp
        self.field.attrs['chunksize'] = self.chunksize
        self.field.attrs['completed'] = True

    def write(self, values):
        self.write_part(values)
        self.flush()


class NewTimestampWriter:
    def __init__(self, group, chunksize, name, timestamp, write_mode='write'):
        NewWriter.__init__(self, group, name, write_mode)
        self.fieldtype = f'timestamp'
        self.timestamp = timestamp
        self.chunksize = chunksize
        self.name = name

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'float64')

    def write_part(self, values):
        DataWriter._write(self.field, 'values', values, len(values))

    def flush(self):
        self.field.attrs['fieldtype'] = self.fieldtype
        self.field.attrs['timestamp'] = self.timestamp
        self.field.attrs['chunksize'] = self.chunksize
        self.field.attrs['completed'] = True

    def write(self, values):
        self.write_part(values)
        self.flush()


class OptionalDateImporter:
    def __init__(self, group, chunksize, name, timestamp, optional=True, write_mode='write'):
        self.date = NewDateWriter(group, chunksize, name, timestamp, write_mode)
        self.datestr = NewFixedStringWriter(group, chunksize, f"{name}_day", timestamp, '10',
                                            write_mode)
        self.dateset = None
        if optional:
            self.dateset =\
                NewNumericWriter(group, chunksize, f"{name}_set", timestamp, 'bool', write_mode)

    def chunk_factory(self, length):
        return self.date.chunk_factory(length)

    def write_part(self, values):
        # TODO: use a timestamp writer instead of a datetime writer and do the conversion here
        days = self.datestr.chunk_factory(len(values))
        flags = None
        if self.dateset is not None:
            flags = self.dateset.chunk_factory(len(values))
            for i in range(len(values)):
                flags[i] = values[i] != b''
                days[i] = values[i][:10]
        else:
            for i in range(len(values)):
                days[i] = values[i][:10]

        self.date.write_part(values)
        self.datestr.write_part(days)
        if self.dateset is not None:
            self.dateset.write_part(flags)

    def flush(self):
        self.date.flush()
        self.datestr.flush()
        if self.dateset is not None:
            self.dateset.flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


def sort_on(src_group, dest_group, keys, fields=None, timestamp=datetime.now(timezone.utc),
            write_mode='write'):
    sort_keys = ('patient_id', 'created_at')
    readers = tuple(get_reader_from_field(src_group[f]) for f in keys)
    # patient_id_reader = NewFixedStringReader(assessments_src['patient_id'])
    # raw_patient_ids = patient_id_reader[:]
    # created_at_reader = NewTimestampReader(assessments_src['created_at'])
    # raw_created_ats = created_at_reader[:]
    t1 = time.time()
    sorted_index = dataset_sort(
        np.arange(len(readers[0]), dtype=np.uint32), readers)
    print(f'sorted {sort_keys} index in {time.time() - t1}s')

    t0 = time.time()
    for k in src_group.keys():
        t1 = time.time()
        r = get_reader_from_field(src_group[k])
        w = r.getwriter(dest_group, k, timestamp, write_mode=write_mode)
        apply_sort(sorted_index, r, w)
        del r
        del w
        print(f"  '{k}' reordered in {time.time() - t1}s")
    print(f"fields reordered in {time.time() - t0}s")
