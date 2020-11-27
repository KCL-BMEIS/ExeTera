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
from contextlib import contextmanager
from datetime import datetime, timezone
import time

import h5py
import numpy as np
import numba
from numba import jit, njit
import pandas as pd

from exetera.core import validation as val
from exetera.core import readerwriter as rw
from exetera.core.operations import INVALID_INDEX, DEFAULT_CHUNKSIZE

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

# TODO:
"""
 * mapping and joining
   * by function
     * get_mapping (fwd / bwd)
     * first_in_second -> filter
     * second_in_first -> filter
     * join(left, right, inner, outer)
     * aggregate_and_join(left, right, inner, outer, left_fn, right_fn
   * use of pandas
   * use of dicts
"""

chunk_sizes = {
    'patient': (DEFAULT_CHUNKSIZE,), 'assessment': (DEFAULT_CHUNKSIZE,), 'test': (DEFAULT_CHUNKSIZE,)
}



# @numba.njit
# def _safe_map(data_field, map_field, map_filter=None):
#     result = np.zeros_like(map_field, dtype=data_field.dtype)
#     empty_val = result[0]
#     if map_filter is not None:
#         for i in range(len(map_field)):
#             if map_filter[i]:
#                 result[i] = data_field[map_field[i]]
#             else:
#                 result[i] = empty_val
#     else:
#         for i in range(len(map_field)):
#             result[i] = data_field[map_field[i]]
#
#     return result


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


def _apply_filter_to_array(values, filter):
    return values[filter]


@njit
def _apply_filter_to_index_values(index_filter, indices, values):
    # pass 1 - determine the destination lengths
    cur_ = indices[:-1]
    next_ = indices[1:]
    count = 0
    total = 0
    for i in range(len(index_filter)):
        if index_filter[i] == True:
            count += 1
            total += next_[i] - cur_[i]
    dest_indices = np.zeros(count+1, indices.dtype)
    dest_values = np.zeros(total, values.dtype)
    dest_indices[0] = 0
    count = 1
    total = 0
    for i in range(len(index_filter)):
        if index_filter[i] == True:
            n = next_[i]
            c = cur_[i]
            delta = n - c
            dest_values[total:total + delta] = values[c:n]
            total += delta
            dest_indices[count] = total
            count += 1
    return dest_indices, dest_values


@njit
def _apply_indices_to_index_values(indices_to_apply, indices, values):
    # pass 1 - determine the destination lengths
    cur_ = indices[:-1]
    next_ = indices[1:]
    count = 0
    total = 0
    for i in indices_to_apply:
        count += 1
        total += next_[i] - cur_[i]
    dest_indices = np.zeros(count+1, indices.dtype)
    dest_values = np.zeros(total, values.dtype)
    dest_indices[0] = 0
    count = 1
    total = 0
    for i in indices_to_apply:
        n = next_[i]
        c = cur_[i]
        delta = n - c
        dest_values[total:total + delta] = values[c:n]
        total += delta
        dest_indices[count] = total
        count += 1
    return dest_indices, dest_values


def _apply_sort_to_array(index, values):
    return values[index]


@njit
def _apply_sort_to_index_values(index, indices, values):

    s_indices = np.zeros_like(indices)
    s_values = np.zeros_like(values)
    accumulated = np.int64(0)
    s_indices[0] = 0
    for di, si in enumerate(index):
        src_field_start = indices[si]
        src_field_end = indices[si + 1]
        length = np.int64(src_field_end - src_field_start)
        if length > 0:
            s_values[accumulated:accumulated + length] =\
                values[src_field_start:src_field_end]
        accumulated += length
        if s_indices[di + 1] != 0:
            print('non-zero index!')
        s_indices[di + 1] = accumulated

    return s_indices, s_values


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


def _get_spans(field, fields):
    if field is not None:
        return _get_spans_for_field(field)
    elif len(fields) == 1:
        return _get_spans_for_field(fields[0])
    elif len(fields) == 2:
        return _get_spans_for_2_fields(*fields)
    else:
        raise NotImplementedError("This operation does not support more than two fields at present")


@njit
def _index_spans(spans, results):
    sp_sta = spans[:-1]
    sp_end = spans[1:]
    for s in range(len(sp_sta)):
        results[sp_sta[s]:sp_end[s]] = s
    return results


def _get_spans_for_field(field0):
    results = np.zeros(len(field0) + 1, dtype=np.bool)
    if np.issubdtype(field0.dtype, np.number):
        fn = np.not_equal
    else:
        fn = np.char.not_equal
    results[1:-1] = fn(field0[:-1], field0[1:])

    results[0] = True
    results[-1] = True
    return np.nonzero(results)[0]


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


@njit
def _apply_spans_index_of_max(spans, src_array, dest_array):
    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]

        if next - cur == 1:
            dest_array[i] = cur
        else:
            dest_array[i] = cur + src_array[cur:next].argmax()


@njit
def _apply_spans_index_of_min(spans, src_array, dest_array, filter_array):
    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]
        if next - cur == 0:
            filter_array[i] = False
        elif next - cur == 1:
            dest_array[i] = cur
        else:
            dest_array[i] = cur = src_array[cur:next].argmin()


@njit
def _apply_spans_index_of_first(spans, dest_array):
    dest_array[:] = spans[:-1]


@njit
def _apply_spans_index_of_last(spans, dest_array):
    dest_array[:] = spans[1:] - 1


@njit
def _apply_spans_count(spans, dest_array):
    for i in range(len(spans)-1):
        dest_array[i] = np.int64(spans[i+1] - spans[i])


@njit
def _apply_spans_first(spans, src_array, dest_array):
    dest_array[:] = src_array[spans[:-1]]


@njit
def _apply_spans_last(spans, src_array, dest_array):
    spans = spans[1:]-1
    dest_array[:] = src_array[spans]


@njit
def _apply_spans_max(spans, src_array, dest_array):

    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]
        if next - cur == 1:
            dest_array[i] = src_array[cur]
        else:
            dest_array[i] = src_array[cur:next].max()


@njit
def _apply_spans_min(spans, src_array, dest_array):

    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]
        if next - cur == 1:
            dest_array[i] = src_array[cur]
        else:
            dest_array[i] = src_array[cur:next].min()


# def _apply_spans_concat(spans, src_field):
#     dest_values = [None] * (len(spans)-1)
#     for i in range(len(spans)-1):
#         cur = spans[i]
#         next = spans[i+1]
#         if next - cur == 1:
#             dest_values[i] = src_field[cur]
#         else:
#             src = [s for s in src_field[cur:next] if len(s) > 0]
#             if len(src) > 0:
#                 dest_values[i] = ','.join(utils.to_escaped(src))
#             else:
#                 dest_values[i] = ''
#             # if len(dest_values[i]) > 0:
#             #     print(dest_values[i])
#     return dest_values


@njit
def _apply_spans_concat(spans, src_index, src_values, dest_index, dest_values,
                        max_index_i, max_value_i, s_start, separator, delimiter):
    if s_start == 0:
        index_i = np.uint32(1)
        index_v = np.int64(0)
        dest_index[0] = spans[0]
    else:
        index_i = np.uint32(0)
        index_v = np.int64(0)

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
                next_index_v = next_src_i - cur_src_i + np.int64(index_v)
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
                    next_index_v = next_src_i - cur_src_i + np.int64(index_v)
                    dest_values[index_v:next_index_v] = src_values[cur_src_i:next_src_i]
                    index_v = next_index_v
                else:
                    # the outer conditional already determines that we have a non-empty entry
                    # so there must be multiple non-empty entries and commas are required
                    for e in range(cur, next):
                        src_start = src_index[e]
                        src_end = src_index[e+1]
                        comma = False
                        quotes = False
                        for i_c in range(src_start, src_end):
                            if src_values[i_c] == separator:
                                comma = True
                            elif src_values[i_c] == delimiter:
                                quotes = True

                        d_index = np.int64(0)
                        if comma or quotes:
                            dest_values[d_index] = delimiter
                            d_index += 1
                            for i_c in range(src_start, src_end):
                                if src_values[i_c] == delimiter:
                                    dest_values[d_index] = src_values[i_c]
                                    d_index += 1
                                dest_values[d_index] = src_values[i_c]
                                d_index += 1
                            dest_values[d_index] = delimiter
                            d_index += 1
                        else:
                            s_len = np.int64(src_end - src_start)
                            dest_values[index_v:index_v + s_len] = src_values[src_start:src_end]
                            d_index += s_len
                        index_v += np.int64(d_index)

        # if either the index or values are past the threshold, write them
        if index_i >= max_index_i or index_v >= max_value_i:
            break
    return s+1, index_i, index_v


# TODO - this can go if it isn't needed
def timestamp_to_date(values):
    results = np.zeros(len(values), dtype='|S10')
    template = "{:04d}-{:02d}-{:02d}"
    for i_r in range(len(values)):
        dt = datetime.fromtimestamp(values[i_r])
        results[i_r] = template.format(dt.year, dt.month, dt.day).encode()
    return results

# TODO: refactor into datastore
@jit
def filtered_iterator(values, filter, default=np.nan):
    for i in range(len(values)):
        if filter[i]:
            yield default
        else:
            yield values[i]

@njit
def _map_valid_indices(src, map, default):

    filter = map < INVALID_INDEX
    #dest = np.where(filter, src[map], default)
    dest = np.zeros(len(map), dtype=src.dtype)
    for i_r in range(len(map)):
        if filter[i_r]:
            dest[i_r] = src[map[i_r]]
        else:
            dest[i_r] = default
    return dest


def _values_from_reader_or_ndarray(name, field):
    if isinstance(field, rw.Reader):
        raw_field = field[:]
    elif isinstance(field, np.ndarray):
        raw_field = field
    else:
        raise ValueError(f"'{name}' must be a Reader or an ndarray but is {type(field)}")
    return raw_field


# TODO: handle usage of reader
def filter_duplicate_fields(field):

    filter_ = np.ones(len(field), dtype=np.bool)
    _filter_duplicate_fields(field, filter_)
    return filter_

def _filter_duplicate_fields(field, filter):
    seen_ids = dict()
    for i in range(len(field)):
        f = field[i]
        if f in seen_ids:
            filter[i] = False
        else:
            seen_ids[f] = 1
            filter[i] = True
    return filter


def foreign_key_is_in_primary_key(primary_key, foreign_key):
    val._check_is_reader_or_ndarray('primary_key', primary_key)
    val._check_is_reader_or_ndarray('foreign_key', foreign_key)
    if isinstance(primary_key, rw.Reader):
        pk = primary_key[:]
    else:
        pk = primary_key
    if isinstance(foreign_key, rw.Reader):
        fk = foreign_key[:]
    else:
        fk = foreign_key

    result = np.zeros(len(fk), dtype=np.bool)
    return _filter_non_orphaned_foreign_keys(pk, fk, result)


def _filter_non_orphaned_foreign_keys(primary_key, foreign_key, results):
    pkids = dict()
    trueval = np.bool(True)
    falseval = np.bool(False)
    for p in primary_key:
        pkids[p] = trueval

    for i, f in enumerate(foreign_key):
        results[i] = pkids.get(f, falseval)
    return results


# Newest
# ======


def _aggregate_impl(predicate, fkey_indices=None, fkey_index_spans=None,
                    reader=None, writer=None, result_dtype=None):
    if fkey_indices is None and fkey_index_spans is None:
        raise ValueError("One of 'fkey_indices' or 'fkey_index_spans' must be set")
    if fkey_indices is not None and fkey_index_spans is not None:
        raise ValueError("Only one of 'fkey_indices' and 'fkey_index_spans' may be set")
    if writer is not None:
        if not isinstance(writer, (rw.Writer, np.ndarray)):
            raise ValueError("'writer' must be either a Writer or an ndarray instance")

    if fkey_index_spans is None:
        fkey_index_spans = _get_spans(field=fkey_indices)

    if isinstance(writer, np.ndarray):
        if len(writer) != len(fkey_index_spans) - 1:
            error = "'writer': ndarray must be of length {} but is of length {}"
            raise ValueError(error.format(len(fkey_index_spans) - 1), len(writer))
        elif writer.dtype != result_dtype:
            raise ValueError(f"'writer' dtype must be {result_dtype} but is {writer.dtype}")

    if isinstance(writer, rw.Writer) or writer is None:
        results = np.zeros(len(fkey_index_spans) - 1, dtype=result_dtype)
    else:
        results = writer

    # execute the predicate (note that not every predicate requires a reader)
    predicate(fkey_index_spans, reader, results)

    if isinstance(writer, rw.Writer):
        writer.write(results)

    return writer if writer is not None else results


class DataStore:

    def __init__(self, chunksize=DEFAULT_CHUNKSIZE,
                 timestamp=str(datetime.now(timezone.utc))):
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.chunksize = chunksize
        self.timestamp = timestamp


    def set_timestamp(self, timestamp=str(datetime.now(timezone.utc))):
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.timestamp = timestamp


    # TODO: fields is being ignored at present
    def sort_on(self, src_group, dest_group, keys, fields=None,
                timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = self.timestamp
        # sort_keys = ('patient_id', 'created_at')
        readers = tuple(self.get_reader(src_group[f]) for f in keys)
        t1 = time.time()
        sorted_index = self.dataset_sort(readers, np.arange(len(readers[0]), dtype=np.uint32))
        print(f'sorted {keys} index in {time.time() - t1}s')

        t0 = time.time()
        for k in src_group.keys():
            t1 = time.time()
            r = self.get_reader(src_group[k])
            w = r.get_writer(dest_group, k, timestamp, write_mode=write_mode)
            self.apply_sort(sorted_index, r, w)
            del r
            del w
            print(f"  '{k}' reordered in {time.time() - t1}s")
        print(f"fields reordered in {time.time() - t0}s")


    def dataset_sort(self, readers, index=None):
        r_readers = reversed(readers)
        if index is None:
            index = np.arange(len(r_readers[0]))

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


    # TODO: index should be able to be either a reader or an ndarray
    def apply_sort(self, index, reader, writer=None):
        val._check_is_appropriate_writer_if_set(self, 'writer', reader, writer)

        if isinstance(reader, rw.IndexedStringReader):
            src_indices = reader.field['index'][:]
            src_values = reader.field.get('values', np.zeros(0, dtype=np.uint8))[:]
            indices, values = _apply_sort_to_index_values(index, src_indices, src_values)
            if writer:
                writer.write_raw(indices, values)
            return indices, values
        elif isinstance(reader, rw.Reader):
            result = _apply_sort_to_array(index, reader[:])
            if writer:
                writer.write(result)
            return result
        elif isinstance(reader, np.ndarray):
            result = _apply_sort_to_array(index, reader)
            if writer:
                writer.write(result)
            return result
        else:
            raise ValueError(f"'reader' must be a Reader or an ndarray, but is {type.reader}")


    # TODO: write filter with new readers / writers rather than deleting this
    def apply_filter(self, filter_to_apply, reader, writer=None):
        if isinstance(reader, rw.IndexedStringReader):
            val._check_is_appropriate_writer_if_set(self, 'writer', reader, writer)

            src_indices = reader.field['index'][:]
            src_values = reader.field.get('values', np.zeros(0, dtype=np.uint8))[:]
            if len(src_indices) != len(filter_to_apply) + 1:
                raise ValueError(f"'indices' (length {len(indices)}) must be one longer than "
                                 f"'index_filter' (length {len(index_filter)})")

            indices, values = _apply_filter_to_index_values(filter_to_apply,
                                                            src_indices, src_values)
            if writer:
                writer.write_raw(indices, values)
            return indices, values
        elif isinstance(reader, rw.Reader):
            if len(filter_to_apply) != len(reader):
                msg = ("'filter_to_apply' and 'reader' must be the same length "
                       "but are length {} and {} respectively")
                raise ValueError(msg.format(len(filter_to_apply), len(reader)))

            result = reader[:][filter_to_apply]
            if writer:
                writer.write(result)
            return result
        elif isinstance(reader, np.ndarray):
            result = reader[filter_to_apply]
            if writer:
                writer.write(result)
            return result
        else:
            raise ValueError("'reader' must be a Reader or an ndarray, but is {}".format(type(reader)))


    def apply_indices(self, indices_to_apply, reader, writer=None):
        if isinstance(reader, rw.IndexedStringReader):
            val._check_is_appropriate_writer_if_set(self, 'writer', reader, writer)

            src_indices = reader.field['index'][:]
            src_values = reader.field.get('values', np.zeros(0, dtype=np.uint8))[:]

            indices, values = _apply_indices_to_index_values(indices_to_apply,
                                                             src_indices, src_values)
            if writer:
                writer.write_raw(indices, values)
            return indices, values
        elif isinstance(reader, rw.Reader):
            result = reader[:][indices_to_apply]
            if writer:
                writer.write(result)
            return result
        elif isinstance(reader, np.ndarray):
            result = reader[indices_to_apply]
            if writer:
                writer.write(result)
            return result
        else:
            raise ValueError(f"'reader' must be a Reader or an ndarray, but is {type(reader)}")


    # TODO: write distinct with new readers / writers rather than deleting this
    def distinct(self, field=None, fields=None, filter=None):
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


    def get_spans(self, field=None, fields=None):
        if field is None and fields is None:
            raise ValueError("One of 'field' and 'fields' must be set")
        if field is not None and fields is not None:
            raise ValueError("Only one of 'field' and 'fields' may be set")
        raw_field = None
        raw_fields = None
        if field is not None:
            val._check_is_reader_or_ndarray('field', field)
            raw_field = field[:] if isinstance(field, rw.Reader) else field
        else:
            raw_fields = []
            for f in fields:
                val._check_is_reader_or_ndarray('elements of tuple/list fields', f)
                raw_fields.append(f[:] if isinstance(f, rw.Reader) else f)
        return _get_spans(raw_field, raw_fields)


    def index_spans(self, spans):
        raw_spans = val.raw_array_from_parameter(self, "spans", spans)
        results = np.zeros(raw_spans[-1], dtype=np.int64)
        return _index_spans(raw_spans, results)

    # TODO: needs a predicate to break ties: first, last?
    def apply_spans_index_of_min(self, spans, reader, writer=None):
        val._check_is_reader_or_ndarray('reader', reader)
        val._check_is_reader_or_ndarray_if_set('writer', reader)

        if isinstance(reader, rw.Reader):
            raw_reader = reader[:]
        else:
            raw_reader = reader

        if writer is not None:
            if isinstance(writer, rw.Writer):
                raw_writer = writer[:]
            else:
                raw_writer = writer
        else:
            raw_writer = np.zeros(len(spans)-1, dtype=np.int64)
        _apply_spans_index_of_min(spans, raw_reader, raw_writer)
        if isinstance(writer, rw.Writer):
            writer.write(raw_writer)
        else:
            return raw_writer


    def apply_spans_index_of_max(self, spans, reader, writer=None):
        val._check_is_reader_or_ndarray('reader', reader)
        val._check_is_reader_or_ndarray_if_set('writer', reader)

        if isinstance(reader, rw.Reader):
            raw_reader = reader[:]
        else:
            raw_reader = reader

        if writer is not None:
            if isinstance(writer, rw.Writer):
                raw_writer = writer[:]
            else:
                raw_writer = writer
        else:
            raw_writer = np.zeros(len(spans)-1, dtype=np.int64)
        _apply_spans_index_of_max(spans, raw_reader, raw_writer)
        if isinstance(writer, rw.Writer):
            writer.write(raw_writer)
        else:
            return raw_writer

    def apply_spans_index_of_first(self, spans, writer=None):

        if writer is not None:
            if isinstance(writer, rw.Writer):
                raw_writer = writer[:]
            else:
                raw_writer = writer
        else:
            raw_writer = np.zeros(len(spans) - 1, dtype=np.int64)
        _apply_spans_index_of_first(spans, raw_writer)
        if isinstance(writer, rw.Writer):
            writer.write(raw_writer)
        else:
            return raw_writer

    def apply_spans_index_of_last(self, spans, writer=None):

        if writer is not None:
            if isinstance(writer, rw.Writer):
                raw_writer = writer[:]
            else:
                raw_writer = writer
        else:
            raw_writer = np.zeros(len(spans) - 1, dtype=np.int64)
        _apply_spans_index_of_last(spans, raw_writer)
        if isinstance(writer, rw.Writer):
            writer.write(raw_writer)
        else:
            return raw_writer


    # TODO - for all apply_spans methods, spans should be able to be an ndarray
    def apply_spans_count(self, spans, _, writer=None):
        if writer is None:
            _values_from_reader_or_ndarray('spans', spans)
            dest_values = np.zeros(len(spans)-1, dtype=np.int64)
            _apply_spans_count(spans, dest_values)
            return dest_values
        else:
            if isinstance(writer, rw.Writer):
                dest_values = writer.chunk_factory(len(spans) - 1)
                _apply_spans_count(spans, dest_values)
                writer.write(dest_values)
            elif isinstance(writer, np.ndarray):
                _apply_spans_count(spans, writer)
            else:
                raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")

    def apply_spans_first(self, spans, reader, writer):
        if isinstance(reader, rw.Reader):
            read_values = reader[:]
        elif isinstance(reader, np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, rw.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_first(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            _apply_spans_first(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_last(self, spans, reader, writer):
        if isinstance(reader, rw.Reader):
            read_values = reader[:]
        elif isinstance(reader, np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, rw.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_last(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            _apply_spans_last(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_min(self, spans, reader, writer):
        if isinstance(reader, rw.Reader):
            read_values = reader[:]
        elif isinstance(reader, np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, rw.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_min(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(reader, rw.Reader):
            _apply_spans_min(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_max(self, spans, reader, writer):
        if isinstance(reader, rw.Reader):
            read_values = reader[:]
        elif isinstance(reader, np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, rw.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_max(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(reader, rw.Reader):
            _apply_spans_max(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_concat(self, spans, reader, writer):
        if not isinstance(reader, rw.IndexedStringReader):
            raise ValueError(f"'reader' must be one of 'IndexedStringReader' but is {type(reader)}")
        if not isinstance(writer, rw.IndexedStringWriter):
            raise ValueError(f"'writer' must be one of 'IndexedStringWriter' but is {type(writer)}")

        src_index = reader.field['index'][:]
        src_values = reader.field['values'][:]
        dest_index = np.zeros(reader.chunksize, src_index.dtype)
        dest_values = np.zeros(reader.chunksize * 16, src_values.dtype)

        max_index_i = reader.chunksize
        max_value_i = reader.chunksize * 8

        if src_values.dtype == 'S1':
            separator = b','
            delimiter = b'"'
        elif src_values.dtype == np.uint8:
            separator = np.frombuffer(b',', dtype='S1')[0]
            delimiter = np.frombuffer(b'"', dtype='S1')[0]

        s = 0
        while s < len(spans) - 1:
            s, index_i, index_v = _apply_spans_concat(spans, src_index, src_values,
                                                      dest_index, dest_values,
                                                      max_index_i, max_value_i, s,
                                                      separator, delimiter)

            if index_i > 0 or index_v > 0:
                writer.write_raw(dest_index[:index_i], dest_values[:index_v])
        writer.flush()


    def aggregate_count(self, fkey_indices=None, fkey_index_spans=None,
                        reader=None, writer=None):
        return _aggregate_impl(self.apply_spans_count, fkey_indices, fkey_index_spans,
                               reader, writer, np.uint32)


    def aggregate_first(self, fkey_indices=None, fkey_index_spans=None,
                        reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_first, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_last(self, fkey_indices=None, fkey_index_spans=None,
                       reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_last, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_min(self,fkey_indices=None, fkey_index_spans=None,
                      reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_min, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_max(self, fkey_indices=None, fkey_index_spans=None,
                      reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_max, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_custom(self,
                         predicate, fkey_indices=None, fkey_index_spans=None,
                         reader=None, writer=None):
        if reader is None:
            raise ValueError("'reader' must not be None")
        if not isinstance(reader, (rw.Reader, np.ndarray)):
            raise ValueError(f"'reader' must be a Reader or an ndarray but is {type(reader)}")
        if isinstance(reader, rw.Reader):
            required_dtype = reader.dtype()
        else:
            required_dtype = reader.dtype
        return _aggregate_impl(predicate, fkey_indices, fkey_index_spans,
                               reader, writer, required_dtype)


    def join(self,
             destination_pkey, fkey_indices, values_to_join,
             writer=None, fkey_index_spans=None):
        if fkey_indices is not None:
            if not isinstance(fkey_indices, (rw.Reader, np.ndarray)):
                raise ValueError(f"'fkey_indices' must be a type of Reader or an ndarray")
        if values_to_join is not None:
            if not isinstance(values_to_join, (rw.Reader, np.ndarray)):
                raise ValueError(f"'values_to_join' must be a type of Reader but is {type(values_to_join)}")
            if isinstance(values_to_join, rw.IndexedStringReader):
                raise ValueError(f"Joins on indexed string fields are not supported")

        if isinstance(values_to_join, rw.Reader):
            raw_values_to_join = values_to_join[:]
        else:
            raw_values_to_join = values_to_join

        # generate spans for the sorted key indices if not provided
        if fkey_index_spans is None:
            fkey_index_spans = self.get_spans(field=fkey_indices)

        # select the foreign keys from the start of each span to get an ordered list
        # of unique id indices in the destination space that the results of the predicate
        # execution are mapped to
        unique_fkey_indices = fkey_indices[:][fkey_index_spans[:-1]]

        # generate a filter to remove invalid foreign key indices (where values in the
        # foreign key don't map to any values in the destination space
        invalid_filter = unique_fkey_indices < INVALID_INDEX
        safe_unique_fkey_indices = unique_fkey_indices[invalid_filter]

        # the predicate results are in the same space as the unique_fkey_indices, which
        # means they may still contain invalid indices, so filter those now
        safe_values_to_join = raw_values_to_join[invalid_filter]

        # now get the memory that the results will be mapped to
        destination_space_values = writer.chunk_factory(len(destination_pkey))

        # finally, map the results from the source space to the destination space
        destination_space_values[safe_unique_fkey_indices] = safe_values_to_join

        if writer is not None:
            writer.write(destination_space_values)
        else:
            return destination_space_values


    def predicate_and_join(self,
                           predicate, destination_pkey, fkey_indices,
                           reader=None, writer=None, fkey_index_spans=None):
        if reader is not None:
            if not isinstance(reader, rw.Reader):
                raise ValueError(f"'reader' must be a type of Reader but is {type(reader)}")
            if isinstance(reader, rw.IndexedStringReader):
                raise ValueError(f"Joins on indexed string fields are not supported")

        # generate spans for the sorted key indices if not provided
        if fkey_index_spans is None:
            fkey_index_spans = self.get_spans(field=fkey_indices)

        # select the foreign keys from the start of each span to get an ordered list
        # of unique id indices in the destination space that the results of the predicate
        # execution are mapped to
        unique_fkey_indices = fkey_indices[:][fkey_index_spans[:-1]]

        # generate a filter to remove invalid foreign key indices (where values in the
        # foreign key don't map to any values in the destination space
        invalid_filter = unique_fkey_indices < INVALID_INDEX
        safe_unique_fkey_indices = unique_fkey_indices[invalid_filter]

        # execute the predicate (note that not every predicate requires a reader)
        if reader is not None:
            dtype = reader.dtype()
        else:
            dtype = np.uint32
        results = np.zeros(len(fkey_index_spans)-1, dtype=dtype)
        predicate(fkey_index_spans, reader, results)

        # the predicate results are in the same space as the unique_fkey_indices, which
        # means they may still contain invalid indices, so filter those now
        safe_results = results[invalid_filter]

        # now get the memory that the results will be mapped to
        destination_space_values = writer.chunk_factory(len(destination_pkey))
        # finally, map the results from the source space to the destination space
        destination_space_values[safe_unique_fkey_indices] = safe_results

        writer.write(destination_space_values)


    def get_reader(self, field):
        if 'fieldtype' not in field.attrs.keys():
            raise ValueError(f"'{field_name}' is not a well-formed field")

        fieldtype_map = {
            'indexedstring': rw.IndexedStringReader,
            'fixedstring': rw.FixedStringReader,
            'categorical': rw.CategoricalReader,
            'boolean': rw.NumericReader,
            'numeric': rw.NumericReader,
            'datetime': rw.TimestampReader,
            'date': rw.TimestampReader,
            'timestamp': rw.TimestampReader
        }

        fieldtype = field.attrs['fieldtype'].split(',')[0]
        return fieldtype_map[fieldtype](self, field)


    def get_existing_writer(self, field, timestamp=None):
        if 'fieldtype' not in field.attrs.keys():
            raise ValueError(f"'{field_name}' is not a well-formed field")

        fieldtype_map = {
            'indexedstring': rw.IndexedStringReader,
            'fixedstring': rw.FixedStringReader,
            'categorical': rw.CategoricalReader,
            'boolean': rw.NumericReader,
            'numeric': rw.NumericReader,
            'datetime': rw.TimestampReader,
            'date': rw.TimestampReader,
            'timestamp': rw.TimestampReader
        }

        fieldtype = field.attrs['fieldtype'].split(',')[0]
        reader = fieldtype_map[fieldtype](self, field)
        group = field.parent
        name = field.name.split('/')[-1]
        return reader.get_writer(group, name, timestamp=timestamp, write_mode='overwrite')


    def get_indexed_string_writer(self, group, name,
                                  timestamp=None, writemode='write'):
        return rw.IndexedStringWriter(self, group, name, timestamp, writemode)


    def get_fixed_string_writer(self, group, name, width,
                                timestamp=None, writemode='write'):
        return rw.FixedStringWriter(self, group, name, width, timestamp, writemode)


    def get_categorical_writer(self, group, name, categories,
                               timestamp=None, writemode='write'):
        return rw.CategoricalWriter(self, group, name, categories, timestamp, writemode)


    def get_numeric_writer(self, group, name, dtype, timestamp=None, writemode='write'):
        return rw.NumericWriter(self, group, name, dtype, timestamp, writemode)


    def get_timestamp_writer(self, group, name, timestamp=None, writemode='write'):
        return rw.TimestampWriter(self, group, name, timestamp, writemode)


    def get_compatible_writer(self, field, dest_group, dest_name,
                              timestamp=None, writemode='write'):
        reader = self.get_reader(field)
        return reader.get_writer(dest_group, dest_name, timestamp, writemode)


    def get_or_create_group(self, group, name):
        if name in group:
            return group[name]
        return group.create_group(name)


    def chunks(self, length, chunksize=None):
        if chunksize is None:
            chunksize = self.chunksize
        cur = 0
        while cur < length:
            next = min(length, cur + chunksize)
            yield cur, next
            cur = next

    def process(self, inputs, outputs, predicate):

        # TODO: modifying the dictionaries in place is not great
        input_readers = dict()
        for k, v in inputs.items():
            if isinstance(v, rw.Reader):
                input_readers[k] = v
            else:
                input_readers[k] = self.get_reader(v)
        output_writers = dict()
        output_arrays = dict()
        for k, v in outputs.items():
            if isinstance(v, rw.Writer):
                output_writers[k] = v
            else:
                raise ValueError("'outputs': all values must be 'Writers'")

        reader = next(iter(input_readers.values()))
        input_length = len(reader)
        writer = next(iter(output_writers.values()))
        chunksize = writer.chunksize
        required_chunksize = min(input_length, chunksize)
        for k, v in outputs.items():
            output_arrays[k] = output_writers[k].chunk_factory(required_chunksize)

        for c in self.chunks(input_length, chunksize):
            kwargs = dict()

            for k, v in inputs.items():
                kwargs[k] = v[c[0]:c[1]]
            for k, v in output_arrays.items():
                kwargs[k] = v[:c[1] - c[0]]
            predicate(**kwargs)

            # TODO: write back to the writer
            for k in output_arrays.keys():
                output_writers[k].write_part(kwargs[k])
        for k, v in output_writers.items():
            output_writers[k].flush()


    def get_index(self, target, foreign_key, destination=None):
        # print('  building patient_id index')
        t0 = time.time()
        target_lookup = dict()
        for i, v in enumerate(target[:]):
            target_lookup[v] = i
        # print(f'  target lookup built in {time.time() - t0}s')

        # print('  perform initial index')
        t0 = time.time()
        foreign_key_elems = foreign_key[:]
        # foreign_key_index = np.asarray([target_lookup.get(i, -1) for i in foreign_key_elems],
        #                                    dtype=np.int64)
        foreign_key_index = np.zeros(len(foreign_key_elems), dtype=np.int64)

        current_invalid = np.int64(INVALID_INDEX)
        for i_k, k in enumerate(foreign_key_elems):
            index = target_lookup.get(k, current_invalid)
            if index >= INVALID_INDEX:
                current_invalid += 1
                target_lookup[k] = index
            foreign_key_index[i_k] = index
        # print(f'  initial index performed in {time.time() - t0}s')

        if destination:
            destination.write(foreign_key_index)
        else:
            return foreign_key_index


    def get_shared_index(self, keys):
        if not isinstance(keys, tuple):
            raise ValueError("'keys' must be a tuple")
        concatted = None
        for k in keys:
            raw_field = val.raw_array_from_parameter(self, 'keys', k)
            if concatted is None:
                concatted = pd.unique(raw_field)
            else:
                concatted = np.concatenate((concatted, raw_field), axis=0)
        concatted = pd.unique(concatted)
        concatted = np.sort(concatted)

        return tuple(np.searchsorted(concatted, k) for k in keys)


    def get_trash_group(self, group):

        group_names = group.name[1:].split('/')

        while True:
            id = str(uuid.uuid4())
            try:
                result = group.create_group(f"/trash/{'/'.join(group_names[:-1])}/{id}")
                return result
            except KeyError:
                pass


    def temp_filename(self):
        uid = str(uuid.uuid4())
        while os.path.exists(uid + '.hdf5'):
            uid = str(uuid.uuid4())
        return uid + '.hdf5'
