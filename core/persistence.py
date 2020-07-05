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

from threading import Thread
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
import time

import h5py
import numpy as np
from numba import jit, njit

DEFAULT_CHUNKSIZE = 1 << 20
INVALID_INDEX = 1 << 62

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
    'patient': (DEFAULT_CHUNKSIZE,), 'assessment': (DEFAULT_CHUNKSIZE,), 'test': (DEFAULT_CHUNKSIZE,)
}


class DataWriter:

    @staticmethod
    def _create_group(parent_group, name, attrs):
        group = parent_group.create_group(name)
        for k, v in attrs:
            group.attrs[k] = v
            group.attrs['completed'] = False

    @staticmethod
    def create_group(parent_group, name, attrs):
        t = Thread(target=DataWriter._create_group,
                   args=(parent_group, name, attrs))
        t.start()
        t.join()

    @staticmethod
    def write(group, name, field, count, dtype=None):
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
    def write_first(group, name, field, count, dtype=None):
        t = Thread(target=DataWriter._write_first,
                   args=(group, name, field, count, dtype))
        t.start()
        t.join()

    @staticmethod
    def _write_additional(group, name, field, count):
        gv = group[name]
        gv.resize((gv.size + count,))
        if count == len(field):
            gv[-count:] = field
        else:
            gv[-count:] = field[:count]

    @staticmethod
    def write_additional(group, name, field, count):
        t = Thread(target=DataWriter._write_additional,
                   args=(group, name, field, count))
        t.start()
        t.join()

    @staticmethod
    def _flush(group):
        group.attrs['completed'] = True

    @staticmethod
    def flush(group):
        t = Thread(target=DataWriter._flush, args=(group,))
        t.start()
        t.join()

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


@njit
def _not_equals(a, b, c):
    a_len = len(a)
    for i_r in range(a_len):
        c[i_r] = a[i_r] != b[i_r]


def _get_spans(field, fields):
    if field is None and fields is None:
        raise ValueError("One of 'field' and 'fields' must be set")
    if field is not None and fields is not None:
        raise ValueError("Only one of 'field' and 'fields' may be set")

    if field is not None:
        # return _get_spans_for_field(field)
        return _get_spans_for_field(field)
    elif len(fields) == 1:
        return _get_spans_for_field(fields[0])
    elif len(fields) == 2:
        return _get_spans_for_2_fields(*fields)
    else:
        raise NotImplementedError("This operation does not support more than two fields at present")


def _get_spans_for_field(field0):

    results = np.zeros(len(field0) + 1, dtype=np.bool)
    _not_equals(field0[:-1], field0[1:], results[1:])
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
                        max_index_i, max_value_i, s_start):
    separator = np.frombuffer(b',', dtype=np.uint8)[0]
    delimiter = np.frombuffer(b'"', dtype=np.uint8)[0]
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


# Newest
# ======
class Reader:
    def __init__(self, field):
        self.field = field


class IndexedStringReader(Reader):
    def __init__(self, datastore, field):
        Reader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype']
        if fieldtype != 'indexedstring':
            error = "'fieldtype of '{} should be 'indexedstring' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']
        self.datastore = datastore

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
                    bytestr[index[ir]-np.int64(startindex):
                            index[ir+1]-np.int64(startindex)].tobytes().decode()
            return results

    def __len__(self):
        return len(self.field['index']) - 1

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return IndexedStringWriter(self.datastore, dest_group, dest_name,
                                   timestamp, write_mode)

    def dtype(self):
        return self.field['index'].dtype, self.field['values'].dtype

    def sort(self, index, writer):
        field_index = self.field['index'][:]
        field_values = self.field['values'][:]
        r_field_index, r_field_values =\
            self.datastore.apply_sort_to_index_values(index, field_index, field_values)
        writer.write_raw(r_field_index, r_field_values)


class NumericReader(Reader):
    def __init__(self, datastore, field):
        Reader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype'].split(',')
        if fieldtype[0] != 'numeric':
            error = "'fieldtype of '{} should be 'numeric' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']
        self.datastore = datastore

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return NumericWriter(self.datastore, dest_group, dest_name, timestamp,
                             self.field.attrs['fieldtype'].split(',')[1], write_mode)

    def dtype(self):
        return self.field['values'].dtype


class CategoricalReader(Reader):
    def __init__(self, datastore, field):
        Reader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype']
        if fieldtype != 'categorical':
            error = "'fieldtype of '{} should be 'categorical' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']
        self.keys = self.field['keys'][()]
        self.datastore = datastore

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return CategoricalWriter(self.datastore, dest_group, dest_name, timestamp,
                                 {v: k for k, v in enumerate(self.field['keys'])}, write_mode)

    def dtype(self):
        return self.field['values'].dtype


class FixedStringReader(Reader):
    def __init__(self, datastore, field):
        Reader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype'].split(',')
        if fieldtype[0] != 'fixedstring':
            error = "'fieldtype of '{} should be 'numeric' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']
        self.datastore = datastore

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return FixedStringWriter(self.datastore, dest_group, dest_name, timestamp,
                                 self.field.attrs['fieldtype'].split(',')[1], write_mode)

    def dtype(self):
        return self.field['values'].dtype


class TimestampReader(Reader):
    def __init__(self, datastore, field):
        Reader.__init__(self, field)
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype'].split(',')
        if fieldtype[0] not in ('datetime', 'date', 'timestamp'):
            error = "'fieldtype of '{} should be 'datetime' or 'date' but is {}"
            raise ValueError(error.format(field, fieldtype))
        self.chunksize = field.attrs['chunksize']
        self.datastore = datastore

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def getwriter(self, dest_group, dest_name, timestamp, write_mode='write'):
        return TimestampWriter(self.datastore, dest_group, dest_name, timestamp,
                               write_mode)

    def dtype(self):
        return self.field['values'].dtype


write_modes = {'write', 'overwrite'}


class Writer:
    def __init__(self, datastore, group, name, write_mode, attributes):
        self.trash_field = None
        if write_mode not in write_modes:
            raise ValueError(f"'write_mode' must be one of {write_modes}")
        if name in group:
            if write_mode == 'overwrite':
                field = group[name]
                trash = datastore.get_trash_group(field)
                dest_name = trash.name + f"/{name.split('/')[-1]}"
                group.move(field.name, dest_name)
                self.trash_field = trash[name]
                field = group.create_group(name)
            else:
                error = (f"Field '{name}' already exists. Set 'write_mode' to 'overwrite' "
                         "if you want to overwrite the existing contents")
                raise KeyError(error)
        else:
            DataWriter.create_group(group, name, attributes)
            field = group[name]
        self.field = field
        self.name = name

    def flush(self):
        DataWriter.flush(self.field)
        if self.trash_field is not None:
            del self.trash_field


class IndexedStringWriter(Writer):
    def __init__(self, datastore, group, name, timestamp, write_mode='write'):
        fieldtype = f'indexedstring'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize)))
        self.fieldtype = fieldtype
        self.timestamp = timestamp
        self.datastore = datastore

        self.values = np.zeros(self.datastore.chunksize, dtype=np.uint8)
        self.indices = np.zeros(self.datastore.chunksize, dtype=np.int64)
        self.ever_written = False
        self.accumulated = 0
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
                if self.value_index == self.datastore.chunksize:
                    DataWriter.write(self.field, 'values', self.values, self.value_index)
                    self.value_index = 0
                self.accumulated += 1
            self.indices[self.index_index] = self.accumulated
            self.index_index += 1
            if self.index_index == self.datastore.chunksize:
                DataWriter.write(self.field, 'index', self.indices, self.index_index)
                self.index_index = 0

    def flush(self):
        if self.value_index != 0:
            DataWriter.write(self.field, 'values', self.values, self.value_index)
            self.value_index = 0
        if self.index_index != 0:
            DataWriter.write(self.field, 'index', self.indices, self.index_index)
            self.index_index = 0
        # self.field.attrs['fieldtype'] = self.fieldtype
        # self.field.attrs['timestamp'] = self.timestamp
        # self.field.attrs['chunksize'] = self.chunksize
        # self.field.attrs['completed'] = True
        super().flush()

    def write(self, values):
        self.write_part(values)
        self.flush()

    def write_part_raw(self, index, values):
        if index.dtype != np.int64:
            raise ValueError(f"'index' must be an ndarray of '{np.int64}'")
        if values.dtype != np.uint8:
            raise ValueError(f"'values' must be an ndarray of '{np.uint8}'")
        DataWriter.write(self.field, 'index', index, len(index))
        DataWriter.write(self.field, 'values', values, len(values))

    def write_raw(self, index, values):
        self.write_part_raw(index, values)
        self.flush()


# TODO: should produce a warning for unmappable strings and a corresponding filter, rather
# than raising an exception; or at least have a mode where this is possible
class CategoricalImporter:
    def __init__(self, datastore, group, name, timestamp, categories,
                 write_mode='write'):
        self.writer = CategoricalWriter(datastore, group, name,
                                        timestamp, categories, write_mode)
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


class CategoricalWriter(Writer):
    def __init__(self, datastore, group, name, timestamp, categories,
                 write_mode='write'):
        fieldtype = f'categorical'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize)))
        self.fieldtype = fieldtype
        self.timestamp = timestamp
        self.datastore = datastore
        self.keys = categories


    def chunk_factory(self, length):
        return np.zeros(length, dtype='uint8')

    def write_part(self, values):
        DataWriter.write(self.field, 'values', values, len(values))

    def flush(self):
        key_indices = [None] * len(self.keys)
        for k, v in self.keys.items():
            key_indices[v] = k
        DataWriter.write(self.field, 'keys', key_indices, len(self.keys),
                          dtype=h5py.string_dtype())
        # self.field.attrs['fieldtype'] = self.fieldtype
        # self.field.attrs['timestamp'] = self.timestamp
        # self.field.attrs['chunksize'] = self.chunksize
        # self.field.attrs['completed'] = True
        super().flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


class NumericImporter:
    def __init__(self, datastore, group, name, timestamp, nformat,
                 parser, write_mode='write'):
        self.data_writer = NumericWriter(datastore, group, name,
                                         timestamp, nformat, write_mode)
        self.flag_writer = NumericWriter(datastore, group, f"{name}_valid",
                                         timestamp, 'bool', write_mode)
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


class NumericWriter(Writer):
    def __init__(self, datastore, group, name, timestamp, nformat,
                 write_mode='write'):
        fieldtype = f'numeric,{nformat}'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize), ('nformat', nformat)))
        self.fieldtype = fieldtype
        self.nformat = nformat
        self.timestamp = timestamp
        self.datastore = datastore

    def chunk_factory(self, length):
        nformat = self.fieldtype.split(',')[1]
        return np.zeros(length, dtype=nformat)

    def write_part(self, values):
        DataWriter.write(self.field, 'values', values, len(values))

    def flush(self):
        # self.field.attrs['fieldtype'] = self.fieldtype
        # self.field.attrs['timestamp'] = self.timestamp
        # self.field.attrs['chunksize'] = self.chunksize
        # self.field.attrs['nformat'] = self.nformat
        # self.field.attrs['completed'] = True
        super().flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


class FixedStringWriter(Writer):
    def __init__(self, datastore, group, name, timestamp, strlen,
                 write_mode='write'):
        fieldtype = f'fixedstring,{strlen}'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize), ('strlen', strlen)))
        self.fieldtype = fieldtype
        self.timestamp = timestamp
        self.datastore = datastore
        self.strlen = strlen

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'S{self.strlen}')

    def write_part(self, values):
        DataWriter.write(self.field, 'values', values, len(values))

    def flush(self):
        # self.field.attrs['fieldtype'] = self.fieldtype
        # self.field.attrs['timestamp'] = self.timestamp
        # self.field.attrs['chunksize'] = self.chunksize
        # self.field.attrs['strlen'] = self.strlen
        # self.field.attrs['completed'] = True
        super().flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


class DateTimeImporter:
    def __init__(self, datastore, group, name, timestamp, optional=True,
                 write_mode='write'):
        self.datetime = DateTimeWriter(datastore, group, name,
                                       timestamp, write_mode)
        self.datestr = FixedStringWriter(datastore, group, f"{name}_day",
                                         timestamp, '10', write_mode)
        self.datetimeset = None
        if optional:
            self.datetimeset = NumericWriter(datastore, group, f"{name}_set",
                                             timestamp, 'bool', write_mode)

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
class DateTimeWriter(Writer):
    def __init__(self, datastore, group, name, timestamp, write_mode='write'):
        fieldtype = f'datetime'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize)))
        self.fieldtype = fieldtype
        self.timestamp = timestamp
        self.datastore = datastore

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
        DataWriter.write(self.field, 'values', timestamps, len(timestamps))

    def flush(self):
        # self.field.attrs['fieldtype'] = self.fieldtype
        # self.field.attrs['timestamp'] = self.timestamp
        # self.field.attrs['chunksize'] = self.chunksize
        # self.field.attrs['completed'] = True
        super().flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


class DateWriter(Writer):
    def __init__(self, datastore, group, name, timestamp, write_mode='write'):
        fieldtype = 'date'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize)))
        self.fieldtype = fieldtype
        self.timestamp = timestamp
        self.datastore = datastore

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'S10')

    def write_part(self, values):

        timestamps = np.zeros(len(values), dtype=np.float64)
        for i in range(len(values)):
            value = values[i]
            if value == b'':
                timestamps[i] = 0
            else:
                ts = datetime.strptime(value.decode(), '%Y-%m-%d')
                timestamps[i] = ts.timestamp()
        DataWriter.write(self.field, 'values', timestamps, len(timestamps))

    def flush(self):
        # self.field.attrs['fieldtype'] = self.fieldtype
        # self.field.attrs['timestamp'] = self.timestamp
        # self.field.attrs['chunksize'] = self.chunksize
        # self.field.attrs['completed'] = True
        super().flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


class TimestampWriter(Writer):
    def __init__(self, datastore, group, name, timestamp, write_mode='write'):
        fieldtype = 'timestamp'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize)))
        self.fieldtype = fieldtype
        self.timestamp = timestamp
        self.datastore = datastore

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'float64')

    def write_part(self, values):
        DataWriter.write(self.field, 'values', values, len(values))

    def flush(self):
        # self.field.attrs['fieldtype'] = self.fieldtype
        # self.field.attrs['timestamp'] = self.timestamp
        # self.field.attrs['chunksize'] = self.chunksize
        # self.field.attrs['completed'] = True
        Writer.flush(self)

    def write(self, values):
        self.write_part(values)
        self.flush()


class OptionalDateImporter:
    def __init__(self, datastore, group, name, timestamp,
                 optional=True, write_mode='write'):
        self.date = DateWriter(datastore, group, name, timestamp, write_mode)
        self.datestr = FixedStringWriter(datastore, group, f"{name}_day",
                                         timestamp, '10', write_mode)
        self.dateset = None
        if optional:
            self.dateset =\
                NumericWriter(datastore, group, f"{name}_set", timestamp, 'bool', write_mode)

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


def _aggregate_impl(predicate, fkey_indices=None, fkey_index_spans=None,
                    reader=None, writer=None, result_dtype=None):
    if fkey_indices is None and fkey_index_spans is None:
        raise ValueError("One of 'fkey_indices' or 'fkey_index_spans' must be set")
    if fkey_indices is not None and fkey_index_spans is not None:
        raise ValueError("Only one of 'fkey_indices' and 'fkey_index_spans' may be set")
    if writer is not None:
        if not isinstance(writer, (Writer, np.ndarray)):
            raise ValueError("'writer' must be either a Writer or an ndarray instance")

    if fkey_index_spans is None:
        fkey_index_spans = _get_spans(field=fkey_indices)

    if isinstance(writer, np.ndarray):
        if len(writer) != len(fkey_index_spans) - 1:
            error = "'writer': ndarray must be of length {} but is of length {}"
            raise ValueError(error.format(len(fkey_index_spans) - 1), len(writer))
        elif writer.dtype != result_dtype:
            raise ValueError(f"'writer' dtype must be {result_dtype} but is {writer.dtype}")

    if isinstance(writer, Writer) or writer is None:
        results = np.zeros(len(fkey_index_spans) - 1, dtype=result_dtype)
    else:
        results = writer

    # execute the predicate (note that not every predicate requires a reader)
    predicate(fkey_index_spans, reader, results)

    if isinstance(writer, Writer):
        writer.write(results)

    return writer if writer is not None else results


class DataStore:

    def __init__(self, chunksize=DEFAULT_CHUNKSIZE):
        self.chunksize = chunksize


    # TODO: fields is being ignored at present
    def sort_on(self, src_group, dest_group, keys, fields=None,
                timestamp=datetime.now(timezone.utc), write_mode='write'):
        # sort_keys = ('patient_id', 'created_at')
        readers = tuple(self.get_reader(src_group[f]) for f in keys)
        t1 = time.time()
        sorted_index = self.dataset_sort(
            np.arange(len(readers[0]), dtype=np.uint32), readers)
        print(f'sorted {keys} index in {time.time() - t1}s')

        t0 = time.time()
        for k in src_group.keys():
            t1 = time.time()
            r = self.get_reader(src_group[k])
            w = r.getwriter(dest_group, k, timestamp, write_mode=write_mode)
            self.apply_sort(sorted_index, r, w)
            del r
            del w
            print(f"  '{k}' reordered in {time.time() - t1}s")
        print(f"fields reordered in {time.time() - t0}s")


    def dataset_sort(self, index, readers):
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


    # TODO: index should be able to be either a reader or an ndarray
    def apply_sort(self, index, reader, writer):
        if isinstance(reader, IndexedStringReader):
            src_indices = reader.field['index'][:]
            src_values = reader.field.get('values', np.zeros(0, dtype='S1'))[:]
            indices, values = _apply_sort_to_index_values(index, src_indices, src_values)
            writer.write_raw(indices, values)
        elif isinstance(reader, Reader):
            result = _apply_sort_to_array(index, reader[:])
            writer.write(result)
        elif isinstance(reader, np.ndarray):
            result = _apply_sort_to_array(index, reader)
            writer.write(result)
        else:
            raise ValueError(f"'reader' must be a Reader or an ndarray, but is {type.reader}")


    # TODO: write filter with new readers / writers rather than deleting this
    def filter(self, dataset, field, name, predicate, timestamp=datetime.now(timezone.utc)):
        raise NotImplementedError()


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
        return _get_spans(field, fields)


    # TODO - for all apply_spans methods, spans should be able to be an ndarray
    def apply_spans_count(self, spans, _, writer):
        if isinstance(writer, Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_count(spans, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            _apply_spans_count(spans, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")

    def apply_spans_first(self, spans, reader, writer):
        if isinstance(reader, Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_first(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            _apply_spans_first(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_last(self, spans, reader, writer):
        if isinstance(reader, Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_last(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            _apply_spans_last(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_min(self, spans, reader, writer):
        if isinstance(reader, Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_min(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(reader, Reader):
            _apply_spans_min(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_max(self, spans, reader, writer):
        if isinstance(reader, Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            _apply_spans_max(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(reader, Reader):
            _apply_spans_max(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_concat(self, spans, reader, writer):
        if not isinstance(reader, IndexedStringReader):
            raise ValueError(f"'reader' must be one of 'IndexedStringReader' but is {type(reader)}")
        if not isinstance(writer, IndexedStringWriter):
            raise ValueError(f"'writer' must be one of 'IndexedStringWriter' but is {type(writer)}")

        src_index = reader.field['index'][:]
        src_values = reader.field['values'][:]
        dest_index = np.zeros(reader.chunksize, src_index.dtype)
        dest_values = np.zeros(reader.chunksize * 16, src_values.dtype)

        max_index_i = reader.chunksize
        max_value_i = reader.chunksize * 8
        s = 0
        while s < len(spans) - 1:
            s, index_i, index_v = _apply_spans_concat(spans, src_index, src_values,
                                                      dest_index, dest_values,
                                                      max_index_i, max_value_i, s)

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
        if not isinstance(reader, (Reader, np.ndarray)):
            raise ValueError(f"'reader' must be a Reader or an ndarray but is {type(reader)}")
        if isinstance(reader, Reader):
            required_dtype = reader.dtype()
        else:
            required_dtype = reader.dtype
        return _aggregate_impl(predicate, fkey_indices, fkey_index_spans,
                               reader, writer, required_dtype)


    def join(self,
             destination_pkey, fkey_indices, values_to_join,
             writer=None, fkey_index_spans=None):
        if fkey_indices is not None:
            if not isinstance(fkey_indices, (Reader, np.ndarray)):
                raise ValueError(f"'fkey_indices' must be a type of Reader or an ndarray")
        if values_to_join is not None:
            if not isinstance(values_to_join, (Reader, np.ndarray)):
                raise ValueError(f"'values_to_join' must be a type of Reader but is {type(values_to_join)}")
            if isinstance(values_to_join, IndexedStringReader):
                raise ValueError(f"Joins on indexed string fields are not supported")

        if isinstance(values_to_join, Reader):
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

        writer.write(destination_space_values)


    def predicate_and_join(self,
                           predicate, destination_pkey, fkey_indices,
                           reader=None, writer=None, fkey_index_spans=None):
        if reader is not None:
            if not isinstance(reader, Reader):
                raise ValueError(f"'reader' must be a type of Reader but is {type(reader)}")
            if isinstance(reader, IndexedStringReader):
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
            'indexedstring': IndexedStringReader,
            'fixedstring': FixedStringReader,
            'categorical': CategoricalReader,
            'boolean': NumericReader,
            'numeric': NumericReader,
            'datetime': TimestampReader,
            'date': TimestampReader,
            'timestamp': TimestampReader
        }

        fieldtype = field.attrs['fieldtype'].split(',')[0]
        return fieldtype_map[fieldtype](self, field)


    def get_indexed_string_writer(self, group, name, timestamp, writemode='write'):
        return IndexedStringWriter(self, group, name, timestamp, writemode)


    def get_fixed_string_writer(self, group, name, timestamp, width, writemode='write'):
        return FixedStringWriter(self, group, name, timestamp, width, writemode)


    def get_categorical_writer(self, group, name, timestamp, categories, writemode='write'):
        return CategoricalWriter(self, group, name, timestamp, categories, writemode)


    def get_numeric_writer(self, group, name, timestamp, dtype, writemode='write'):
        return NumericWriter(self, group, name, timestamp, dtype, writemode)


    def get_timestamp_writer(self, group, name, timestamp, writemode='write'):
        return TimestampWriter(self, group, name, timestamp, writemode)


    def get_compatible_writer(self, field, dest_group, dest_name, writemode='write'):
        reader = self.get_reader(field)
        return reader.get_writer(dest_group, dest_name, writemode)


    def get_or_create_group(self, group, name):
        if name in group:
            return group[name]
        return group.create_group(name)


    def chunks(self, length, chunksize):
        cur = 0
        while cur < length:
            next = min(length, cur + chunksize)
            yield cur, next
            cur = next

    def process(self, inputs, outputs, predicate):

        # TODO: modifying the dictionaries in place is not great
        input_readers = dict()
        for k, v in inputs.items():
            if isinstance(v, Reader):
                input_readers[k] = v
            else:
                input_readers[k] = self.get_reader(v)
        output_writers = dict()
        output_arrays = dict()
        for k, v in outputs.items():
            if isinstance(v, Writer):
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


    def get_index(self, target, foreign_key, destination):
        print('  building patient_id index')
        t0 = time.time()
        target_lookup = dict()
        for i, v in enumerate(target[:]):
            target_lookup[v] = i
        print(f'  target lookup built in {time.time() - t0}s')

        print('  perform initial index')
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
        print(f'  initial index performed in {time.time() - t0}s')

        destination.write(foreign_key_index)


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
