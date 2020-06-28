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
def distinct(field, filter=None):
    # raise NotImplementedError()
    return np.unique(field)
#     d = Series(field)
#     distinct_values = set()
#     if filter is not None:
#         f = Series(filter)
#         for i_r in range(len(d)):
#             if f[i_r] == 0:
#                 distinct_values.add(d[i_r])
#     else:
#         for i_r in range(len(d)):
#             distinct_values.add(d[i_r])
#
#     return distinct_values

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

        self.values = np.zeros(chunksize, dtype=np.byte)
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
