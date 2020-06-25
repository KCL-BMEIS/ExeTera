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




# Original Writers
# ================


class BaseWriter:
    def __init__(self, group, name, chunksize, fieldtype, timestamp):
        self.group = group.create_group(name)
        self.chunksize_ = chunksize
        self.fieldtype = fieldtype
        self.timestamp = timestamp

    def chunksize(self):
        return self.chunksize_

    def flush(self):
        self.group.attrs['fieldtype'] = self.fieldtype
        self.group.attrs['timestamp'] = self.timestamp
        self.group.attrs['chunksize'] = self.chunksize_
        self.group.attrs['completed'] = True


class IndexedStringWriter(BaseWriter):
    def __init__(self, group, chunksize, name, timestamp):
        BaseWriter.__init__(self, group, name, chunksize, "indexedstring", timestamp)
        self.name = name
        self.values = np.zeros(chunksize, dtype=np.byte)
        self.indices = np.zeros(chunksize, dtype=np.uint32)
        self.accumulated = 0
        self.value_index = 0
        self.indices[0] = self.accumulated
        self.index_index = 1

    def append(self, value):
        evalue = value.encode()
        for v in evalue:
            self.values[self.value_index] = v
            self.value_index += 1
            if self.value_index == self.chunksize_:
                DataWriter._write(self.group, 'values', self.values, self.value_index)
                self.value_index = 0
            self.accumulated += 1
        self.indices[self.index_index] = self.accumulated
        self.index_index += 1
        if self.index_index == self.chunksize_:
            DataWriter._write(self.group, 'index', self.indices, self.index_index)
            self.index_index = 0

    def flush(self):
        print(f'flush {self.name}: value_index =', self.value_index)
        if self.value_index != 0:
            DataWriter._write(self.group, 'values', self.values, self.value_index)
            self.value_index = 0
        if self.index_index != 0:
            DataWriter._write(self.group, 'index', self.indices, self.index_index)
            self.index_index = 0
        BaseWriter.flush(self)


class CategoricalWriter(BaseWriter):
    def __init__(self, group, chunksize, name, timestamp, categories):
        BaseWriter.__init__(self, group, name, chunksize, "categorical", timestamp)
        self.name = name
        max_len = 0
        for c in categories:
            max_len = max(max_len, len(c))
        self.keys = categories
        self.values = np.zeros(chunksize, dtype=np.ubyte)
        self.index = 0

    def append(self, value):
        self.values[self.index] = self.keys[value]
        self.index += 1

        if self.index == self.chunksize_:
            DataWriter._write(self.group, 'values', self.values, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'values', self.values, self.index)
            self.index = 0
        key_index = [None] * len(self.keys)
        for k, v in self.keys.items():
            try:
                key_index[v] = k
            except:
                print(f"{self.name}: index {v} is out of range for key {k}")
                raise
        DataWriter._write(self.group, 'keys', key_index, len(self.keys),
                          dtype=h5py.string_dtype())
        BaseWriter.flush(self)


class BooleanWriter(BaseWriter):

    def __init__(self, group, chunksize, name, timestamp):
        BaseWriter.__init__(self, group, name, chunksize, f"boolean", timestamp)
        self.name = name
        self.values = np.zeros(chunksize, dtype=np.bool)
        self.index = 0

    def append(self, value):
        self.values[self.index] = value
        self.index += 1

        if self.index == self.chunksize_:
            DataWriter._write(self.group, 'values', self.values, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'values', self.values, self.index)
        BaseWriter.flush(self)


def str_to_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def str_to_int(value):
    try:
        return int(value)
    except ValueError:
        return None

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


class NumericWriter(BaseWriter):
    # format_checks = {
    #     'bool': lambda x: bool(x),
    #     'uint8': lambda x: int(x), 'uint16': lambda x: int(x),
    #     'uint32': lambda x: int(x), 'uint64': lambda x: int(x),
    #     'int8': lambda x: int(x), 'int16': lambda x: int(x),
    #     'int32': lambda x: int(x), 'int64': lambda x: int(x),
    #     'float32': lambda x: float(x), 'float64': lambda x: float(x)
    # }
    supported_formats = ('bool',
                         'uint8', 'uint16', 'uint32', 'uint64',
                         'int8', 'int16', 'int32', 'int64',
                         'float32', 'float64')

    def __init__(self, group, chunksize, name, timestamp, nformat,
                 converter=None, needs_filter=False):
        BaseWriter.__init__(self, group, name, chunksize, f"numeric,{nformat}", timestamp)
        if nformat not in NumericWriter.supported_formats:
            error_str = "'nformat' must be one of {} but is {}"
            raise ValueError(error_str.format(NumericWriter.supported_formats, nformat))
        if converter is None:
            self.converter = lambda x: x
        else:
            self.converter = converter
        # self.converter =\
        #     lambda x: x if converter is None else converter
        self.name = name
        self.values = np.zeros(chunksize, dtype=nformat)
        self.needs_filter = needs_filter
        if self.needs_filter:
            self.filter = np.zeros(chunksize, dtype=np.bool)
        self.index = 0
        self.chunksize = chunksize

    def append(self, value):
        v = self.converter(value)
        # try:
        #     v = self.converter(value) if self.converter else
        #     f = False
        # except ValueError:
        #     v = 0
        #     f = True
        self.values[self.index] = 0 if v is None else v
        if self.needs_filter:
            f = v is None
            self.filter[self.index] = f
        self.index += 1
        if self.index == self.chunksize_:
            DataWriter._write(self.group, 'values', self.values, self.index)
            if self.needs_filter:
                DataWriter._write(self.group, 'filter', self.filter, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'values', self.values, self.index)
            if self.needs_filter:
                DataWriter._write(self.group, 'filter', self.filter, self.index)
            self.index = 0
        BaseWriter.flush(self)


class FixedStringWriter(BaseWriter):
    def __init__(self, group, chunksize, name, timestamp, strlen):
        BaseWriter.__init__(self, group, name, chunksize, f"fixedstring,{strlen}", timestamp)
        self.name = name
        self.values = np.zeros(chunksize, dtype=f"S{strlen}")
        self.index = 0

    def append(self, value):
        self.values[self.index] = value
        self.index += 1
        if self.index == self.chunksize_:
            DataWriter._write(self.group, 'values', self.values, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'values', self.values, self.index)
            self.index = 0
        BaseWriter.flush(self)


class DateWriter(BaseWriter):
    def __init__(self, group, chunksize, name, timestamp):
        BaseWriter.__init__(self, group, name, chunksize, "date", timestamp)
        self.name = name
        self.timestamps = np.zeros(chunksize, dtype='float64')
        self.years = np.zeros(chunksize, dtype='u2')
        self.months = np.zeros(chunksize, dtype='u1')
        self.days = np.zeros(chunksize, dtype='u1')
        self.index = 0

    def append(self, value):
        if value == '':
            self.timestamps[self.index] = 0
            self.years[self.index] = 0
            self.months[self.index] = 0
            self.days[self.index] = 0
        else:
            ts = datetime.strptime(value, '%Y-%m-%d')
            self.timestamps[self.index] = 0
            self.years[self.index] = ts.year
            self.months[self.index] = ts.month
            self.days[self.index] = ts.day
        self.index += 1

        if self.index == self.chunksize_:
            DataWriter._write(self.group, 'timestamps', self.timestamps, self.index)
            DataWriter._write(self.group, 'years', self.years, self.index)
            DataWriter._write(self.group, 'months', self.months, self.index)
            DataWriter._write(self.group, 'days', self.days, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'timestamps', self.timestamps, self.index)
            DataWriter._write(self.group, 'years', self.years, self.index)
            DataWriter._write(self.group, 'months', self.months, self.index)
            DataWriter._write(self.group, 'days', self.days, self.index)
            self.index = 0
        BaseWriter.flush(self)


class DatetimeWriter(BaseWriter):
    def __init__(self, group, chunksize, name, timestamp):
        BaseWriter.__init__(self, group, name, chunksize, "datetime", timestamp)
        self.name = name
        self.timestamps = np.zeros(chunksize, dtype='float64')
        self.years = np.zeros(chunksize, dtype='u2')
        self.months = np.zeros(chunksize, dtype='u1')
        self.days = np.zeros(chunksize, dtype='u1')
        self.hours = np.zeros(chunksize, dtype='u1')
        self.minutes = np.zeros(chunksize, dtype='u1')
        self.seconds = np.zeros(chunksize, dtype='u1')
        self.microseconds = np.zeros(chunksize, dtype='u4')
        self.utcoffsets = np.zeros(chunksize, dtype='u4')
        self.index = 0

    def append(self, value):
        fraction = None
        if value == '':
            self.timestamps[self.index] = 0
            self.years[self.index] = 0
            self.months[self.index] = 0
            self.days[self.index] = 0
            self.hours[self.index] = 0
            self.minutes[self.index] = 0
            self.seconds[self.index] = 0
            self.microseconds[self.index] = 0
            self.utcoffsets[self.index] = 0
        else:
            try:
                ts = datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f%z')
            except ValueError:
                ts = datetime.strptime(value, '%Y-%m-%d %H:%M:%S%z')
                fraction = 0

                #print(ts.microsecond)

            self.timestamps[self.index] = ts.timestamp()
            self.years[self.index] = ts.year
            self.months[self.index] = ts.month
            self.days[self.index] = ts.day
            self.hours[self.index] = ts.hour
            self.minutes[self.index] = ts.minute
            self.seconds[self.index] = ts.second
            self.microseconds[self.index] = fraction if fraction is not None else ts.microsecond
            utco = ts.utcoffset()
            self.utcoffsets[self.index] = utco.days * 86400 + utco.seconds
        self.index += 1

        if self.index == self.chunksize_:
            DataWriter._write(self.group, 'timestamps', self.timestamps, self.index)
            DataWriter._write(self.group, 'years', self.years, self.index)
            DataWriter._write(self.group, 'months', self.months, self.index)
            DataWriter._write(self.group, 'days', self.days, self.index)
            DataWriter._write(self.group, 'hours', self.hours, self.index)
            DataWriter._write(self.group, 'minutes', self.minutes, self.index)
            DataWriter._write(self.group, 'seconds', self.seconds, self.index)
            DataWriter._write(self.group, 'microseconds', self.microseconds, self.index)
            DataWriter._write(self.group, 'utcoffsets', self.utcoffsets, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'timestamps', self.timestamps, self.index)
            DataWriter._write(self.group, 'years', self.years, self.index)
            DataWriter._write(self.group, 'months', self.months, self.index)
            DataWriter._write(self.group, 'days', self.days, self.index)
            DataWriter._write(self.group, 'hours', self.hours, self.index)
            DataWriter._write(self.group, 'minutes', self.minutes, self.index)
            DataWriter._write(self.group, 'seconds', self.seconds, self.index)
            DataWriter._write(self.group, 'microseconds', self.microseconds, self.index)
            DataWriter._write(self.group, 'utcoffsets', self.utcoffsets, self.index)
            self.index = 0
        BaseWriter.flush(self)




# Fast Writers
# ============


class IndexedStringWriter2:
    def __init__(self, group, chunksize, name, timestamp):
        BaseWriter.__init__(self, group, name, chunksize, "indexedstring", timestamp)
        self.name = name
        self.values = np.zeros(chunksize, dtype=np.byte)
        self.indices = np.zeros(chunksize, dtype=np.uint32)
        self.accumulated = 0
        self.value_index = 0
        self.indices[0] = self.accumulated
        self.index_index = 1

    def append(self, values):
        for s in values:
            evalue = s.encode()
            for v in evalue:
                self.values[self.value_index] = v
                self.value_index += 1
                if self.value_index == self.chunksize_:
                    DataWriter._write(self.group, 'values', self.values, self.value_index)
                    self.value_index = 0
                self.accumulated += 1
            self.indices[self.index_index] = self.accumulated
            self.index_index += 1
            if self.index_index == self.chunksize_:
                DataWriter._write(self.group, 'index', self.indices, self.index_index)
                self.index_index = 0

    def flush(self):
        print(f'flush {self.name}: value_index =', self.value_index)
        if self.value_index != 0:
            DataWriter._write(self.group, 'values', self.values, self.value_index)
            self.value_index = 0
        if self.index_index != 0:
            DataWriter._write(self.group, 'index', self.indices, self.index_index)
            self.index_index = 0
        BaseWriter.flush(self)


class CategoricalWriter2:
    def __init__(self, group, chunksize, name, timestamp, categories):
        self.values = np.zeros(chunksize, dtype='uint8')
        self.group = group.create_group(name)
        max_len = 0
        for c in categories:
            max_len = max(max_len, len(c))
        self.fieldtype = f'numeric,{nformat}'
        self.timestamp = timestamp
        self.chunksize = chunksize

    def write_chunk(self, count=None):
        count = len(self.values) if count is None else count
        DataWriter._write(self.group, 'values', self.values, count)
        if self.filter:
            DataWriter._write(self.group, 'filter', self.filter, count)

    def flush(self, count=None):
        self.group.create_dataset['keys']
        self.group.attrs['fieldtype'] = self.fieldtype
        self.group.attrs['timestamp'] = self.timestamp
        self.group.attrs['chunksize'] = self.chunksize
        self.group.attrs['completed'] = True


class NumericWriter2:
    def __init__(self, group, chunksize, name, timestamp, nformat, needs_filter=False):
        if needs_filter:
            self.filter = np.zeros(chunksize, dtype=np.bool)
        else:
            self.filter = None
        self.values = np.zeros(chunksize, dtype=nformat)
        self.group = group.create_group(name)
        self.fieldtype = f'numeric,{nformat}'
        self.timestamp = timestamp
        self.chunksize = chunksize

    def write_chunk(self, count=None):
        count = len(self.values) if count is None else count
        DataWriter._write(self.group, 'values', self.values, count)
        if self.filter:
            DataWriter._write(self.group, 'filter', self.filter, count)

    def flush(self, count=None):
        self.group.attrs['fieldtype'] = self.fieldtype
        self.group.attrs['timestamp'] = self.timestamp
        self.group.attrs['chunksize'] = self.chunksize
        self.group.attrs['completed'] = True




# Readers
# =======


def _chunkcount(dataset, chunksize, istart=0, iend=None):
    if iend is None:
        iend = dataset.size
    requested_size = iend - istart
    chunkmax = int(requested_size / chunksize)
    if requested_size % chunksize != 0:
        chunkmax += 1
    return chunkmax


def _slice_for_chunk(c, dataset, chunksize, istart=0, iend=None):
    if iend is None:
        iend = len(dataset)
    requested_size = iend - istart
    # if c == chunkmax - 1:
    if c >= _chunkcount(dataset, chunksize, istart, iend):
        raise ValueError("Asking for out of range chunk")

    if istart + (c + 1) * chunksize> iend:
        length = requested_size % chunksize
    else:
        length = chunksize
    return istart + c * chunksize, istart + c * chunksize + length


def iterator(field):
    iterator_map = {
        'indexedstring': indexed_string_iterator,
        'fixedstring': fixed_string_iterator,
        'categorical': categorical_iterator,
        # 'boolean': boolean_iterator,
        'numeric': numeric_iterator,
        'datetime': timestamps_iterator,
        'date': timestamps_iterator
    }
    fieldtype = field.attrs['fieldtype'].split(',')[0]
    return iterator_map[fieldtype](field)


def reader(field, istart=0, iend=None):
    getter_map = {
        'indexedstring': indexed_string_reader,
        'fixedstring': fixed_string_reader,
        'categorical': categorical_reader,
        'boolean': boolean_reader,
        'numeric': numeric_reader,
        'datetime': timestamps_reader,
        'date': timestamps_reader
    }
    fieldtype = field.attrs['fieldtype'].split(',')[0]
    return getter_map[fieldtype](field, istart, iend)


def indexed_string_iterator(field):
    if field.attrs['fieldtype'].split(',')[0] != 'indexedstring':
        raise ValueError(
            f"{field} must be 'indexedstring' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']

    index = field['index']
    ichunkmax = _chunkcount(index, chunksize)

    values = field['values']
    vchunkmax = _chunkcount(values, chunksize)

    hackpadding = 10000
    vc = 0
    vistart, viend = _slice_for_chunk(vc, vchunkmax, values, chunksize)
    vcur = values[vistart:min(viend+hackpadding, values.size)]
    lastindex, curindex = None, None
    for ic in range(ichunkmax):
        istart, iend = _slice_for_chunk(ic, ichunkmax, index, chunksize)
        icur = index[istart:iend]
        if istart == 0:
            lastindex = icur[0]
            inchunkstart = 1
        else:
            inchunkstart = 0
        for i in range(inchunkstart, len(icur)):
            curindex = icur[i]
            relativelastindex = lastindex - vistart
            relativecurindex = curindex - vistart

            yield vcur[relativelastindex:relativecurindex].tostring().decode()

            if curindex >= viend:
                vc += 1
                vistart, viend = _slice_for_chunk(vc, vchunkmax, values, chunksize)
                vcur = values[vistart:min(viend+hackpadding, values.size)]
            lastindex = curindex


def indexed_string_reader(field, istart=0, iend=None):
    if field.attrs['fieldtype'].split(',')[0] != 'indexedstring':
        raise ValueError(
            f"{field} must be 'indexedstring' but is {field.attrs['fieldtype']}")

    iend = field['values'].size-1
    indices = field['index'][istart:iend+1]
    values = field['values'][indices[istart]:indices[iend+1]]
    results = [None] * len(indices) - 1
    for i in range(0, len(indices)-1):
        results[i] = values[indices[i], indices[i+1]].tostring().decode()
    return results


def fixed_string_iterator(field):
    if field.attrs['fieldtype'].split(',')[0] != 'fixedstring':
        raise ValueError(
            f"{field} must be 'fixedstring' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']

    values = field['values']
    chunkmax = _chunkcount(values, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, values, chunksize)
        vcur = values[istart:iend]
        for i in range(len(vcur)):
            yield vcur[i].tobytes().decode()


def fixed_string_reader(field, istart=0, iend=None):
    if field.attrs['fieldtype'].split(',')[0] != 'fixedstring':
        raise ValueError(
            f"{field} must be 'fixedstring' but is {field.attrs['fieldtype']}")
    iend = field['values'].size
    fieldlength = field.attrs['fieldtype'].split(',')[1]
    values = field['values'][istart:iend]
    results = np.zeros(len(values), dtype=f'U{fieldlength}')
    results = [v.tostring().decode() for v in values]
    return results


def categorical_iterator(field):
    if field.attrs['fieldtype'] != 'categorical':
        raise ValueError(
            f"{field} must be 'categorical' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']

    values = field['values']
    chunkmax = _chunkcount(values, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, values, chunksize)
        vcur = values[istart:iend]
        # for i in range(len(vcur)):
        #     yield vcur[i]
        for v in vcur:
            yield v


def categorical_reader(field, istart=0, iend=None):
    if field.attrs['fieldtype'] != 'categorical':
        raise ValueError(
            f"{field} must be 'categorical' but is {field.attrs['fieldtype']}")
    iend = field['values'].size
    values = field['values'][istart:iend]
    return values


def boolean_reader(field, istart=0, iend=None, invalid=None):
    if field.attrs['fieldtype'].split(',')[0] != 'numeric':
        raise ValueError(
            f"{field} must be 'boolean' but is {field.attrs['fieldtype']}")
    if iend is None:
        iend = field['values'].size
    values = field['values'][istart:iend]
    filter = field['filter'][istart:iend]
    if invalid is not None:
        for i_v in range(len(values)):
            if filter[i_v]:
                values[i_v] = invalid
    return values


def numeric_iterator(field, invalid=None):
    if field.attrs['fieldtype'].split(',')[0] != 'numeric':
        raise ValueError(
            f"{field} must be 'numeric' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']

    values = field['values']
    filter = field['filter']
    chunkmax = _chunkcount(values, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, values, chunksize)
        vcur = values[istart:iend]
        vflt = filter[istart:iend]
        for i in range(len(vcur)):
            if vflt[i] == False:
                yield vflt[i], vcur[i]
            else:
                yield vflt[i], invalid


def numeric_reader(field, istart=0, iend=None, invalid=None):
    if field.attrs['fieldtype'].split(',')[0] != 'numeric':
        raise ValueError(
            f"{field} must be 'numeric' but is {field.attrs['fieldtype']}")
    if iend is None:
        iend = field['values'].size
    values = field['values'][istart:iend]
    filter = field['filter'][istart:iend]
    if invalid is not None:
        for i_v in range(len(values)):
            if filter[i_v]:
                values[i_v] = invalid
    return values


def timestamps_iterator(field):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    chunksize = field.attrs['chunksize']
    timestamps = field['timestamps']
    chunkmax = _chunkcount(timestamps, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, timestamps, chunksize)
        tcur = timestamps[istart:iend]
        for i in range(len(tcur)):
            yield tcur[i]


def timestamps_reader(field, istart=0, iend=None):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['timestamps'].size
    timestamps = field['timestamps'][istart:iend]
    return timestamps


def years_iterator(field):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    chunksize = field.attrs['chunksize']
    years = field['years']
    chunkmax = _chunkcount(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, years, chunksize)
        ycur = years[istart:iend]
        for i in range(len(ycur)):
            yield ycur[i]


def years_getter(field, istart=0, iend=None):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['years'].size
    years = field['years'][istart:iend]
    return years


def months_iterator(field):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    chunkmax = _chunkcount(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, years, chunksize)
        ycur = years[istart:iend]
        mcur = months[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mcur[i])


def months_getter(field, istart=0, iend=None):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['years'].size
    years = field['years'][istart:iend]
    months = field['months'][istart:iend]
    results = zip(years, months)
    return results


def days_iterator(field):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    days = field['days']
    chunkmax = _chunkcount(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, years, chunksize)
        ycur = years[istart:iend]
        mcur = months[istart:iend]
        dcur = days[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mcur[i], dcur[i])


def days_getter(field, istart=0, iend=None):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['years'].size
    years = field['years'][istart:iend]
    months = field['months'][istart:iend]
    days = field['days'][istart:iend]
    return zip(years, months, days)


def hours_iterator(field):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    days = field['days']
    hours = field['hours']
    chunkmax = _chunkcount(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, years, chunksize)
        ycur = years[istart:iend]
        mcur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mcur[i], dcur[i], hcur[i])


def hours_getter(field, istart=0, iend=None):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['years'].size
    years = field['years'][istart:iend]
    months = field['months'][istart:iend]
    days = field['days'][istart:iend]
    hours = field['hours'][istart:iend]
    return zip(years, months, days, hours)


def minutes_iterator(field):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    days = field['days']
    hours = field['hours']
    minutes = field['minutes']
    chunkmax = _chunkcount(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, years, chunksize)
        ycur = years[istart:iend]
        mocur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        micur = minutes[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mocur[i], dcur[i], hcur[i], micur[i])


def minutes_getter(field, istart=0, iend=None):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['years'].size
    years = field['years'][istart:iend]
    months = field['months'][istart:iend]
    days = field['days'][istart:iend]
    hours = field['hours'][istart:iend]
    minutes = field['minutes'][istart:iend]
    return zip(years, months, days, hours, minutes)


def seconds_iterator(field):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    days = field['days']
    hours = field['hours']
    minutes = field['minutes']
    seconds = field['seconds']
    chunkmax = _chunkcount(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, years, chunksize)
        ycur = years[istart:iend]
        mocur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        micur = minutes[istart:iend]
        scur = seconds[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mocur[i], dcur[i], hcur[i], micur[i], scur[i])


def seconds_getter(field, istart=0, iend=None):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['years'].size
    years = field['years'][istart:iend]
    months = field['months'][istart:iend]
    days = field['days'][istart:iend]
    hours = field['hours'][istart:iend]
    minutes = field['minutes'][istart:iend]
    seconds = field['seconds'][istart:iend]
    return zip(years, months, days, hours, minutes, seconds)


def microseconds_iterator(field):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    days = field['days']
    hours = field['hours']
    minutes = field['minutes']
    seconds = field['seconds']
    microseconds = field['microseconds']
    chunkmax = _chunkcount(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, years, chunksize)
        ycur = years[istart:iend]
        mocur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        micur = minutes[istart:iend]
        scur = seconds[istart:iend]
        mscur = microseconds[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mocur[i], dcur[i], hcur[i], micur[i], scur[i], mscur[i])


def microseconds_getter(field, istart=0, iend=None):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")

    if iend is None:
        iend = field['years'].size
    years = field['years'][istart:iend]
    months = field['months'][istart:iend]
    days = field['days'][istart:iend]
    hours = field['hours'][istart:iend]
    minutes = field['minutes'][istart:iend]
    seconds = field['seconds'][istart:iend]
    microseconds = field['microseconds'][istart:iend]
    return zip(years, months, days, hours, minutes, seconds, microseconds)


# def dataset_sort(index, readers):
#     r_readers = reversed(readers)
#
#     for f in r_readers:
#         fdata = f[:]
#         index = sorted(index, key=lambda x: fdata[x])
#     return np.asarray(index, dtype=np.uint32)

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


def dataset_merge_sort(group, index, fields):
    def sort_comparison(*args):
        if len(args) == 1:
            a0 = args[0]
            def _inner(r):
                return a0[r]
            return _inner
        if len(args) == 2:
            a0 = args[0]
            a1 = args[1]
            def _inner(r):
                return a0[r], a1[r]
            return _inner
        if len(args) == 3:
            a0 = args[0]
            a1 = args[1]
            a2 = args[2]
            def _inner(r):
                return a0[r], a1[r], a2[r]
            return _inner
        if len(args) > 3:
            def _inner(r):
                return tuple(a[r] for a in args)
            return _inner

    def sort_function(index, fields):
        sort_group = temp_dataset()

        # sort each chunk individually
        chunksize = 1 << 24
        chunkcount = _chunkcount(index, chunksize)
        for c in range(chunkcount):
            istart, iend = _slice_for_chunk(c, index, chunksize)
            length = iend - istart
            fieldchunks = [None] * len(fields)
            indexchunk = index[istart:iend]
            for i_f, f in enumerate(fields):
                fc = reader(f, istart, iend)
                fieldchunks[i_f] = fc
            sfn = sort_comparison(*fieldchunks)
            sindexchunk = sorted(indexchunk, key=sfn)
            sort_group.create_dataset(f'chunk{c}', (length,), data=sindexchunk)

    sort_function(index, fields)

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


# TODO: indexed chunks can be written do, but you have to append to the end and
# then resort
class IndexedSeries:
    def __init__(self, dataset, chunksize=DEFAULT_CHUNKSIZE):
        self.dataset = dataset
        self.istart = None
        self.iend = None
        self.curchunk = None
        self.curchunkindex = None
        self.curchunkdirty = None
        self.chunksize_ = chunksize

    def __getitem__(self, index):
        self._updatechunk(index)
        return self.curchunk[index - self.istart]

    def __len__(self):
        return self.dataset

    def chunksize(self):
        return self.chunksize_

    def _updatechunk(self, index):
        nextchunkindex = self._chunkindex(index)
        if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
            istart, iend = _slice_for_chunk(nextchunkindex, self.dataset, self.chunksize_)
            nextchunk = self.dataset[istart:iend]
            # if self.curchunkdirty is True:
            #     self.dataset[self.istart:self.iend] = self.curchunk
            self.curchunkdirty = False
            self.curchunk = nextchunk
            self.curchunkindex = nextchunkindex
            self.istart = istart
            self.iend = iend

    def _chunkindex(self, iarg):
        return int(iarg / self.chunksize())


class Series:

    def __init__(self, dataset, chunksize=DEFAULT_CHUNKSIZE):
        self.dataset = dataset
        self.istart = None
        self.iend = None
        self.curchunk = None
        self.curchunkindex = None
        self.curchunkdirty = None
        self.chunksize_ = chunksize

    def __getitem__(self, index):
        self._updatechunk(index)
        return self.curchunk[index - self.istart]

    def __setitem__(self, index, value):
        self._updatechunk(index)
        self.curchunk[index - self.istart] = value

    def __len__(self):
        return self.dataset.size

    def chunksize(self):
        return self.chunksize_

    def _updatechunk(self, index):
        nextchunkindex = self._chunkindex(index)
        if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
            istart, iend = _slice_for_chunk(nextchunkindex, self.dataset, self.chunksize_)
            nextchunk = self.dataset[istart:iend]
            if self.curchunkdirty is True:
                self.dataset[self.istart:self.iend] = self.curchunk
            self.curchunkdirty = False
            self.curchunk = nextchunk
            self.curchunkindex = nextchunkindex
            self.istart = istart
            self.iend = iend

    def _chunkindex(self, iarg):
        return int(iarg / self.chunksize())


def filter(dataset, field, name, predicate, timestamp=datetime.now(timezone.utc)):
    c = Series(field)
    writer = BooleanWriter(dataset, DEFAULT_CHUNKSIZE, name, timestamp)
    for r in c:
        writer.append(predicate(r))
    writer.flush()
    return dataset[name]


def distinct(dataset, field, name, filter=None, timestamp=datetime.now(timezone.utc)):
    d = Series(field)
    distinct_values = set()
    if filter is not None:
        f = Series(filter)
        for i_r in range(len(d)):
            if f[i_r] == 0:
                distinct_values.add(d[i_r])
    else:
        for i_r in range(len(d)):
            distinct_values.add(d[i_r])

    return distinct_values


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


# def get_field(dataset, field_name):
#     if field_name not in dataset.keys():
#         error = "{} is not a field in {}. The following fields are available: {}"
#         raise ValueError(error.format(field_name, dataset, dataset.keys()))
#     return
#     return get_reader_from_field(dataset[field_name])


def get_writer_from_field(field, dest_group, dest_name):
    reader = get_reader_from_field(field)
    return reader.get_writer(dest_group, dest_name)


class IndexedStringReader:
    def __init__(self, field, converter=None):
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype']
        if fieldtype != 'indexedstring':
            error = "'fieldtype' of '{}' should be 'indexedstring' but is {}"
            raise ValueError(error.format(field, fieldtype))

        self.index_dataset = field['index']
        self.value_dataset = field['values']
        self.istart = None
        self.iend = None
        self.curchunkindex = None
        self.chunksize_ = field.attrs['chunksize']
        if converter is None:
            self.converter = lambda x: x.tostring()
        else:
            self.converter = converter

    def __getitem__(self, index):
        self._updatechunk(index)
        vi = index - self.istart
        vstart = self.curindexchunk[0]
        return self.converter(
            self.curvaluechunk[self.curindexchunk[vi]-vstart:
                               self.curindexchunk[vi+1]-vstart])

    def __len__(self):
        return self.index_dataset.size - 1

    def _updatechunk(self, index):
        nextchunkindex = self._chunkindex(index)
        if self.curchunkindex != nextchunkindex:
            istart, iend =\
                _slice_for_chunk(nextchunkindex, self.index_dataset, self.chunksize_)
            nextindexchunk = self.index_dataset[istart:iend+1]
            nextvaluechunk = self.value_dataset[nextindexchunk[0]:nextindexchunk[-1]]
            self.curchunkindex = nextchunkindex
            self.curindexchunk = nextindexchunk
            self.curvaluechunk = nextvaluechunk
            self.istart = istart
            self.iend = iend

    def _chunkindex(self, index):
        return int(index / self.chunksize_)


class FixedStringReader:
    def __init__(self, field, as_string=False):
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype, fieldlen = field.attrs['fieldtype'].split(',')
        if fieldtype != 'fixedstring':
            error = "'fieldtype' of '{}' should be 'fixedstring' but is {}"
            raise ValueError(error.format(field, fieldtype))

        self.dataset = field['values']
        self.istart = None
        self.iend = None
        self.curchunk = None
        self.curchunkindex = None
        self.curchunkdirty = None
        self.chunksize_ = field.attrs['chunksize']
        if as_string:
            self.converter = lambda x: x.decode()
        else:
            self.converter = lambda x: x

    def __getitem__(self, index):
        self._updatechunk(index)
        return self.converter(self.curchunk[index - self.istart])

    def __len__(self):
        return self.dataset.size

    def _updatechunk(self, index):
        nextchunkindex = self._chunkindex(index)
        if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
            istart, iend = _slice_for_chunk(nextchunkindex, self.dataset, self.chunksize_)
            nextchunk = self.dataset[istart:iend]
            if self.curchunkdirty is True:
                self.dataset[self.istart:self.iend] = self.curchunk
            self.curchunkdirty = False
            self.curchunk = nextchunk
            self.curchunkindex = nextchunkindex
            self.istart = istart
            self.iend = iend

    def _chunkindex(self, index):
        return int(index / self.chunksize_)


class CategoricalReader:

    def __init__(self, field, as_string=False):
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype']
        if fieldtype != 'categorical':
            error = "'fieldtype of '{} should be 'categorical' but is {}"
            raise ValueError(error.format(field, fieldtype))

        self.dataset = field['values']
        self.key = field['keys']
        self.istart = None
        self.iend = None
        self.curchunk = None
        self.curchunkindex = None
        self.curchunkdirty = None
        self.chunksize_ = field.attrs['chunksize']
        if as_string:
            self.converter = lambda x: self.key[x]
        else:
            self.converter = lambda x: x

    def __getitem__(self, index):
        self._updatechunk(index)
        return self.converter(self.curchunk[index - self.istart])

    def __len__(self):
        return self.dataset.size

    def _updatechunk(self, index):
        nextchunkindex = self._chunkindex(index)
        if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
            istart, iend = _slice_for_chunk(nextchunkindex, self.dataset, self.chunksize_)
            nextchunk = self.dataset[istart:iend]
            if self.curchunkdirty is True:
                self.dataset[self.istart:self.iend] = self.curchunk
            self.curchunkdirty = False
            self.curchunk = nextchunk
            self.curchunkindex = nextchunkindex
            self.istart = istart
            self.iend = iend

    def _chunkindex(self, index):
        return int(index / self.chunksize_)


class NumericReader:

    def __init__(self, field, flagged_value=None):
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype'].split(',')
        if fieldtype[0] != 'numeric':
            error = "'fieldtype of '{} should be 'categorical' but is {}"
            raise ValueError(error.format(field, fieldtype))

        self.dataset = field['values']
        self.flags = None
        if 'filter' in field.keys():
            self.flags = field['filter']
        self.istart = None
        self.iend = None
        self.curchunk = None
        self.curflags = None
        self.curchunkindex = None
        self.curchunkdirty = None
        self.chunksize_ = field.attrs['chunksize']
        if self.flags is not None:
            self.converter = lambda i, v, f: flagged_value if f[i] else v[i]
        else:
            self.converter = lambda i, v, _: v[i]

    def __getitem__(self, index):
        self._updatechunk(index)
        return self.converter(index - self.istart, self.curchunk, self.curflags)

    def __len__(self):
        return self.dataset.size

    def _updatechunk(self, index):
        # nextchunkindex = self._chunkindex(index)
        # if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
        if self.curchunkindex is None or index < self.istart or index >= self.iend:
            nextchunkindex = self._chunkindex(index)
            istart, iend = _slice_for_chunk(nextchunkindex, self.dataset, self.chunksize_)
            nextchunk = self.dataset[istart:iend]
            if self.flags is not None:
                nextflags = self.flags[istart:iend]
            if self.curchunkdirty is True:
                self.dataset[self.istart:self.iend] = self.curchunk
                if self.flags is not None:
                    self.flags[self.istart:self.iend] = self.curflags
            self.curchunkdirty = False
            self.curchunk = nextchunk
            if self.flags is not None:
                self.curflags = nextflags
            self.curchunkindex = nextchunkindex
            self.istart = istart
            self.iend = iend

    def _chunkindex(self, index):
        return int(index / self.chunksize_)


class TimestampReader:

    def __init__(self, field):
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype = field.attrs['fieldtype']
        if fieldtype not in ('datetime', 'date'):
            error = "'fieldtype of '{} should be one of {}' but is {}"
            raise ValueError(error.format(field, "'datetime' or 'date'", fieldtype))

        self.dataset = field['timestamps']
        self.istart = None
        self.iend = None
        self.curchunk = None
        self.curchunkindex = None
        self.curchunkdirty = None
        self.chunksize_ = field.attrs['chunksize']

    def __getitem__(self, index):
        self._updatechunk(index)
        return self.curchunk[index - self.istart]

    def __len__(self):
        return self.dataset.size

    def _updatechunk(self, index):
        nextchunkindex = self._chunkindex(index)
        if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
            istart, iend = _slice_for_chunk(nextchunkindex, self.dataset, self.chunksize_)
            nextchunk = self.dataset[istart:iend]
            if self.curchunkdirty is True:
                self.dataset[self.istart:self.iend] = self.curchunk
            self.curchunkdirty = False
            self.curchunk = nextchunk
            self.curchunkindex = nextchunkindex
            self.istart = istart
            self.iend = iend

    def _chunkindex(self, index):
        return int(index / self.chunksize_)


@jit
def filtered_iterator(values, filter, default=np.nan):
    for i in range(len(values)):
        if filter[i]:
            yield default
        else:
            yield values[i]


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
            input_readers[k] = get_new_reader(v)
    output_writers = dict()
    output_arrays = dict()
    for k, v in outputs.items():
        if isinstance(v, NewWriter):
            output_writers[k] = v
        else:
            outputs[k] = get_new_writer(v)

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


def get_new_reader(dataset, field_name):
    fieldtype_map = {
        'indexedstring': IndexedStringReader,
        'fixedstring': FixedStringReader,
        'categorical': CategoricalReader,
        'numeric': NewNumericReader,
        'datetime': TimestampReader,
        'date': TimestampReader
    }
    if field_name not in dataset.keys():
        error = "{} is not a field in {}. The following fields are available: {}"
        raise ValueError(error.format(field_name, dataset, dataset.keys()))

    field = dataset[field_name]
    if 'fieldtype' not in field.attrs.keys():
        error = "'fieldtype' is not a field in {}."
        raise ValueError(error.format(field))

    return fieldtype_map[field.attrs['fieldmap'].split(',')[0]](field)


def get_new_writer(dataset, field_name):
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

    def getwriter(self, dest_group, dest_name, timestamp):
        return NewIndexedStringWriter(dest_group, self.chunksize, dest_name, timestamp)

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

    def getwriter(self, dest_group, dest_name, timestamp):
        return NewNumericWriter(dest_group, self.chunksize, dest_name, timestamp,
                                self.field.attrs['fieldtype'].split(',')[1])

    def dtype(self):
        return self.field['values'].dtype


class NewCategoricalReader(NewReader):
    def __init__(self, field, flagged_value=None):
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

    def getwriter(self, dest_group, dest_name, timestamp):
        return NewCategoricalWriter(dest_group, self.chunksize, dest_name, timestamp,
                                    {v: k for k, v in enumerate(self.field['keys'])})

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

    def getwriter(self, dest_group, dest_name, timestamp):
        return NewFixedStringWriter(dest_group, self.chunksize, dest_name, timestamp,
                                    self.field.attrs['fieldtype'].split(',')[1])

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

    def getwriter(self, dest_group, dest_name, timestamp):
        return NewTimestampWriter(dest_group, self.chunksize, dest_name, timestamp)

    def dtype(self):
        return self.field['values'].dtype


class NewWriter:
    def __init__(self, field):
        self.field = field

class NewIndexedStringWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp):
        NewWriter.__init__(self, group.create_group(name))
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

    def write(self, values):
        self.write_part(values)
        self.flush()

    def write_part_raw(self, index, values):
        DataWriter._write(self.field, 'index', index, len(index))
        DataWriter._write(self.field, 'values', values, len(values))

    def write_raw(self, index, values):
        self.write_part_raw(index, values)
        self.flush()


class NewCategoricalImporter():
    def __init__(self, group, chunksize, name, timestamp, categories):
        self.writer = NewCategoricalWriter(group, chunksize, name, timestamp, categories)
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
    def __init__(self, group, chunksize, name, timestamp, categories):
        NewWriter.__init__(self, group.create_group(name))
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
    def __init__(self, group, chunksize, name, timestamp, nformat, parser):
        self.data_writer = NewNumericWriter(group, chunksize, name, timestamp, nformat)
        self.flag_writer = NewNumericWriter(group, chunksize, f"{name}_valid", timestamp, 'bool')
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
    def __init__(self, group, chunksize, name, timestamp, nformat, needs_filter=False):
        NewWriter.__init__(self, group.create_group(name))
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
    def __init__(self, group, chunksize, name, timestamp, strlen):
        NewWriter.__init__(self, group.create_group(name))
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
    def __init__(self, group, chunksize, name, timestamp):
        self.datetime = NewDateTimeWriter(group, chunksize, name, timestamp)
        self.datetimeset = NewNumericWriter(group, chunksize, f"{name}_set", timestamp, 'bool')

    def chunk_factory(self, length):
        return self.datetime.chunk_factory(length)

    def write_part(self, values):
        # TODO: use a timestamp writer instead of a datetime writer and do the conversion here
        flags = self.datetimeset.chunk_factory(len(values))
        for i in range(len(values)):
            flags[i] = values[i] != b''
        self.datetime.write_part(values)
        self.datetimeset.write_part(flags)

    def flush(self):
        self.datetime.flush()
        self.datetimeset.flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


# TODO writers can write out more than one field; offset could be done this way
class NewDateTimeWriter(NewWriter):
    def __init__(self, group, chunksize, name, timestamp):
        NewWriter.__init__(self, group.create_group(name))
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
    def __init__(self, group, chunksize, name, timestamp):
        NewWriter.__init__(self, group.create_group(name))
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
    def __init__(self, group, chunksize, name, timestamp):
        NewWriter.__init__(self, group.create_group(name))
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
    def __init__(self, group, chunksize, name, timestamp):
        self.date = NewDateWriter(group, chunksize, name, timestamp)
        self.dateset = NewNumericWriter(group, chunksize, f"{name}_set", timestamp, 'bool')

    def chunk_factory(self, length):
        return self.date.chunk_factory(length)

    def write_part(self, values):
        # TODO: use a timestamp writer instead of a datetime writer and do the conversion here
        flags = self.dateset.chunk_factory(len(values))
        for i in range(len(values)):
            flags[i] = values[i] != b''
        self.date.write_part(values)
        self.dateset.write_part(flags)

    def flush(self):
        self.date.flush()
        self.dateset.flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


# Sorters
# =======

# def fixed_string_sorter(index, reader, writer):
