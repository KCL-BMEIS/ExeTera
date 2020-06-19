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
from io import BytesIO

import h5py
import numpy as np

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
            yield vcur[i].tostring().decode()


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
        for i in range(len(vcur)):
            yield vcur[i]


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


def dataset_sort(index, fields):
    rfields = reversed(fields)

    for f in rfields:
        fdata = reader(f)
        index = sorted(index, key=lambda x: fdata[x])
    return index


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


def get_reader(dataset, field_name):
    fieldtype_map = {
        'indexedstring': IndexedStringReader,
        'fixedstring': FixedStringReader,
        'categorical': CategoricalReader,
        'boolean': NumericReader,
        'numeric': NumericReader,
        'datetime': TimestampReader,
        'date': TimestampReader
    }
    if field_name not in dataset.keys():
        error = "{} is not a field in {}. The following fields are available: {}"
        raise ValueError(error.format(field_name, dataset, dataset.keys()))

    group = dataset[field_name]
    if 'fieldtype' not in group.attrs.keys():
        error = "'fieldtype' is not a field in {}."
        raise ValueError(error.format(group))

    fieldtype_elems = group.attrs['fieldtype'].split(',')


class FixedStringReader:
    def __init__(self, field, as_string=False):
        if 'fieldtype' not in field.attrs.keys():
            error = "{} must have 'fieldtype' in its attrs property"
            raise ValueError(error.format(field))
        fieldtype, fieldlen = field.attrs['fieldtype'].split(',')
        if fieldtype != 'fixedstring':
            error = "'fieldtype of '{} should be 'categorical' but is {}"
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
        nextchunkindex = self._chunkindex(index)
        if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
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


# class Series:
#
#     def __init__(self, dataset, chunksize=DEFAULT_CHUNKSIZE):
#         self.dataset = dataset
#         self.istart = None
#         self.iend = None
#         self.curchunk = None
#         self.curchunkindex = None
#         self.curchunkdirty = None
#         self.chunksize_ = chunksize
#
#     def __getitem__(self, index):
#         self._updatechunk(index)
#         return self.curchunk[index - self.istart]
#
#     def __setitem__(self, index, value):
#         self._updatechunk(index)
#         self.curchunk[index - self.istart] = value
#
#     def __len__(self):
#         return self.dataset.size
#
#     def chunksize(self):
#         return self.chunksize_
#
#     def _updatechunk(self, index):
#         nextchunkindex = self._chunkindex(index)
#         if self.curchunkindex is None or self.curchunkindex != nextchunkindex:
#             istart, iend = _slice_for_chunk(nextchunkindex, self.dataset, self.chunksize_)
#             nextchunk = self.dataset[istart:iend]
#             if self.curchunkdirty is True:
#                 self.dataset[self.istart:self.iend] = self.curchunk
#             self.curchunkdirty = False
#             self.curchunk = nextchunk
#             self.curchunkindex = nextchunkindex
#             self.istart = istart
#             self.iend = iend
#
#     def _chunkindex(self, iarg):
#         return int(iarg / self.chunksize())
