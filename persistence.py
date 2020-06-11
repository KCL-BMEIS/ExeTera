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

from datetime import datetime
from io import BytesIO

import h5py
import numpy as np

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

def save_to_hdf5(doc, space, fields, categories={}, filters={}, formats={}):
    for fk, fv in fields.items():
        gspace = doc[space]
        if isinstance(fv, list):
            if fk not in formats:
                save_variable_strs_to_hdf5(gspace, fk, fv)
            elif formats[fk] == 'datetime':
                save_datetimes_to_hdf5(gspace, fk, fv)

        else:
            save_nparray_to_hdf5(gspace, fk, fv)

        # if fk in categories:
        #     cats = categories[fk].values_to_strings
        #     values = gspace.create_dataset('category_names', (len(cats),), dtype=h5py.string_dtype())
        #     values[:] = cats
        #
        # if fk in filters:
        #     fltr = filters[fk]
        #     gspace.create_dataset('filter', (fltr.size,),
        #                           chunks=chunk_sizes[space], maxshape=(None,), data=fltr)


def save_nparray_to_hdf5(parent, name, values, chunksize=None):
    gspace = parent.create_group(name)
    if chunksize is None:
        gspace.create_dataset('values', (values.size,), maxshape=(None,), data=values)
    else:
        gspace.create_dataset('values', (values.size,),
                              chunks=(chunksize,), maxshape=(None,), data=values)


def save_variable_strs_to_hdf5(parent, name, values):
    bytestream = BytesIO()
    count = len(values) + 1
    indices = np.zeros(count, dtype=np.uint32)
    index = 0
    indices[0] = index
    for i in range(len(values)):
        bseq = values[i].encode()
        bytestream.write(bseq)
        index += len(bseq)
        indices[i+1] = index
    bytevalues = np.frombuffer(bytestream.getvalue(), dtype="S1")
    group = parent.create_group(name) if name not in parent.keys() else parent[name]
    group.create_dataset('values', (bytevalues.size,), data=bytevalues)
    group.create_dataset('index', (count,), data=indices)


def save_datetimes_to_hdf5(parent, name, values):
    group = parent.create_group(name)
    count = len(values)
    years = np.zeros(count, dtype='u2')
    months = np.zeros(count, dtype='u1')
    days = np.zeros(count, dtype='u1')
    hours = np.zeros(count, dtype='u1')
    minutes = np.zeros(count, dtype='u1')
    seconds = np.zeros(count, dtype='u1')
    fractions = np.zeros(count, dtype='u4')
    for i, sts in enumerate(values):
        try:
            ts = datetime.strptime(sts, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            ts = datetime.strptime(sts, '%Y-%m-%d %H:%M:%S')
        years[i] = ts.year
        months[i] = ts.month
        days[i] = ts.day
        hours[i] = ts.hour
        minutes[i] = ts.minute
        seconds[i] = ts.second
        fractions[i] = ts.microsecond
    group['years'] = years
    group['months'] = months
    group['days'] = days
    group['hours'] = hours
    group['minutes'] = minutes
    group['seconds'] = seconds
    group['fractions'] = fractions


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


def save_variable_strs_to_hdf5(parent, name, values):
    bytestream = BytesIO()
    count = len(values) + 1
    indices = np.zeros(count, dtype=np.uint32)
    index = 0
    indices[0] = index
    for i in range(len(values)):
        bseq = values[i].encode()
        bytestream.write(bseq)
        index += len(bseq)
        indices[i+1] = index
    bytevalues = np.frombuffer(bytestream.getvalue(), dtype="S1")
    group = parent.create_group(name) if name not in parent.keys() else parent[name]
    group.create_dataset('values', (bytevalues.size,), data=bytevalues)
    group.create_dataset('index', (count,), data=indices)


class BaseWriter:
    def __init__(self, group, name, chunksize, fieldtype, timestamp):
        self.group = group.create_group(name)
        self.chunksize = chunksize
        self.fieldtype = fieldtype
        self.timestamp = timestamp

    def flush(self):
        self.group.attrs['fieldtype'] = self.fieldtype
        self.group.attrs['timestamp'] = self.timestamp
        self.group.attrs['chunksize'] = self.chunksize
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
            if self.value_index == self.chunksize:
                DataWriter._write(self.group, 'values', self.values, self.value_index)
                self.value_index = 0
            self.accumulated += 1
        self.indices[self.index_index] = self.accumulated
        self.index_index += 1
        if self.index_index == self.chunksize:
            DataWriter._write(self.group, 'index', self.indices, self.index_index)
            self.index_index = 0

    def flush(self):
        print('flush: value_index =', self.value_index)
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

        if self.index == self.chunksize:
            DataWriter._write(self.group, 'values', self.values, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'values', self.values, self.index)
            self.index = 0
        key_index = [None] * len(self.keys)
        for k, v in self.keys.items():
            key_index[v] = k
        DataWriter._write(self.group, 'keys', key_index, len(self.keys),
                          dtype=h5py.string_dtype())
        BaseWriter.flush(self)


class NumericWriter(BaseWriter):
    def __init__(self, group, chunksize, name, timestamp, nformat):
        BaseWriter.__init__(self, group, name, chunksize, f"numeric,{nformat}", timestamp)
        self.name = name
        self.values = np.zeros(chunksize, dtype=nformat)
        self.filter = np.zeros(chunksize, dtype=np.bool)
        self.index = 0

    def append(self, value):
        try:
            v = float(value)
            f = False
        except:
            v = 0
            f = True
        self.values[self.index] = v
        self.filter[self.index] = f
        self.index += 1
        if self.index == self.chunksize:
            DataWriter._write(self.group, 'values', self.values, self.index)
            DataWriter._write(self.group, 'filter', self.filter, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'values', self.values, self.index)
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
        if self.index == self.chunksize:
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
        self.years = np.zeros(chunksize, dtype='u2')
        self.months = np.zeros(chunksize, dtype='u1')
        self.days = np.zeros(chunksize, dtype='u1')
        self.index = 0

    def append(self, value):
        if value == '':
            self.years[self.index] = 0
            self.months[self.index] = 0
            self.days[self.index] = 0
        else:
            ts = datetime.strptime(value, '%Y-%m-%d')
            self.years[self.index] = ts.year
            self.months[self.index] = ts.month
            self.days[self.index] = ts.day
        self.index += 1

        if self.index == self.chunksize:
            DataWriter._write(self.group, 'years', self.years, self.index)
            DataWriter._write(self.group, 'months', self.months, self.index)
            DataWriter._write(self.group, 'days', self.days, self.index)
            self.index = 0

    def flush(self):
        if self.index != 0:
            DataWriter._write(self.group, 'years', self.years, self.index)
            DataWriter._write(self.group, 'months', self.months, self.index)
            DataWriter._write(self.group, 'days', self.days, self.index)
            self.index = 0
        BaseWriter.flush(self)


class DatetimeWriter(BaseWriter):
    def __init__(self, group, chunksize, name, timestamp):
        BaseWriter.__init__(self, group, name, chunksize, "datetime", timestamp)
        self.name = name
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

        if self.index == self.chunksize:
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
def _chunkmax(dataset, chunksize):
    chunkmax = int(dataset.size / chunksize)
    if dataset.size % chunksize != 0:
        chunkmax += 1
    return chunkmax

def _slice_for_chunk(c, chunkmax, dataset, chunksize):
    if c == chunkmax - 1:
        length = dataset.size % chunksize
    else:
        length = chunksize
    return c * chunksize, c * chunksize + length


def indexed_string_iterator(group, name):
    field = group[name]
    if field.attrs['fieldtype'].split(',')[0] != 'indexedstring':
        raise ValueError(
            f"{field} must be 'indexedstring' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']

    index = field['index']
    ichunkmax = _chunkmax(index, chunksize)

    values = field['values']
    vchunkmax = _chunkmax(values, chunksize)

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


def categorical_iterator(group, name):
    field = group[name]
    if field.attrs['fieldtype'] != 'categorical':
        raise ValueError(
            f"{field} must be 'categorical' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']

    values = field['values']
    chunkmax = _chunkmax(values, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, values, chunksize)
        vcur = values[istart:iend]
        for i in range(len(vcur)):
            yield vcur[i]


def numeric_iterator(group, name, invalid=None):
    field = group[name]
    if field.attrs['fieldtype'].split(',')[0] != 'numeric':
        raise ValueError(
            f"{field} must be 'numeric' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']

    values = field['values']
    filter = field['filter']
    chunkmax = _chunkmax(values, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, values, chunksize)
        vcur = values[istart:iend]
        vflt = filter[istart:iend]
        for i in range(len(vcur)):
            if vflt[i] == False:
                yield vcur[i]
            else:
                yield invalid


def years_iterator(field):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    chunksize = field.attrs['chunksize']
    years = field['years']
    chunkmax = _chunkmax(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, years, chunksize)
        ycur = years[istart:iend]
        for i in range(len(ycur)):
            yield ycur[i]


def months_iterator(field):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    chunkmax = _chunkmax(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, years, chunksize)
        ycur = years[istart:iend]
        mcur = months[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mcur[i])


def days_iterator(field):
    valid_types = ('date', 'datetime')
    if field.attrs['fieldtype'] not in valid_types:
        raise ValueError(
            f"{field} must be one of {value_types} but is {field.attrs['fieldtype']}")

    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    days = field['days']
    chunkmax = _chunkmax(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, years, chunksize)
        ycur = years[istart:iend]
        mcur = months[istart:iend]
        dcur = days[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mcur[i], dcur[i])


def hours_iterator(field):
    if field.attrs['fieldtype'] != 'datetime':
        raise ValueError(
            f"{field} must be 'datetime' but is {field.attrs['fieldtype']}")
    chunksize = field.attrs['chunksize']
    years = field['years']
    months = field['months']
    days = field['days']
    hours = field['hours']
    chunkmax = _chunkmax(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, years, chunksize)
        ycur = years[istart:iend]
        mcur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mcur[i], dcur[i], hcur[i])


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
    chunkmax = _chunkmax(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, years, chunksize)
        ycur = years[istart:iend]
        mocur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        micur = minutes[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mocur[i], dcur[i], hcur[i], micur[i])


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
    chunkmax = _chunkmax(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, years, chunksize)
        ycur = years[istart:iend]
        mocur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        micur = minutes[istart:iend]
        scur = seconds[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mocur[i], dcur[i], hcur[i], micur[i], scur[i])


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
    chunkmax = _chunkmax(years, chunksize)
    for c in range(chunkmax):
        istart, iend = _slice_for_chunk(c, chunkmax, years, chunksize)
        ycur = years[istart:iend]
        mocur = months[istart:iend]
        dcur = days[istart:iend]
        hcur = hours[istart:iend]
        micur = minutes[istart:iend]
        scur = seconds[istart:iend]
        mscur = microseconds[istart:iend]
        for i in range(len(ycur)):
            yield (ycur[i], mocur[i], dcur[i], hcur[i], micur[i], scur[i], mscur[i])


