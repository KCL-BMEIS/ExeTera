from datetime import datetime, timezone

import numpy as np
import numba
import h5py

from hystore.core.readerwriter import DataWriter
from hystore.core import utils


# def test_field_iterator(data):
#     @numba.njit
#     def _inner():
#         for d in data:
#             yield d
#     return _inner()
#
# iterator_type = numba.from_dtype(test_field_iterator)
#
# @numba.jit
# def sum_iterator(iter_):
#     total = np.int64(0)
#     for i in iter_:
#         total += i
#     return total


class Field:
    def __init__(self, session, group, name=None, write_enabled=False):
        # if name is None, the group is an existing field
        # if name is set but group[name] doesn't exist, then create the field
        if name is None:
            # the group is an existing field
            field = group
        else:
            field = group[name]
        self._session = session
        self._field = field
        self._fieldtype = self._field.attrs['fieldtype']
        self._write_enabled = write_enabled
        self._value_wrapper = None

    @property
    def name(self):
        return self._field.name

    @property
    def timestamp(self):
        return self._field.attrs['timestamp']

    @property
    def chunksize(self):
        return self._field.attrs['chunksize']


class ReadOnlyFieldArray:
    def __init__(self, field, dataset_name):
        self._field = field
        self._name = dataset_name
        self._dataset = field[dataset_name]

    def __len__(self):
        return len(self._dataset)

    @property
    def dtype(self):
        return self._dataset.dtype

    def __getitem__(self, item):
        return self._dataset[item]

    def __setitem__(self, key, value):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def clear(self):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write_part(self, part):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write(self, part):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def complete(self):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")


class WriteableFieldArray:
    def __init__(self, field, dataset_name):
        self._field = field
        self._name = dataset_name
        self._dataset = field[dataset_name]

    def __len__(self):
        return len(self._dataset)

    @property
    def dtype(self):
        return self._dataset.dtype

    def __getitem__(self, item):
        return self._dataset[item]

    def __setitem__(self, key, value):
        self._dataset[key] = value

    def clear(self):
        DataWriter._clear_dataset(self._field, self._name)

    def write_part(self, part):
        DataWriter.write(self._field, self._name, part, len(part), dtype=self._dataset.dtype)

    def write(self, part):
        DataWriter.write(self._field, self._name, part, len(part), dtype=self._dataset.dtype)
        self.complete()

    def complete(self):
        DataWriter.flush(self._field[self._name])


class ReadOnlyIndexedFieldArray:
    def __init__(self, field, index_name, values_name):
        self._field = field
        self._index_name = index_name
        self._index_dataset = field[index_name]
        self._values_name = values_name
        self._values_dataset = field[values_name]

    def __len__(self):
        return len(self._index_dataset)-1

    def __getitem__(self, item):
        try:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop if item.stop is not None else len(self._index_dataset) - 1
                step = item.step
                #TODO: validate slice
                index = self._index_dataset[start:stop+1]
                bytestr = self._values_dataset[index[0]:index[-1]]
                results = [None] * (len(index)-1)
                startindex = start
                for ir in range(len(results)):
                    results[ir] =\
                        bytestr[index[ir]-np.int64(startindex):
                                index[ir+1]-np.int64(startindex)].tobytes().decode()
                return results
        except Exception as e:
            print("{}: unexpected exception {}".format(self._field.name, e))
            raise

    def __setitem__(self, key, value):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def clear(self):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write_part(self, part):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write(self, part):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def complete(self):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")


class WriteableIndexedFieldArray:
    def __init__(self, field, index_name, values_name):
        self._field = field
        self._index_name = index_name
        self._index_dataset = field[index_name]
        self._values_name = values_name
        self._values_dataset = field[values_name]
        self._chunksize = self._field.attrs['chunksize']
        self._raw_values = np.zeros(self._chunksize, dtype=np.uint8)
        self._raw_indices = np.zeros(self._chunksize, dtype=np.int64)
        self._accumulated = self._index_dataset[-1] if len(self._index_dataset) else 0
        self._index_index = 0
        self._value_index = 0

    def __len__(self):
        return len(self._index_dataset) - 1

    def __getitem__(self, item):
        try:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop if item.stop is not None else len(self._index_dataset) - 1
                step = item.step
                # TODO: validate slice
                index = self._index_dataset[start:stop + 1]
                bytestr = self._values_dataset[index[0]:index[-1]]
                results = [None] * (len(index) - 1)
                startindex = start
                rmax = min(len(results), stop - start)
                for ir in range(rmax):
                    rbytes = bytestr[index[ir] - np.int64(startindex):
                                index[ir + 1] - np.int64(startindex)].tobytes()
                    rstr = rbytes.decode()
                    results[ir] = rstr
                return results
        except Exception as e:
            print("{}: unexpected exception {}".format(self._field.name, e))
            raise

    def __setitem__(self, key, value):
        raise PermissionError("IndexedStringField instances cannot be edited via array syntax;"
                              "use clear and then write/write_part or write_raw/write_part_raw")

    def clear(self):
        self._accumulated = 0
        DataWriter.clear_dataset(self._field, self._index_name)
        DataWriter.clear_dataset(self._field, self._values_name)
        DataWriter.write(self._field, self._index_name, [], 0, 'int64')
        DataWriter.write(self._field, self._values_name, [], 0, 'uint8')
        self._index_dataset = self._field[self._index_name]
        self._values_dataset = self._field[self._values_name]
        self._accumulated = 0


    def write_part(self, part):
        for s in part:
            evalue = s.encode()
            for v in evalue:
                self._raw_values[self._value_index] = v
                self._value_index += 1
                if self._value_index == self._chunksize:
                    DataWriter.write(self._field, self._values_name,
                                     self._raw_values, self._value_index)
                    self._value_index = 0
                self._accumulated += 1
            self._raw_indices[self._index_index] = self._accumulated
            self._index_index += 1
            if self._index_index == self._chunksize:
                if len(self._field['index']) == 0:
                    DataWriter.write(self._field, self._index_name, [0], 1)
                DataWriter.write(self._field, self._index_name,
                                 self._raw_indices, self._index_index)
                self._index_index = 0


    def write(self, part):
        self.write_part(part)
        self.complete()

    def complete(self):
        if self._value_index != 0:
            DataWriter.write(self._field, self._values_name,
                             self._raw_values, self._value_index)
            self._value_index = 0
        if self._index_index != 0:
            if len(self._field['index']) == 0:
                DataWriter.write(self._field, self._index_name, [0], 1)
            DataWriter.write(self._field, self._index_name,
                             self._raw_indices, self._index_index)
            self._index_index = 0


def base_field_contructor(session, group, name, timestamp=None, chunksize=None):
    if name in group:
        msg = "Field '{}' already exists in group '{}'"
        raise ValueError(msg.format(name, group))

    field = group.create_group(name)
    field.attrs['chunksize'] = session.chunksize if chunksize is None else chunksize
    field.attrs['timestamp'] = session.chunksize if chunksize is None else chunksize
    return field


def indexed_string_field_constructor(session, group, name, timestamp=None, chunksize=None):
    field = base_field_contructor(session, group, name, timestamp, chunksize)
    field.attrs['fieldtype'] = 'indexedstring'
    DataWriter.write(field, 'index', [], 0, 'int64')
    DataWriter.write(field, 'values', [], 0, 'uint8')


def fixed_string_field_constructor(session, group, name, length, timestamp=None, chunksize=None):
    field = base_field_contructor(session, group, name, timestamp, chunksize)
    field.attrs['fieldtype'] = 'fixedstring,{}'.format(length)
    field.attrs['strlen'] = length
    DataWriter.write(field, 'values', [], 0, "S{}".format(length))


def numeric_field_constructor(session, group, name, nformat, timestamp=None, chunksize=None):
    field = base_field_contructor(session, group, name, timestamp, chunksize)
    field.attrs['fieldtype'] = 'numeric,{}'.format(nformat)
    field.attrs['nformat'] = nformat
    DataWriter.write(field, 'values', [], 0, nformat)


def categorical_field_constructor(session, group, name, nformat, key,
                                  timestamp=None, chunksize=None):
    field = base_field_contructor(session, group, name, timestamp, chunksize)
    field.attrs['fieldtype'] = 'categorical,{}'.format(nformat)
    field.attrs['nformat'] = nformat
    DataWriter.write(field, 'values', [], 0, nformat)
    key_values = [v for k, v in key.items()]
    key_names = [k for k, v in key.items()]
    DataWriter.write(field, 'key_values', key_values, len(key_values), 'int8')
    DataWriter.write(field, 'key_names', key_names, len(key_names), h5py.special_dtype(vlen=str))


def timestamp_field_constructor(session, group, name, timestamp=None, chunksize=None):
    field = base_field_contructor(session, group, name, timestamp, chunksize)
    field.attrs['fieldtype'] = 'timestamp'
    DataWriter.write(field, 'values', [], 0, 'float64')


class IndexedStringField(Field):
    def __init__(self, session, group, name=None, write_enabled=False):
        super().__init__(session, group, name=name, write_enabled=write_enabled)
        self._session = session
        self._data_wrapper = None
        self._index_wrapper = None
        self._value_wrapper = None

    def writeable(self):
        return IndexedStringField(self._session, self._field, write_enabled=True)

    def create_like(self, group, name, timestamp=None):
        ts = self.timestamp if timestamp is None else timestamp
        indexed_string_field_constructor(self._session, group, name, ts, self.chunksize)
        return IndexedStringField(self._session, group, name, write_enabled=True)

    @property
    def data(self):
        if self._data_wrapper is None:
            wrapper =\
                WriteableIndexedFieldArray if self._write_enabled else ReadOnlyIndexedFieldArray
            self._data_wrapper = wrapper(self._field, 'index', 'values')
        return self._data_wrapper

    @property
    def indices(self):
        if self._index_wrapper is None:
            wrapper = WriteableFieldArray if self._write_enabled else ReadOnlyFieldArray
            self._index_wrapper = wrapper(self._field, 'index')
        return self._index_wrapper

    @property
    def values(self):
        if self._value_wrapper is None:
            wrapper = WriteableFieldArray if self._write_enabled else ReadOnlyFieldArray
            self._value_wrapper = wrapper(self._field, 'values')
        return self._value_wrapper


class FixedStringField(Field):
    def __init__(self, session, group, name=None, write_enabled=False):
        super().__init__(session, group, name=name, write_enabled=write_enabled)

    def writeable(self):
        return FixedStringField(self._session, self._field, write_enabled=True)

    def create_like(self, group, name, timestamp=None):
        ts = self.timestamp if timestamp is None else timestamp
        length = self._field.attrs['strlen']
        fixed_string_field_constructor(self._session, group, name, length, ts, self.chunksize)
        return FixedStringField(self._session, group, name, write_enabled=True)

    @property
    def data(self):
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper


class NumericField(Field):
    def __init__(self, session, group, name=None, write_enabled=False):
        super().__init__(session, group, name=name, write_enabled=write_enabled)

    def writeable(self):
        return NumericField(self._session, self._field, write_enabled=True)

    def create_like(self, group, name, timestamp=None):
        ts = self.timestamp if timestamp is None else timestamp
        nformat = self._field.attrs['nformat']
        numeric_field_constructor(self._session, group, name, nformat, ts, self.chunksize)
        return NumericField(self._session, group, name, write_enabled=True)

    @property
    def data(self):
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper


class CategoricalField(Field):
    def __init__(self, session, group,
                 name=None, write_enabled=False):
        super().__init__(session, group, name=name, write_enabled=write_enabled)

    def writeable(self):
        return CategoricalField(self._session, self._field, write_enabled=True)

    def create_like(self, group, name, timestamp=None):
        ts = self.timestamp if timestamp is None else timestamp
        nformat = self._field.attrs['nformat'] if 'nformat' in self._field.attrs else 'int8'
        keys = {v: k for k, v in self.keys.items()}
        categorical_field_constructor(self._session, group, name, nformat, keys,
                                      ts, self.chunksize)
        return CategoricalField(self._session, group, name, write_enabled=True)

    @property
    def data(self):
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper


    # Note: key is presented as value: str, even though the dictionary must be presented
    # as str: value
    @property
    def keys(self):
        kv = self._field['key_values']
        kn = self._field['key_names']
        keys = dict(zip(kv, kn))
        return keys


class TimestampField(Field):
    def __init__(self, session, group, name=None, write_enabled=False):
        super().__init__(session, group, name=name, write_enabled=write_enabled)

    def writeable(self):
        return TimestampField(self._session, self._field, write_enabled=True)

    def create_like(self, group, name, timestamp=None):
        ts = self.timestamp if timestamp is None else timestamp
        timestamp_field_constructor(self._session, group, name, ts, self.chunksize)
        return TimestampField(self._session, group, name, write_enabled=True)

    @property
    def data(self):
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper


class IndexedStringImporter:
    def __init__(self, session, group, name, timestamp=None, chunksize=None):
        indexed_string_field_constructor(session, group, name, timestamp, chunksize)
        self._field = IndexedStringField(session, group, name, write_enabled=True)

    def chunk_factory(self, length):
        return [None] * length

    def write_part(self, values):
        with utils.Timer("writing {}".format(self._field.name)):
            self._field.data.write_part(values)

    def complete(self):
        self._field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()


class FixedStringImporter:
    def __init__(self, session, group, name, length, timestamp=None, chunksize=None):
        fixed_string_field_constructor(session, group, name, length, timestamp, chunksize)
        self._field = FixedStringField(session, group, name, write_enabled=True)

    def chunk_factory(self, length):
        return np.zeros(length, dtype=self._field.data.dtype)

    def write_part(self, values):
        with utils.Timer("writing {}".format(self._field.name)):
            # self._field.data.write_part([v.encode() for v in values])
            self._field.data.write_part(values)

    def complete(self):
        self._field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()


class NumericImporter:
    def __init__(self, session, group, name, dtype, parser, timestamp=None, chunksize=None):
        numeric_field_constructor(session, group, name, dtype, timestamp, chunksize)
        numeric_field_constructor(session, group, '{}_valid'.format(name), 'bool',
                                  timestamp, chunksize)

        chunksize = session.chunksize if chunksize is None else chunksize
        self._field = NumericField(session, group, name, write_enabled=True)
        self._filter_field = NumericField(session, group, name, write_enabled=True)

        self._parser = parser
        self._values = np.zeros(chunksize, dtype=self._field.data.dtype)

        self._filter_values = np.zeros(chunksize, dtype='bool')

    def chunk_factory(self, length):
        # return np.zeros(length, dtype=self._field.data.dtype)
        return [None] * length

    def write_part(self, values):
        with utils.Timer("writing {}".format(self._field.name)):
            for i in range(len(values)):
                valid, value = self._parser(values[i])
                self._values[i] = value
                self._filter_values[i] = valid
            self._field.data.write_part(self._values)
            self._filter_field.data.write_part(self._filter_values)

    def complete(self):
        self._field.data.complete()
        self._filter_field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()


class CategoricalImporter:
    def __init__(self, session, group, name, value_type, keys, timestamp=None, chunksize=None):
        chunksize = session.chunksize if chunksize is None else chunksize
        categorical_field_constructor(session, group, name, value_type, keys, timestamp, chunksize)
        self._field = CategoricalField(session, group, name, write_enabled=True)
        self._keys = keys
        self._dtype = value_type
        self._key_type = 'U{}'.format(max(len(k.encode()) for k in keys))
        # self._results = np.zeros(chunksize, dtype=value_type)

    def chunk_factory(self, length):
        # return np.zeros(length, dtype=self._key_type)
        return np.zeros(length, dtype=self._dtype)

    def write_part(self, values):
        with utils.Timer("writing {}".format(self._field.name)):
            keys = self._keys
            # results = self._results
            # for i in range(len(values)):
            #     results = keys[values[i]]
            # self._field.data.write_part(results)
            self._field.data.write_part(values)

    def complete(self):
        self._field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()


class LeakyCategoricalImporter:
    def __init__(self, session, group, name, value_type, keys, out_of_range,
                 timestamp=None, chunksize=None):
        chunksize = session.chunksize if chunksize is None else chunksize
        categorical_field_constructor(session, group, name, value_type, keys,
                                      timestamp, chunksize)
        out_of_range_name = '{}_{}'.format(name, out_of_range)
        indexed_string_field_constructor(session, group, out_of_range_name,
                                         timestamp, chunksize)

        self._field = CategoricalField(session, group, name, write_enabled=True)
        self._str_field = IndexedStringField(session, group, out_of_range_name, write_enabled=True)

        self._keys = keys
        self._dtype = value_type
        self._key_type = 'S{}'.format(max(len(k.encode()) for k in keys))

        self._results = np.zeros(chunksize, dtype=value_type)
        self._strresult = [None] * chunksize

    def chunk_factory(self, length):
        return [None] * length

    def write_part(self, values):
        with utils.Timer("writing {}".format(self._field.name)):
            keys = self._keys
            results = self._results
            strresults = self._strresult
            for i in range(len(values)):
                value = keys.get(values[i], -1)
                if value == -1:
                    strresults[i] = values[i]
                else:
                    strresults[i] = ''
                results[i] = value
                # results = keys[values[i]]
            if len(values) != len(results):
                self._field.data.write_part(results[:len(values)])
                self._str_field.data.write_part(strresults[:len(values)])
            else:
                self._field.data.write_part(results)
                self._str_field.data.write_part(strresults)

    def complete(self):
        self._field.data.complete()
        self._str_field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()


class DateTimeImporter:
    def __init__(self, session, group, name,
                 optional=False, write_days=False, timestamp=None, chunksize=None):
        chunksize = session.chunksize if chunksize is None else chunksize
        timestamp_field_constructor(session, group, name, timestamp, chunksize)
        self._field = TimestampField(session, group, name, timestamp, write_enabled=True)
        self._results = np.zeros(chunksize , dtype='float64')
        self._optional = optional

        if optional is True:
            filter_name = '{}_set'.format(name)
            numeric_field_constructor(session, group, filter_name, 'bool',
                                      timestamp, chunksize)
            self._filter_field = NumericField(session, group, filter_name, write_enabled=True)

    def chunk_factory(self, length):
        return np.zeros(length, dtype='U32')

    def write_part(self, values):
        with utils.Timer("writing {}".format(self._field.name)):
            results = self._results

            for i, v in enumerate(values):
                if len(v) == 32:
                    ts = datetime.strptime(v, '%Y-%m-%d %H:%M:%S.%f%z')
                    results[i] = ts.timestamp()
                elif len(v) == 25:
                    ts = datetime.strptime(v, '%Y-%m-%d %H:%M:%S%z')
                    results[i] = ts.timestamp()
                else:
                    if self._optional is True and len(v) == 0:
                        results[i] = np.nan
                    else:
                        msg = "Date field '{}' has unexpected format '{}'"
                        raise ValueError(msg.format(self._field, v))

            self._field.data.write_part(results)

    def complete(self):
        self._field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()


class DateImporter:
    def __init__(self, session, group, name,
                 optional=False, timestamp=None, chunksize=None):
        timestamp_field_constructor(session, group, name, timestamp, chunksize)
        self._field = TimestampField(session, group, name, timestamp, write_enabled=True)
        self._results = np.zeros(chunksize, dtype='float64')

        if optional is True:
            filter_name = '{}_set'.format(name)
            numeric_field_constructor(session, group, filter_name, 'bool',
                                      timestamp, chunksize)
            self._filter_field = NumericField(session, group, filter_name, write_enabled=True)

    def chunk_factory(self, length):
        return np.zeros(length, dtype='U10')

    def write_part(self, values):
        with utils.Timer("writing {}".format(self._field.name)):
            timestamps = np.zeros(len(values), dtype=np.float64)
            for i in range(len(values)):
                value = values[i]
                if value == '':
                    timestamps[i] = np.nan
                else:
                    ts = datetime.strptime(value, '%Y-%m-%d')
                    timestamps[i] = ts.timestamp()
            self._field.data.write_part(timestamps)

    def complete(self):
        self._field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()
