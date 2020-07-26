import numpy as np
import numba
import h5py

from hystore.core.persistence import DataWriter

"""
Design thoughts & snippets for fields:

Is there a difference between writing and importing?
. importing: scalably write data of an unknown length by successive appends
. writing: write the results of a transform on data of a known length
# reader/writer properties
f = ds.get('x')
v = f.r[:]
f.w[:] = v * 2

f = ds.get('x')
v = f[:]
f[:] = v * 2
# or
f.write(v * 2)

f = ds.get('x')

"""


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
#
# class TestField:
#
#     def __init__(self, data):
#         self.data = data
#
#     def __getitem__(self, item):
#         print("__getitem__({})".format(item))
#         return self.data[item]
#
#     def __setitem__(self, key, value):
#         print("__setitem__({}, {}".format(key, value))
#         self.data[key] = value
#
#     def __iter__(self):
#         return test_field_iterator(self.data)


class Field:
    def __init__(self, session, group,
                 name=None, additional_attributes=None, auxilliary_fields=None,
                 timestamp=None, write_enabled=False):
        # if name is None, the group is an existing field
        # if name is set but group[name] doesn't exist, then create the field
        if name is None:
            # the group is an existing field
            self._session = session
            self._field = group
            self._fieldtype = self._field.attrs['fieldtype']
            self._write_enabled = write_enabled
        else:
            if name is not None:
                if name in group.keys():
                    self._field = group['name']
                    self._fieldtype = self._field.attrs['fieldtype']
                    self._write_enabled = write_enabled
                else:
                    all_attributes = {
                        'timestamp': session.timestamp if timestamp is None else timestamp,
                        'chunksize': session.chunksize,
                    }
                    all_attributes.update(additional_attributes)
                    DataWriter.create_group(group, name, all_attributes.items())

                    if auxilliary_fields is not None:
                        for k, v in auxilliary_fields.items():
                            DataWriter.write(group, k, v[0], len(v), dtype=v[1])

                    self._field = group[name]
                    self._fieldtype = self._field.attrs['fieldtype']
                    self._write_enabled = True
        self._value_wrapper = None


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
        self._dataset[item]

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
        #self._dtype = dtype

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
        DataWriter.write_part(self._field, self._name, part, len(part), dtype=self._dataset.dtype)

    def write(self, part):
        DataWriter.write(self._field, self._name, part, len(part), dtype=self._dataset.dtype)

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
                stop = item.stop if item.stop is not None else len(self._field_index) - 1
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
            print("{}: unexpected exception {}".format(self.field.name, e))
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
        self._accumulated = self._index_dataset[-1]
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
        DataWriter.write(self._field, self._index_name, [0], 1, 'int64')
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
    DataWriter.write(field, 'index', [0], 1, 'int64')
    DataWriter.write(field, 'values', [], 0, 'uint8')


def fixed_string_field_constructor(session, group, name, length, timestamp=None, chunksize=None):
    field = base_field_contructor(session, group, name, timestamp, chunksize)
    field.attrs['fieldtype'] = 'fixedstring,{}'.format(length)
    field.attrs['length'] = length
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


class IndexedStringField(Field):
    def __init__(self, session, group,
                 name=None, timestamp=None, write_enabled='write'):
        additional = None
        if name is not None and name not in group:
            # new field
            additional = {'fieldtype': 'indexedstring'}
        super().__init__(session, group, name=name, additional_attributes=additional,
                         timestamp=timestamp, write_enabled=write_enabled)
        self._data_wrapper = None
        self._index_wrapper = None
        self._value_wrapper = None

    def writeable(self):
        return IndexedStringField(self._session, self._field, write_enabled=True)

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

    # def dtype(self):
    #     return self


class FixedStringField(Field):
    def __init__(self, session, group,
                 name=None, length=None, timestamp=None, write_enabled='write'):
        additional = {}
        if length is not None:
            additional['fieldtype'] = 'fixedstring,{}'.format(length)
        super().__init__(session, group, name=name, additional_attributes=additional,
                         timestamp=timestamp, write_enabled=write_enabled)
        self._length = length

    def writeable(self):
        return FixedStringField(self._session, self._field, write_enabled=True)

    @property
    def data(self):
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper

    # @property
    # def dtype(self):
    #     return "S{}".format(self._length)


class NumericField(Field):
    def __init__(self, session, group,
                 name=None, nformat=None, timestamp=None, write_enabled=False):
        additional = {}
        if nformat is not None:
            additional['fieldtype'] = 'fieldtype,{}'.format(nformat)
        super().__init__(session, group, name=name, additional_attributes=additional,
                         timestamp=timestamp, write_enabled=write_enabled)
        self._nformat = nformat

    def writeable(self):
        return NumericField(self._session, self._field, write_enabled=True)

    @property
    def data(self):
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper

    # @property
    # def dtype(self):
    #     return self._nformat


class CategoricalField(Field):
    def __init__(self, session, group,
                 name=None, key=None, nformat=None, timestamp=None, write_enabled=False):
        additional = {}
        key_fields = None
        if nformat is not None:
            additional['fieldtype'] = 'fieldtype,{}'.format(nformat)

        if key is not None:
            key_fields = dict()
            key_fields['key_values'] = ([v for k, v in key.items()], 'int8')
            key_fields['key_names'] = ([k for k, v in key.items()], h5py.special_dtype(vlen=str))

        super().__init__(session, group, name=name, additional_attributes=additional,
                         auxilliary_fields=key_fields, timestamp=timestamp,
                         write_enabled=write_enabled)
        self._nformat = nformat

    def writeable(self):
        return CategoricalField(self._session, self._field, write_enabled=True)

    @property
    def data(self):
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper

    # @property
    # def dtype(self):
    #     return self._nformat

    # Note: key is presented as value: str, even though the dictionary must be presented
    # as str: value
    @property
    def keys(self):
        kv = self._field['key_values']
        kn = self._field['key_names']
        keys = dict(zip(kv, kn))
        return keys





# class TimestampReader(Reader):
#     def __init__(self, datastore, field):
#         Reader.__init__(self, field)
#         if 'fieldtype' not in field.attrs.keys():
#             error = "{} must have 'fieldtype' in its attrs property"
#             raise ValueError(error.format(field))
#         fieldtype = field.attrs['fieldtype'].split(',')
#         if fieldtype[0] not in ('datetime', 'date', 'timestamp'):
#             error = "'fieldtype of '{} should be 'datetime' or 'date' but is {}"
#             raise ValueError(error.format(field, fieldtype))
#         self.chunksize = field.attrs['chunksize']
#         self.datastore = datastore
#
#     def __getitem__(self, item):
#         return self.field['values'][item]
#
#     def __len__(self):
#         return len(self.field['values'])
#
#     def get_writer(self, dest_group, dest_name, timestamp=None, write_mode='write'):
#         return TimestampWriter(self.datastore, dest_group, dest_name, timestamp,
#                                write_mode)
#
#     def dtype(self):
#         return self.field['values'].dtype
#
#
# write_modes = {'write', 'overwrite'}
#
#
# class Writer:
#     def __init__(self, datastore, group, name, write_mode, attributes):
#         self.trash_field = None
#         if write_mode not in write_modes:
#             raise ValueError(f"'write_mode' must be one of {write_modes}")
#         if name in group:
#             if write_mode == 'overwrite':
#                 field = group[name]
#                 trash = datastore.get_trash_group(field)
#                 dest_name = trash.name + f"/{name.split('/')[-1]}"
#                 group.move(field.name, dest_name)
#                 self.trash_field = trash[name]
#                 DataWriter.create_group(group, name, attributes)
#             else:
#                 error = (f"Field '{name}' already exists. Set 'write_mode' to 'overwrite' "
#                          "if you want to overwrite the existing contents")
#                 raise KeyError(error)
#         else:
#             DataWriter.create_group(group, name, attributes)
#         self.field = group[name]
#         self.name = name
#
#     def flush(self):
#         DataWriter.flush(self.field)
#         if self.trash_field is not None:
#             del self.trash_field
#
#
# class IndexedStringWriter(Writer):
#     def __init__(self, datastore, group, name,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         fieldtype = f'indexedstring'
#         super().__init__(datastore, group, name, write_mode,
#                          (('fieldtype', fieldtype), ('timestamp', timestamp),
#                           ('chunksize', datastore.chunksize)))
#         self.fieldtype = fieldtype
#         self.timestamp = timestamp
#         self.datastore = datastore
#
#         self.values = np.zeros(self.datastore.chunksize, dtype=np.uint8)
#         self.indices = np.zeros(self.datastore.chunksize, dtype=np.int64)
#         self.ever_written = False
#         self.accumulated = 0
#         self.value_index = 0
#         self.index_index = 0
#
#     def chunk_factory(self, length):
#         return [None] * length
#
#     def write_part(self, values):
#         """Writes a list of strings in indexed string form to a field
#         Args:
#             values: a list of utf8 strings
#         """
#         if not self.ever_written:
#             self.indices[0] = self.accumulated
#             self.index_index = 1
#             self.ever_written = True
#
#         for s in values:
#             evalue = s.encode()
#             for v in evalue:
#                 self.values[self.value_index] = v
#                 self.value_index += 1
#                 if self.value_index == self.datastore.chunksize:
#                     DataWriter.write(self.field, 'values', self.values, self.value_index)
#                     self.value_index = 0
#                 self.accumulated += 1
#             self.indices[self.index_index] = self.accumulated
#             self.index_index += 1
#             if self.index_index == self.datastore.chunksize:
#                 DataWriter.write(self.field, 'index', self.indices, self.index_index)
#                 self.index_index = 0
#
#     def flush(self):
#         if self.value_index != 0:
#             DataWriter.write(self.field, 'values', self.values, self.value_index)
#             self.value_index = 0
#         if self.index_index != 0:
#             DataWriter.write(self.field, 'index', self.indices, self.index_index)
#             self.index_index = 0
#         # self.field.attrs['fieldtype'] = self.fieldtype
#         # self.field.attrs['timestamp'] = self.timestamp
#         # self.field.attrs['chunksize'] = self.chunksize
#         # self.field.attrs['completed'] = True
#         super().flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#     def write_part_raw(self, index, values):
#         if index.dtype != np.int64:
#             raise ValueError(f"'index' must be an ndarray of '{np.int64}'")
#         if values.dtype != np.uint8:
#             raise ValueError(f"'values' must be an ndarray of '{np.uint8}'")
#         DataWriter.write(self.field, 'index', index, len(index))
#         DataWriter.write(self.field, 'values', values, len(values))
#
#     def write_raw(self, index, values):
#         self.write_part_raw(index, values)
#         self.flush()
#
#
# # TODO: should produce a warning for unmappable strings and a corresponding filter, rather
# # than raising an exception; or at least have a mode where this is possible
# class LeakyCategoricalImporter:
#     def __init__(self, datastore, group, name, categories, out_of_range,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         self.writer = CategoricalWriter(datastore, group, name,
#                                         categories, timestamp, write_mode)
#         self.other_values = IndexedStringWriter(datastore, group, f"{name}_{out_of_range}",
#                                                 timestamp, write_mode)
#         self.field_size = max([len(k) for k in categories.keys()])
#
#     def chunk_factory(self, length):
#         return np.zeros(length, dtype=f'U{self.field_size}')
#
#     def write_part(self, values):
#         results = np.zeros(len(values), dtype='int8')
#         strresults = list([""] * len(values))
#         keys = self.writer.keys
#         anomalous_count = 0
#         for i in range(len(values)):
#             value = keys.get(values[i], -1)
#             if value != -1:
#                 results[i] = value
#             else:
#                 anomalous_count += 1
#                 results[i] = -1
#                 strresults[i] = values[i]
#         self.writer.write_part(results)
#         self.other_values.write_part(strresults)
#
#     def flush(self):
#         # add a 'freetext' value to keys
#         self.writer.keys['freetext'] = -1
#         self.writer.flush()
#         self.other_values.flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# # TODO: should produce a warning for unmappable strings and a corresponding filter, rather
# # than raising an exception; or at least have a mode where this is possible
# class CategoricalImporter:
#     def __init__(self, datastore, group, name, categories,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         self.writer = CategoricalWriter(datastore, group, name,
#                                         categories, timestamp, write_mode)
#         self.field_size = max([len(k) for k in categories.keys()])
#
#     def chunk_factory(self, length):
#         return np.zeros(length, dtype=f'U{self.field_size}')
#
#     def write_part(self, values):
#         results = np.zeros(len(values), dtype='int8')
#         keys = self.writer.keys
#         for i in range(len(values)):
#             results[i] = keys[values[i]]
#         self.writer.write_part(results)
#
#     def flush(self):
#         self.writer.flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class CategoricalWriter(Writer):
#     def __init__(self, datastore, group, name, categories,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         fieldtype = f'categorical'
#         super().__init__(datastore, group, name, write_mode,
#                          (('fieldtype', fieldtype), ('timestamp', timestamp),
#                           ('chunksize', datastore.chunksize)))
#         self.fieldtype = fieldtype
#         self.timestamp = timestamp
#         self.datastore = datastore
#         self.keys = categories
#
#
#     def chunk_factory(self, length):
#         return np.zeros(length, dtype='int8')
#
#     def write_part(self, values):
#         DataWriter.write(self.field, 'values', values, len(values))
#
#     def flush(self):
#         key_strs = list()
#         key_values = np.zeros(len(self.keys), dtype='int8')
#         items = self.keys.items()
#         for i, kv in enumerate(items):
#             k, v = kv
#             key_strs.append(k)
#             key_values[i] = v
#         DataWriter.write(self.field, 'key_values', key_values, len(key_values))
#         DataWriter.write(self.field, 'key_names', key_strs, len(key_strs),
#                          dtype=h5py.string_dtype())
#         # self.field.attrs['fieldtype'] = self.fieldtype
#         # self.field.attrs['timestamp'] = self.timestamp
#         # self.field.attrs['chunksize'] = self.chunksize
#         # self.field.attrs['completed'] = True
#         super().flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class NumericImporter:
#     def __init__(self, datastore, group, name, nformat, parser,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         self.data_writer = NumericWriter(datastore, group, name,
#                                          nformat, timestamp, write_mode)
#         self.flag_writer = NumericWriter(datastore, group, f"{name}_valid",
#                                          'bool', timestamp, write_mode)
#         self.parser = parser
#
#     def chunk_factory(self, length):
#         return [None] * length
#
#     def write_part(self, values):
#         """
#         Given a list of strings, parse the strings and write the parsed values. Values that
#         cannot be parsed are written out as zero for the values, and zero for the flags to
#         indicate that that entry is not valid.
#         Args:
#             values: a list of strings to be parsed
#         """
#         elements = np.zeros(len(values), dtype=self.data_writer.nformat)
#         validity = np.zeros(len(values), dtype='bool')
#         for i in range(len(values)):
#             valid, value = self.parser(values[i])
#             elements[i] = value
#             validity[i] = valid
#         self.data_writer.write_part(elements)
#         self.flag_writer.write_part(validity)
#
#     def flush(self):
#         self.data_writer.flush()
#         self.flag_writer.flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class NumericWriter(Writer):
#     def __init__(self, datastore, group, name, nformat,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         fieldtype = f'numeric,{nformat}'
#         super().__init__(datastore, group, name, write_mode,
#                          (('fieldtype', fieldtype), ('timestamp', timestamp),
#                           ('chunksize', datastore.chunksize), ('nformat', nformat)))
#         self.fieldtype = fieldtype
#         self.nformat = nformat
#         self.timestamp = timestamp
#         self.datastore = datastore
#
#     def chunk_factory(self, length):
#         nformat = self.fieldtype.split(',')[1]
#         return np.zeros(length, dtype=nformat)
#
#     def write_part(self, values):
#         if not np.issubdtype(values.dtype, self.nformat):
#             values = values.astype(self.nformat)
#         DataWriter.write(self.field, 'values', values, len(values))
#
#     def flush(self):
#         # self.field.attrs['fieldtype'] = self.fieldtype
#         # self.field.attrs['timestamp'] = self.timestamp
#         # self.field.attrs['chunksize'] = self.chunksize
#         # self.field.attrs['nformat'] = self.nformat
#         # self.field.attrs['completed'] = True
#         super().flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class FixedStringWriter(Writer):
#     def __init__(self, datastore, group, name, strlen,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         fieldtype = f'fixedstring,{strlen}'
#         super().__init__(datastore, group, name, write_mode,
#                          (('fieldtype', fieldtype), ('timestamp', timestamp),
#                           ('chunksize', datastore.chunksize), ('strlen', strlen)))
#         self.fieldtype = fieldtype
#         self.timestamp = timestamp
#         self.datastore = datastore
#         self.strlen = strlen
#
#     def chunk_factory(self, length):
#         return np.zeros(length, dtype=f'S{self.strlen}')
#
#     def write_part(self, values):
#         DataWriter.write(self.field, 'values', values, len(values))
#
#     def flush(self):
#         # self.field.attrs['fieldtype'] = self.fieldtype
#         # self.field.attrs['timestamp'] = self.timestamp
#         # self.field.attrs['chunksize'] = self.chunksize
#         # self.field.attrs['strlen'] = self.strlen
#         # self.field.attrs['completed'] = True
#         super().flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class DateTimeImporter:
#     def __init__(self, datastore, group, name,
#                  optional=True, timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         self.datetime = DateTimeWriter(datastore, group, name,
#                                        timestamp, write_mode)
#         self.datestr = FixedStringWriter(datastore, group, f"{name}_day",
#                                          '10', timestamp, write_mode)
#         self.datetimeset = None
#         if optional:
#             self.datetimeset = NumericWriter(datastore, group, f"{name}_set",
#                                              'bool', timestamp, write_mode)
#
#     def chunk_factory(self, length):
#         return self.datetime.chunk_factory(length)
#
#     def write_part(self, values):
#         # TODO: use a timestamp writer instead of a datetime writer and do the conversion here
#
#         days = self.datestr.chunk_factory(len(values))
#         flags = None
#         if self.datetimeset is not None:
#             flags = self.datetimeset.chunk_factory(len(values))
#             for i in range(len(values)):
#                 flags[i] = values[i] != b''
#                 days[i] = values[i][:10]
#         else:
#             for i in range(len(values)):
#                 days[i] = values[i][:10]
#
#         self.datetime.write_part(values)
#         self.datestr.write_part(days)
#         if self.datetimeset is not None:
#             self.datetimeset.write_part(flags)
#
#     def flush(self):
#         self.datetime.flush()
#         self.datestr.flush()
#         if self.datetimeset is not None:
#             self.datetimeset.flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# # TODO writers can write out more than one field; offset could be done this way
# class DateTimeWriter(Writer):
#     def __init__(self, datastore, group, name,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         fieldtype = f'datetime'
#         super().__init__(datastore, group, name, write_mode,
#                          (('fieldtype', fieldtype), ('timestamp', timestamp),
#                           ('chunksize', datastore.chunksize)))
#         self.fieldtype = fieldtype
#         self.timestamp = timestamp
#         self.datastore = datastore
#
#     def chunk_factory(self, length):
#         return np.zeros(length, dtype=f'S32')
#
#     def write_part(self, values):
#         timestamps = np.zeros(len(values), dtype=np.float64)
#         for i in range(len(values)):
#             value = values[i]
#             if value == b'':
#                 timestamps[i] = 0
#             else:
#                 if len(value) == 32:
#                     ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S.%f%z')
#                 elif len(value) == 25:
#                     ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S%z')
#                 else:
#                     raise ValueError(f"Date field '{self.field}' has unexpected format '{value}'")
#                 timestamps[i] = ts.timestamp()
#         DataWriter.write(self.field, 'values', timestamps, len(timestamps))
#
#     def flush(self):
#         # self.field.attrs['fieldtype'] = self.fieldtype
#         # self.field.attrs['timestamp'] = self.timestamp
#         # self.field.attrs['chunksize'] = self.chunksize
#         # self.field.attrs['completed'] = True
#         super().flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class DateWriter(Writer):
#     def __init__(self, datastore, group, name,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         fieldtype = 'date'
#         super().__init__(datastore, group, name, write_mode,
#                          (('fieldtype', fieldtype), ('timestamp', timestamp),
#                           ('chunksize', datastore.chunksize)))
#         self.fieldtype = fieldtype
#         self.timestamp = timestamp
#         self.datastore = datastore
#
#     def chunk_factory(self, length):
#         return np.zeros(length, dtype=f'S10')
#
#     def write_part(self, values):
#
#         timestamps = np.zeros(len(values), dtype=np.float64)
#         for i in range(len(values)):
#             value = values[i]
#             if value == b'':
#                 timestamps[i] = 0
#             else:
#                 ts = datetime.strptime(value.decode(), '%Y-%m-%d')
#                 timestamps[i] = ts.timestamp()
#         DataWriter.write(self.field, 'values', timestamps, len(timestamps))
#
#     def flush(self):
#         # self.field.attrs['fieldtype'] = self.fieldtype
#         # self.field.attrs['timestamp'] = self.timestamp
#         # self.field.attrs['chunksize'] = self.chunksize
#         # self.field.attrs['completed'] = True
#         super().flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class TimestampWriter(Writer):
#     def __init__(self, datastore, group, name,
#                  timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         fieldtype = 'timestamp'
#         super().__init__(datastore, group, name, write_mode,
#                          (('fieldtype', fieldtype), ('timestamp', timestamp),
#                           ('chunksize', datastore.chunksize)))
#         self.fieldtype = fieldtype
#         self.timestamp = timestamp
#         self.datastore = datastore
#
#     def chunk_factory(self, length):
#         return np.zeros(length, dtype=f'float64')
#
#     def write_part(self, values):
#         DataWriter.write(self.field, 'values', values, len(values))
#
#     def flush(self):
#         # self.field.attrs['fieldtype'] = self.fieldtype
#         # self.field.attrs['timestamp'] = self.timestamp
#         # self.field.attrs['chunksize'] = self.chunksize
#         # self.field.attrs['completed'] = True
#         Writer.flush(self)
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
#
#
# class OptionalDateImporter:
#     def __init__(self, datastore, group, name,
#                  optional=True, timestamp=None, write_mode='write'):
#         if timestamp is None:
#             timestamp = datastore.timestamp
#         self.date = DateWriter(datastore, group, name, timestamp, write_mode)
#         self.datestr = FixedStringWriter(datastore, group, f"{name}_day",
#                                          '10', timestamp, write_mode)
#         self.dateset = None
#         if optional:
#             self.dateset =\
#                 NumericWriter(datastore, group, f"{name}_set", 'bool', timestamp, write_mode)
#
#     def chunk_factory(self, length):
#         return self.date.chunk_factory(length)
#
#     def write_part(self, values):
#         # TODO: use a timestamp writer instead of a datetime writer and do the conversion here
#         days = self.datestr.chunk_factory(len(values))
#         flags = None
#         if self.dateset is not None:
#             flags = self.dateset.chunk_factory(len(values))
#             for i in range(len(values)):
#                 flags[i] = values[i] != b''
#                 days[i] = values[i][:10]
#         else:
#             for i in range(len(values)):
#                 days[i] = values[i][:10]
#
#         self.date.write_part(values)
#         self.datestr.write_part(days)
#         if self.dateset is not None:
#             self.dateset.write_part(flags)
#
#     def flush(self):
#         self.date.flush()
#         self.datestr.flush()
#         if self.dateset is not None:
#             self.dateset.flush()
#
#     def write(self, values):
#         self.write_part(values)
#         self.flush()
