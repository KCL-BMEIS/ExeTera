from datetime import datetime
from threading import Thread

import h5py

import numpy as np


class DataWriter:

    @staticmethod
    def clear_dataset(parent_group, name):
        t = Thread(target=DataWriter._clear_dataset,
                   args=(parent_group, name))
        t.start()
        t.join()

    @staticmethod
    def _clear_dataset(field, name):
        del field[name]

    @staticmethod
    def _create_group(parent_group, name, attrs):
        group = parent_group.create_group(name)
        for k, v in attrs:
            try:
                group.attrs[k] = v
            except Exception as e:
                print(f"Exception {e} caught while assigning attribute {k} value {v}")
                raise
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
                    name, (count,), maxshape=(None,), chunks=(1 << 20,), dtype=dtype)
                ds[:] = field
            else:
                ds = group.create_dataset(
                    name, (count,), maxshape=(None,), chunks=(1 << 20,), dtype=dtype)
                ds[:] = field[:count]
        else:
            if count == len(field):
                group.create_dataset(name, (count,), maxshape=(None,), chunks=(1 << 20,),
                                     data=field)
            else:
                group.create_dataset(name, (count,), maxshape=(None,), chunks=(1 << 20,),
                                     data=field[:count])

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
        try:
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
        except Exception as e:
            print("{}: unexpected exception {}".format(self.field.name, e))
            raise

    def __len__(self):
        return len(self.field['index']) - 1

    def get_writer(self, dest_group, dest_name, timestamp=None, write_mode='write'):
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

    def get_writer(self, dest_group, dest_name, timestamp=None, write_mode='write'):
        return NumericWriter(self.datastore, dest_group, dest_name,
                             self.field.attrs['fieldtype'].split(',')[1],
                             timestamp, write_mode)

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
        kv = self.field['key_values'][:]
        kn = self.field['key_names'][:]
        self.keys = dict(zip(kv, kn))
        # self.keys = self.field['keys'][()]
        self.datastore = datastore

    def __getitem__(self, item):
        return self.field['values'][item]

    def __len__(self):
        return len(self.field['values'])

    def get_writer(self, dest_group, dest_name, timestamp=None, write_mode='write'):
        keys = {v: k for k, v in zip(self.field['key_values'][:], self.field['key_names'][:])}
        return CategoricalWriter(self.datastore, dest_group, dest_name, keys,
                                 timestamp, write_mode)

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

    def get_writer(self, dest_group, dest_name, timestamp=None, write_mode='write'):
        return FixedStringWriter(self.datastore, dest_group, dest_name,
                                 self.field.attrs['fieldtype'].split(',')[1],
                                 timestamp, write_mode)

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

    def get_writer(self, dest_group, dest_name, timestamp=None, write_mode='write'):
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
                DataWriter.create_group(group, name, attributes)
            else:
                error = (f"Field '{name}' already exists. Set 'write_mode' to 'overwrite' "
                         "if you want to overwrite the existing contents")
                raise KeyError(error)
        else:
            DataWriter.create_group(group, name, attributes)
        self.field = group[name]
        self.name = name

    def flush(self):
        DataWriter.flush(self.field)
        if self.trash_field is not None:
            del self.trash_field


class IndexedStringWriter(Writer):
    def __init__(self, datastore, group, name,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
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
class LeakyCategoricalImporter:
    def __init__(self, datastore, group, name, categories, out_of_range,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
        self.writer = CategoricalWriter(datastore, group, name,
                                        categories, timestamp, write_mode)
        self.other_values = IndexedStringWriter(datastore, group, f"{name}_{out_of_range}",
                                                timestamp, write_mode)
        self.field_size = max([len(k) for k in categories.keys()])

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'U{self.field_size}')

    def write_part(self, values):
        results = np.zeros(len(values), dtype='int8')
        strresults = list([""] * len(values))
        keys = self.writer.keys
        anomalous_count = 0
        for i in range(len(values)):
            value = keys.get(values[i], -1)
            if value != -1:
                results[i] = value
            else:
                anomalous_count += 1
                results[i] = -1
                strresults[i] = values[i]
        self.writer.write_part(results)
        self.other_values.write_part(strresults)

    def flush(self):
        # add a 'freetext' value to keys
        self.writer.keys['freetext'] = -1
        self.writer.flush()
        self.other_values.flush()

    def write(self, values):
        self.write_part(values)
        self.flush()


# TODO: should produce a warning for unmappable strings and a corresponding filter, rather
# than raising an exception; or at least have a mode where this is possible
class CategoricalImporter:
    def __init__(self, datastore, group, name, categories,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
        self.writer = CategoricalWriter(datastore, group, name,
                                        categories, timestamp, write_mode)
        self.field_size = max([len(k) for k in categories.keys()])

    def chunk_factory(self, length):
        return np.zeros(length, dtype=f'U{self.field_size}')

    def write_part(self, values):
        results = np.zeros(len(values), dtype='int8')
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
    def __init__(self, datastore, group, name, categories,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
        fieldtype = f'categorical'
        super().__init__(datastore, group, name, write_mode,
                         (('fieldtype', fieldtype), ('timestamp', timestamp),
                          ('chunksize', datastore.chunksize)))
        self.fieldtype = fieldtype
        self.timestamp = timestamp
        self.datastore = datastore
        # string:number
        self.keys = categories


    def chunk_factory(self, length):
        return np.zeros(length, dtype='int8')

    def write_part(self, values):
        DataWriter.write(self.field, 'values', values, len(values))

    def flush(self):
        key_strs = list()
        key_values = np.zeros(len(self.keys), dtype='int8')
        items = self.keys.items()
        for i, kv in enumerate(items):
            k, v = kv
            key_strs.append(k)
            key_values[i] = v
        DataWriter.write(self.field, 'key_values', key_values, len(key_values))
        DataWriter.write(self.field, 'key_names', key_strs, len(key_strs),
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
    def __init__(self, datastore, group, name, nformat, parser,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
        self.data_writer = NumericWriter(datastore, group, name,
                                         nformat, timestamp, write_mode)
        self.flag_writer = NumericWriter(datastore, group, f"{name}_valid",
                                         'bool', timestamp, write_mode)
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
    def __init__(self, datastore, group, name, nformat,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
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
        if not np.issubdtype(values.dtype, np.dtype(self.nformat)):
            values = values.astype(self.nformat)
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
    def __init__(self, datastore, group, name, strlen,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
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
    def __init__(self, datastore, group, name,
                 optional=True, timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
        self.datetime = DateTimeWriter(datastore, group, name,
                                       timestamp, write_mode)
        self.datestr = FixedStringWriter(datastore, group, f"{name}_day",
                                         '10', timestamp, write_mode)
        self.datetimeset = None
        if optional:
            self.datetimeset = NumericWriter(datastore, group, f"{name}_set",
                                             'bool', timestamp, write_mode)

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
    def __init__(self, datastore, group, name,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
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
                    # ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S.%f%z')
                    ts = datetime(int(value[0:4]), int(value[5:7]), int(value[8:10]),
                                  int(value[11:13]), int(value[14:16]), int(value[17:19]),
                                  int(value[20:26]))
                elif len(value) == 25:
                    # ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S%z')
                    ts = datetime(int(value[0:4]), int(value[5:7]), int(value[8:10]),
                                  int(value[11:13]), int(value[14:16]), int(value[17:19]))
                elif len(value) == 19:
                    ts = datetime(int(value[0:4]), int(value[5:7]), int(value[8:10]),
                                  int(value[11:13]), int(value[14:16]), int(value[17:19]))
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
    def __init__(self, datastore, group, name,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
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
    def __init__(self, datastore, group, name,
                 timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
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
    def __init__(self, datastore, group, name,
                 optional=True, timestamp=None, write_mode='write'):
        if timestamp is None:
            timestamp = datastore.timestamp
        self.date = DateWriter(datastore, group, name, timestamp, write_mode)
        self.datestr = FixedStringWriter(datastore, group, f"{name}_day",
                                         '10', timestamp, write_mode)
        self.dateset = None
        if optional:
            self.dateset =\
                NumericWriter(datastore, group, f"{name}_set", 'bool', timestamp, write_mode)

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
