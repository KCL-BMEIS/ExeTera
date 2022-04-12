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

from typing import Callable, Optional, Union
from datetime import datetime, timezone
import operator

import numpy as np
import h5py

from exetera.core.abstract_types import Field
from exetera.core.data_writer import DataWriter
from exetera.core import operations as ops
from exetera.core import validation as val


def isin(field:Field, test_elements:Union[list, set, np.ndarray]):
    """
    Returns a boolean array of the same length as field
    that is True where an element of field is in test_elements and False otherwise.

    :param field: The field to check.
    :param test_elements: The values against which to test each value of field.
    :return: a boolean array of the same length as field
    """
    if not isinstance(field, Field):
        raise Exception("'field' should be field type")

    ret = NumericMemField(field._session, 'bool')
    ret.data.write(FieldDataOps.apply_isin(field, test_elements))
    return ret


class HDF5Field(Field):
    def __init__(self, session, group, dataframe, write_enabled=False):
        super().__init__()

        # if name is None:
        #     field = group
        # else:
        #     field = group[name]
        self._session = session
        self._field = group
        self._fieldtype = self._field.attrs['fieldtype']
        self._dataframe = dataframe
        self._write_enabled = write_enabled
        self._value_wrapper = None
        self._valid_reference = True

    @property
    def valid(self):
        """
        Returns whether the field is a valid field object. Fields can become invalid as a result
        of certain operations, such as a field being moved from one dataframe to another. A field
        that is invalid with throw exceptions if any other operation is performed on them.
        :return: bool
        """
        return self._valid_reference

    @property
    def name(self):
        """
        The name of the field within a dataframe, if the field belongs to a dataframe
        :return: str
        """
        self._ensure_valid()
        return self._field.name.split('/')[-1]

    @property
    def dataframe(self):
        """
        The owning dataframe of this field, or None if the field is not owned by a dataframe
        :return str or None
        """
        self._ensure_valid()
        return self._dataframe

    @property
    def timestamp(self):
        """
        The timestamp representing the field creation time. This is the time at which the data
        for this field was added to the dataset, rather than the point at which the field wrapper
        was created.
        :return: timestamp
        """
        self._ensure_valid()
        return self._field.attrs['timestamp']

    @property
    def chunksize(self):
        """
        The chunksize for the field. This is not generally required for users, and may be
        ignored depending on the storage medium.
        """
        self._ensure_valid()
        return self._field.attrs['chunksize']

    @property
    def indexed(self):
        """
        Whether the field is an indexed field or not. Indexed fields store their data internally
        as index and value arrays for efficiency, as well as making it accessible through the data
        property.
        """
        self._ensure_valid()
        return False

    def __bool__(self):
        # this method is required to prevent __len__ being called on derived methods when fields are queried as
        #   if f:
        # rather than
        #   if f is not None:
        self._ensure_valid()
        return True

    def get_spans(self):
        """
        Get spans of the field.
        """
        raise NotImplementedError("Please use get_spans() on specific fields, not the field base class.")

    def apply_filter(self, filter_to_apply, dstfld=None):
        """
        Apply filter on the field.
        """
        raise NotImplementedError("Please use apply_filter() on specific fields, not the field base class.")

    def apply_index(self, index_to_apply, dstfld=None):
        """
        Apply index on the field.
        """
        raise NotImplementedError("Please use apply_index() on specific fields, not the field base class.")

    def _ensure_valid(self):
        if not self._valid_reference:
            raise ValueError("This field no longer refers to a valid underlying field object")


class MemoryField(Field):

    def __init__(self, session):
        super().__init__()
        self._session = session
        self._write_enabled = True
        self._value_wrapper = None

    @property
    def valid(self):
        """
        Returns whether the field is a valid field object. Fields can become invalid as a result
        of certain operations, such as a field being moved from one dataframe to another. A field
        that is invalid with throw exceptions if any other operation is performed on them.
        :return: bool
        """
        return True

    @property
    def name(self):
        """
        The name of the field within a dataframe, if the field belongs to a dataframe
        :return: str or None
        """
        return None

    @property
    def dataframe(self):
        """
        The owning dataframe of this field, or None if the field is now owned by a dataframe
        """
        return None

    @property
    def timestamp(self):
        """
        The timestamp representing the field creation time. This is the time at which the data
        for this field was added to the dataset, rather than the point at which the field wrapper
        was created.
        """
        return None

    @property
    def chunksize(self):
        """
        The chunksize for the field. This is not generally required for users, and may be
        ignored depending on the storage medium.
        """
        return None

    @property
    def indexed(self):
        """
        Whether the field is an indexed field or not. Indexed fields store their data internally
        as index and value arrays for efficiency, as well as making it accessible through the data
        property.
        """
        return False

    def __bool__(self):
        # this method is required to prevent __len__ being called on derived methods when fields are queried as
        #   if f:
        # rather than
        #   if f is not None:
        return True

    def apply_filter(self, filter_to_apply, dstfld=None):
        """
        Apply filter on the field.
        """
        raise NotImplementedError("Please use apply_filter() on specific fields, not the field base class.")

    def apply_index(self, index_to_apply, dstfld=None):
        """
        Apply index on the field.
        """
        raise NotImplementedError("Please use apply_index() on specific fields, not the field base class.")


class ReadOnlyFieldArray:
    def __init__(self, field, dataset_name):
        self._field = field
        self._name = dataset_name
        self._dataset = field[dataset_name]

    def __len__(self):
        return len(self._dataset)

    @property
    def dtype(self):
        """
        Return datatype of field.
        """
        return self._dataset.dtype

    def __getitem__(self, item):
        return self._dataset[item]

    def __setitem__(self, key, value):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def clear(self):
        """
        Clear Field Array.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write_part(self, part):
        """
        Write data part to field.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write(self, part):
        """
        Write data to field.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def complete(self):
        """
        Mark writing completed, usually used after calling write_part.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")


# Field arrays
# ============

class WriteableFieldArray:
    def __init__(self, field, dataset_name):
        self._field = field
        self._name = dataset_name
        self._dataset = field[dataset_name]

    def __len__(self):
        """
        Returns length of field array
        :return: int
        """
        return len(self._dataset)

    @property
    def dtype(self):
        """
        Returns the datatype for the dataset
        :return: dtype
        """
        return self._dataset.dtype

    def __getitem__(self, item):
        return self._dataset[item]

    def __setitem__(self, key, value):
        self._dataset[key] = value

    def clear(self):
        """
        Replaces current dataset with empty dataset.
        :return: None
        """
        nformat = self._dataset.dtype
        DataWriter._clear_dataset(self._field, self._name)
        DataWriter.write(self._field, self._name, [], 0, nformat)
        self._dataset = self._field[self._name]

    def write_part(self, part):
        """
        Writes data part to field, followed by calling complete().

        Example::
            part = np.array([97, 97, 100])
            field.write_part(part)
            field.complete()

        :param part: numpy array to write to field
        :return: None
        """
        DataWriter.write(self._field, self._name, part, len(part), dtype=self._dataset.dtype)

    def write(self, part):
        """
        Writes data to field and marks it as complete.

        Example::
            part = np.array([97, 97, 100])
            field.write(part)

        :param part: numpy array to write to field
        :return: None
        """
        if isinstance(part, Field):
            part = part.data[:]
        DataWriter.write(self._field, self._name, part, len(part), dtype=self._dataset.dtype)
        self.complete()

    def complete(self):
        """
        Mark writing completed, usually used after calling write_part.

        Example::
            field.write_part(part)
            field.complete()

        :return: None
        """
        DataWriter.flush(self._field[self._name])


class MemoryFieldArray:

    def __init__(self, dtype):
        """
        :param dtype: datatype (dtype) to be used for this memory field.
        """
        self._dtype = dtype
        self._dataset = None

    def __len__(self):
        """
        Returns length of field dataset, returns 0 if no dataset for field. 
        :return: int
        """
        return 0 if self._dataset is None else len(self._dataset)

    @property
    def dtype(self):
        """
        :return: dtype of field
        """
        return self._dtype

    def __getitem__(self, item):
        """
        :param item: index of item in dataset.
        :return: item in field dataset
        """
        if self._dataset is None:
            # raise ValueError("Cannot get data from an empty Field")
            return np.zeros(0, dtype=np.uint8)
        return self._dataset[item]

    def __setitem__(self, key, value):
        """
        Set a field with given name and given field data

        :param key: index of item in field dataset.
        :param value: value to set on index
        :return: None
        """
        self._dataset[key] = value

    def clear(self):
        """
        Set dataset to None
        :return: None
        """
        self._dataset = None

    def write_part(self, part, move_mem=False):
        """
        Writes data part to field, followed by calling complete().

        Example::
            part = np.array([97, 97, 100])
            field.write_part(part)
            field.complete()

        :param part: numpy array to written to field
        :param move_mem: boolean, use part provided directly or make copy before writing. 
        :return: None
        """
        if not isinstance(part, np.ndarray):
            raise ValueError("'part' must be a numpy array but is '{}'".format(type(part)))
        if self._dataset is None:
            if move_mem is True and dtype_to_str(part.dtype) == self._dtype:
                self._dataset = part
            else:
                self._dataset = part.copy()
        else:
            new_dataset = np.zeros(len(self._dataset) + len(part), dtype=self._dataset.dtype)
            new_dataset[:len(self._dataset)] = self._dataset
            new_dataset[-len(part):] = part
            self._dataset = new_dataset

    def write(self, part):
        """
        Writes data to field and marks it as complete.

        Example::
            part = np.array([97, 97, 100])
            field.write(part)

        :param part: numpy array to write to field
        :return: None
        """
        self.write_part(part)
        self.complete()

    def complete(self):
        """
        Mark writing completed, usually used after calling write_part.
        """
        pass


class ReadOnlyIndexedFieldArray:
    def __init__(self, field, indices, values):
        """
        :param field: Field to use
        :param indices: Indices for numpy array
        :param values: Values for numpy array
        :return: None
        """
        self._field = field
        self._indices = indices
        self._values = values

    def __len__(self):
        """
        Length of field 
        :return: int
        """
        # TODO: this occurs because of the initialized state of an indexed string. It would be better for the
        # index to be initialised as [0]
        return max(len(self._indices) - 1, 0)

    @property
    def dtype(self):
        """
        Get datatype of field.
        """
        return self._dtype  # todo

    def __getitem__(self, item):
        """
        Return item in field dataset on index

        :param item: Index
        :return: Item value from dataset
        """
        try:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop if item.stop is not None else len(self._indices) - 1
                step = item.step
                # TODO: validate slice
                index = self._indices[start:stop + 1]
                bytestr = self._values[index[0]:index[-1]]
                results = [None] * (len(index) - 1)
                startindex = self._indices[start]
                for ir in range(len(results)):
                    results[ir] = \
                        bytestr[index[ir] - np.int64(startindex):
                                index[ir + 1] - np.int64(startindex)].tobytes().decode()
                return results
            elif isinstance(item, int):
                if item >= len(self._indices) - 1:
                    raise ValueError("index is out of range")
                start, stop = self._indices[item:item + 2]
                if start == stop:
                    return ''
                value = self._values[start:stop].tobytes().decode()
                return value
        except Exception as e:
            print("{}: unexpected exception {}".format(self._field.name, e))
            raise

    def __setitem__(self, key, value):
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def clear(self):
        """
        Clears field array.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write_part(self, part):
        """
        Writes data part to field.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def write(self, part):
        """
        Writes data to field.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")

    def complete(self):
        """
        Mark writing completed, usually used after calling write_part.
        """
        raise PermissionError("This field was created read-only; call <field>.writeable() "
                              "for a writeable copy of the field")


class WriteableIndexedFieldArray:
    def __init__(self, chunksize, indices, values):
        """
        :param: chunksize: Size of each chunk
        :param indices: Numpy array of indices
        :param values: Numpy array of values 
        :return: None
        """
        # self._field = field
        self._indices = indices
        self._values = values
        # self._chunksize = self._field.attrs['chunksize']
        self._chunksize = chunksize
        self._raw_values = np.zeros(self._chunksize, dtype=np.uint8)
        self._raw_indices = np.zeros(self._chunksize, dtype=np.int64)
        self._accumulated = self._indices[-1] if len(self._indices) > 0 else 0
        self._index_index = 0
        self._value_index = 0

    def __len__(self):
        """
        Length of field 
        :return: int
        """
        return max(len(self._indices) - 1, 0)

    @property
    def dtype(self):
        """
        Returns datatype of field
        :return: dtype
        """
        return self._dtype  # todo

    def __getitem__(self, item):
        """
        Return item in field dataset on index
        :param item: int or slice
        :return: Item value from dataset
        """
        try:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop if item.stop is not None else len(self._indices) - 1
                step = item.step
                # TODO: validate slice

                index = self._indices[start:stop + 1]
                if len(index) == 0:
                    return []
                bytestr = self._values[index[0]:index[-1]]
                results = [None] * (len(index) - 1)
                startindex = self._indices[start]
                rmax = min(len(results), stop - start)
                for ir in range(rmax):
                    rbytes = bytestr[index[ir] - np.int64(startindex):
                                     index[ir + 1] - np.int64(startindex)].tobytes()
                    rstr = rbytes.decode()
                    results[ir] = rstr
                return results
            elif isinstance(item, int):
                if item >= len(self._indices) - 1:
                    raise ValueError("index is out of range")
                start, stop = self._indices[item:item + 2]
                if start == stop:
                    return ''
                value = self._values[start:stop].tobytes().decode()
                return value
        except Exception as e:
            print(e)
            raise

    def __setitem__(self, key, value):
        raise PermissionError("IndexedStringField instances cannot be edited via array syntax;"
                              "use clear and then write/write_part or write_raw/write_part_raw")

    def clear(self):
        """
        Resets field, clears all indices and values.

        :return: None
        """
        self._accumulated = 0
        self._indices.clear()
        self._values.clear()
        self._accumulated = 0

    def write_part(self, part):
        """
        Writes data part to field, followed by calling complete().

        Example::
            part = np.array([97, 97, 100])
            field.write_part(part)
            field.complete()

        :param part: List of strings to be written
        :return: None
        """
        for s in part:
            evalue = s.encode()
            for v in evalue:
                self._raw_values[self._value_index] = v
                self._value_index += 1
                if self._value_index == self._chunksize:
                    self._values.write_part(self._raw_values[:self._value_index])
                    self._value_index = 0
                self._accumulated += 1
            self._raw_indices[self._index_index] = self._accumulated
            self._index_index += 1
            if self._index_index == self._chunksize:
                if len(self._indices) == 0:
                    self._indices.write_part(np.array([0]))
                self._indices.write_part(self._raw_indices[:self._index_index])
                self._index_index = 0

    def write(self, part):
        """
        Writes data to field and marks it as complete.

        Example::
            part = np.array([97, 97, 100])
            field.write(part)

        :param part: List of strings to write to field
        :return: None
        """
        self.write_part(part)
        self.complete()

    def complete(self):
        """
        Mark writing completed, usually used after calling write_part.

        Example::
            field.write_part(part)
            field.complete()

        :return: None
        """
        if self._value_index != 0:
            self._values.write(self._raw_values[:self._value_index])
            self._value_index = 0
        if self._index_index != 0:
            if len(self._indices) == 0:
                self._indices.write_part(np.array([0]))
            self._indices.write(self._raw_indices[:self._index_index])
            self._index_index = 0


# Memory-based fields
# ===================


class IndexedStringMemField(MemoryField):
    def __init__(self, session, chunksize=None):
        super().__init__(session)
        self._session = session
        self._chunksize = session.chunksize if chunksize is None else chunksize
        self._data_wrapper = None
        self._index_wrapper = None
        self._value_wrapper = None

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        return self

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        return FieldDataOps.indexed_string_create_like(self, group, name, timestamp)

    @property
    def indexed(self):
        """
        Whether the field is an indexed field or not. Indexed fields store their data internally
        as index and value arrays for efficiency, as well as making it accessible through the data
        property.
        """
        return True

    @property
    def data(self):
        """
        Returns indexed writable field array with values from this field
        :return: WriteableIndexedFieldArray
        """
        if self._data_wrapper is None:
            self._data_wrapper = WriteableIndexedFieldArray(self._chunksize, self.indices, self.values)
        return self._data_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        if len(self) < 2:
            return True

        indices = self.indices[:]
        values = self.values[:]
        last = values[indices[0]:indices[1]].tobytes()
        for i in range(1, len(indices) - 1):
            cur = values[indices[i]:indices[i + 1]].tobytes()
            if last > cur:
                return False
            last = cur
        return True

    @property
    def indices(self):
        """
        Get indices for field
        :return: MemoryFieldArray('int64')
        """
        if self._index_wrapper is None:
            self._index_wrapper = MemoryFieldArray('int64')
        return self._index_wrapper

    @property
    def values(self):
        """
        Get values for field
        :return: MemoryFieldArray('8')
        """
        if self._value_wrapper is None:
            self._value_wrapper = MemoryFieldArray('int8')
        return self._value_wrapper

    def __len__(self):
        """
        Get length of field
        :return: int
        """
        return len(self.data)

    def get_spans(self):
        """
        :return: Span of indices as List
        """
        return ops._get_spans_for_index_string_field(self.indices[:], self.values[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, target_field)
            target_field.data[:]  # prints ['bb', 'dddd', 'eeee', 'gg']

        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_filter_to_indexed_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # ['h', 'a', 'gg', 'bb', 'fff', 'ccc', 'eeee', 'dddd', '']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_index_to_indexed_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_first(spans_to_apply, target_field)
            target_field.data[:]  # ['a', 'ccc', 'dddd', 'gg']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_last(spans_to_apply, target_field)
            target_field.data[:]  #  ['bb', 'ccc', 'fff', 'h']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_min(spans_to_apply, in_place=True)
            field.data[:]  # ['a', 'ccc', 'dddd', 'gg']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_max(spans_to_apply, in_place=True)
            field.data[:]  # ['bb', 'ccc', 'fff', 'h']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """        
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of an IndexedStringMemField.
        Returns the sorted unique elements of an IndexedStringMemField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


class FixedStringMemField(MemoryField):
    def __init__(self, session, length):
        super().__init__(session)
        # TODO: caution; we may want to consider the issues with long-lived field instances getting
        # out of sync with their stored counterparts. Maybe a revision number of the stored field
        # is required that we can check to see if we are out of date. That or just make this a
        # property and have it always look the value up
        self._length = length

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        return self

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        return FieldDataOps.fixed_string_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Returns moemory field array with values from this field
        :return: MemoryFieldArray
        """
        if self._value_wrapper is None:
            self._value_wrapper = MemoryFieldArray("S{}".format(self._length))
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(np.char.compare_chararrays(data[:-1], data[1:], "<=", False))

    def __len__(self):
        """
        Get length of field
        :return: int
        """
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field
        :return: Spans of field
        """
        return ops.get_spans_for_field(self.data[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data ['a', 'b', 'c', 'd', '', 'e', 'f', 'g', 'h']
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, target_field)
            target_field.data[:]  # prints ['b', 'd', 'e', 'g']


        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'b', 'c', 'd', '', 'e', 'f', 'g', 'h']
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # ['h', 'a', 'g', 'b', 'f', 'c', 'e', 'd', '']


        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None.
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a FixedStringMemField. 
        Returns the sorted unique elements of a FixedStringMemField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


class NumericMemField(MemoryField):
    def __init__(self, session, nformat):
        super().__init__(session)
        self._nformat = nformat

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        return self

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        return FieldDataOps.numeric_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Returns memory field array with values from this field
        :return: MemoryFieldArray
        """
        if self._value_wrapper is None:
            self._value_wrapper = MemoryFieldArray(self._nformat)
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(data[:-1] <= data[1:])

    def __len__(self):
        """
        Get length of field
        :return: int
        """
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field
        :return: Spans of field
        """
        return ops.get_spans_for_field(self.data[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, in_place=True)
            field.data[:]  # prints [22, 444, 5555, 77]


        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # [8, 1, 77, 22, 666, 333, 5555, 444, 0]

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans to (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (minimum). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def __add__(self, second):
        return FieldDataOps.numeric_add(self._session, self, second)

    def __radd__(self, first):
        return FieldDataOps.numeric_add(self._session, first, self)

    def __sub__(self, second):
        return FieldDataOps.numeric_sub(self._session, self, second)

    def __rsub__(self, first):
        return FieldDataOps.numeric_sub(self._session, first, self)

    def __mul__(self, second):
        return FieldDataOps.numeric_mul(self._session, self, second)

    def __rmul__(self, first):
        return FieldDataOps.numeric_mul(self._session, first, self)

    def __truediv__(self, second):
        return FieldDataOps.numeric_truediv(self._session, self, second)

    def __rtruediv__(self, first):
        return FieldDataOps.numeric_truediv(self._session, first, self)

    def __floordiv__(self, second):
        return FieldDataOps.numeric_floordiv(self._session, self, second)

    def __rfloordiv__(self, first):
        return FieldDataOps.numeric_floordiv(self._session, first, self)

    def __mod__(self, second):
        return FieldDataOps.numeric_mod(self._session, self, second)

    def __rmod__(self, first):
        return FieldDataOps.numeric_mod(self._session, first, self)

    def __divmod__(self, second):
        return FieldDataOps.numeric_divmod(self._session, self, second)

    def __rdivmod__(self, first):
        return FieldDataOps.numeric_divmod(self._session, first, self)

    def __and__(self, second):
        return FieldDataOps.numeric_and(self._session, self, second)

    def __rand__(self, first):
        return FieldDataOps.numeric_and(self._session, first, self)

    def __xor__(self, second):
        return FieldDataOps.numeric_xor(self._session, self, second)

    def __rxor__(self, first):
        return FieldDataOps.numeric_xor(self._session, first, self)

    def __or__(self, second):
        return FieldDataOps.numeric_or(self._session, self, second)

    def __ror__(self, first):
        return FieldDataOps.numeric_or(self._session, first, self)

    def __lt__(self, value):
        return FieldDataOps.less_than(self._session, self, value)

    def __le__(self, value):
        return FieldDataOps.less_than_equal(self._session, self, value)

    def __eq__(self, value):
        return FieldDataOps.equal(self._session, self, value)

    def __ne__(self, value):
        return FieldDataOps.not_equal(self._session, self, value)

    def __gt__(self, value):
        return FieldDataOps.greater_than(self._session, self, value)

    def __ge__(self, value):
        return FieldDataOps.greater_than_equal(self._session, self, value)

    def __invert__(self):
        return FieldDataOps.invert(self._session, self)

    def logical_not(self):
        return FieldDataOps.logical_not(self._session, self)
    
    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a NumericMemField.
        Returns the sorted unique elements of a NumericMemField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        
        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


class CategoricalMemField(MemoryField):
    def __init__(self, session, nformat, keys):
        super().__init__(session)
        self._nformat = nformat
        self._keys = keys

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        return self

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        return FieldDataOps.categorical_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Returns memory field array with values from this field
        :return: MemoryFieldArray
        """
        if self._value_wrapper is None:
            self._value_wrapper = MemoryFieldArray(self._nformat)
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(data[:-1] <= data[1:])

    def __len__(self):
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field.
        """
        return ops.get_spans_for_field(self.data[:])

    # Note: key is presented as value: str, even though the dictionary must be presented
    # as str: value
    @property
    def keys(self):
        """
        Get keys.
        """
        kv = self._keys.values()
        kn = self._keys.keys()
        keys = dict(zip(kv, kn))
        return keys

    def remap(self, key_map, new_key):
        """
        Remap the key names and key values.

        :param key_map: The mapping rule of convert the old key into the new key.
        :param new_key: The new key.
        :return: A CategoricalMemField with the new key.

        Example::

            cat_field = df.create_categorical('cat', 'int32', {"a": 1, "b": 2})
            cat_field.data.write([1,2,1,2])
            newfield = cat_field.remap([(1, 4), (2, 5)], {"a": 4, "b": 5})
            print(newfield.data[:])
        """
        # make sure all key values are included in the key_map
        for k in self._keys.values():
            if k not in [x[0] for x in key_map]:
                raise ValueError("Not all old key values are included in the mapping rule.")
        # remap the value
        values = self.data[:]
        new_values = np.zeros(len(values), values.dtype)
        for k in key_map:
            new_values = np.where(values == k[0], k[1], new_values)
        result = CategoricalMemField(self._session, self._nformat, new_key)
        result.data.write(new_values)
        return result

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data [1, 2, 3, 4, 0, 5, 6, 7, 8]
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, in_place=True)
            field.data[:]  # prints [2, 4, 5, 7]

        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable. If 'target' is set, 
            'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data [1, 2, 3, 4, 0, 5, 6, 7, 8]
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # [8, 1, 7, 2, 6, 3, 5, 4, 0]


        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def __lt__(self, value):
        return FieldDataOps.less_than(self._session, self, value)

    def __le__(self, value):
        return FieldDataOps.less_than_equal(self._session, self, value)

    def __eq__(self, value):
        return FieldDataOps.equal(self._session, self, value)

    def __ne__(self, value):
        return FieldDataOps.not_equal(self._session, self, value)

    def __gt__(self, value):
        return FieldDataOps.greater_than(self._session, self, value)

    def __ge__(self, value):
        return FieldDataOps.greater_than_equal(self._session, self, value)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field 
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a CategoricalMemField.
        Returns the sorted unique elements of a CategoricalMemField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)



class TimestampMemField(MemoryField):
    def __init__(self, session):
        super().__init__(session)

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        return self

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        return FieldDataOps.timestamp_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Returns memory field array with values from this field
        :return: MemoryFieldArray
        """
        if self._value_wrapper is None:
            self._value_wrapper = MemoryFieldArray(np.float64)
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(data[:-1] <= data[1:])

    def __len__(self):
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field.
        """
        return ops.get_spans_for_field(self.data[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, in_place=True)
            field.data[:]  # prints [22, 444, 5555, 77]


        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # [8, 1, 77, 22, 666, 333, 5555, 444, 0]


        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def __add__(self, second):
        return FieldDataOps.numeric_add(self._session, self, second)

    def __radd__(self, first):
        return FieldDataOps.numeric_add(self._session, first, self)

    def __sub__(self, second):
        return FieldDataOps.numeric_sub(self._session, self, second)

    def __rsub__(self, first):
        return FieldDataOps.numeric_sub(self._session, first, self)

    def __mul__(self, second):
        return FieldDataOps.numeric_mul(self._session, self, second)

    def __rmul__(self, first):
        return FieldDataOps.numeric_mul(self._session, first, self)

    def __truediv__(self, second):
        return FieldDataOps.numeric_truediv(self._session, self, second)

    def __rtruediv__(self, first):
        return FieldDataOps.numeric_truediv(self._session, first, self)

    def __floordiv__(self, second):
        return FieldDataOps.numeric_floordiv(self._session, self, second)

    def __rfloordiv__(self, first):
        return FieldDataOps.numeric_floordiv(self._session, first, self)

    def __mod__(self, second):
        return FieldDataOps.numeric_mod(self._session, self, second)

    def __rmod__(self, first):
        return FieldDataOps.numeric_mod(self._session, first, self)

    def __divmod__(self, second):
        return FieldDataOps.numeric_divmod(self._session, self, second)

    def __rdivmod__(self, first):
        return FieldDataOps.numeric_divmod(self._session, first, self)

    def __lt__(self, value):
        return FieldDataOps.less_than(self._session, self, value)

    def __le__(self, value):
        return FieldDataOps.less_than_equal(self._session, self, value)

    def __eq__(self, value):
        return FieldDataOps.equal(self._session, self, value)

    def __ne__(self, value):
        return FieldDataOps.not_equal(self._session, self, value)

    def __gt__(self, value):
        return FieldDataOps.greater_than(self._session, self, value)

    def __ge__(self, value):
        return FieldDataOps.greater_than_equal(self._session, self, value)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a TimestampMemField.
        Returns the sorted unique elements of a TimestampMemField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


# HDF5 field constructors
# =======================


def base_field_contructor(session, group, name, timestamp=None, chunksize=None):
    """
    Constructor are for 1)create the field (hdf5 group), 2)add basic attributes like chunksize,
    timestamp, field type, and 3)add the dataset to the field (hdf5 group) under the name 'values'
    """
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
    key_ = val.validate_and_normalize_categorical_key('key', key)
    key_values = [v for k, v in key_.items()]
    key_names = [k for k, v in key_.items()]
    DataWriter.write(field, 'key_values', key_values, len(key_values), 'int8')
    DataWriter.write(field, 'key_names', key_names, len(key_names), h5py.special_dtype(vlen=str))


def timestamp_field_constructor(session, group, name, timestamp=None, chunksize=None):
    field = base_field_contructor(session, group, name, timestamp, chunksize)
    field.attrs['fieldtype'] = 'timestamp'
    DataWriter.write(field, 'values', [], 0, 'float64')


# HDF5 fields
# ===========


class IndexedStringField(HDF5Field):
    def __init__(self, session, group, dataframe, write_enabled=False):
        super().__init__(session, group, dataframe, write_enabled=write_enabled)
        self._session = session
        self._dataframe = None
        self._data_wrapper = None
        self._index_wrapper = None
        self._value_wrapper = None

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        self._ensure_valid()
        return IndexedStringField(self._session, self._field, self._dataframe,
                                  write_enabled=True)

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        self._ensure_valid()
        return FieldDataOps.indexed_string_create_like(self, group, name, timestamp)

    @property
    def indexed(self):
        """
        Whether the field is an indexed field or not. Indexed fields store their data internally
        as index and value arrays for efficiency, as well as making it accessible through the data
        property.
        """
        self._ensure_valid()
        return True

    @property
    def data(self):
        """
        Returns indexes writable field array with values of field
        :return: WriteableIndexedFieldArray
        """
        self._ensure_valid()
        if self._data_wrapper is None:
            wrapper = \
                WriteableIndexedFieldArray if self._write_enabled else ReadOnlyIndexedFieldArray
            self._data_wrapper = wrapper(self.chunksize, self.indices, self.values)
        return self._data_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        self._ensure_valid()
        if len(self) < 2:
            return True

        indices = self.indices[:]
        values = self.values[:]
        last = values[indices[0]:indices[1]].tobytes()
        for i in range(1, len(indices) - 1):
            cur = values[indices[i]:indices[i + 1]].tobytes()
            if last > cur:
                return False
            last = cur
        return True

    @property
    def indices(self):
        """
        Get indices.
        """
        self._ensure_valid()
        if self._index_wrapper is None:
            wrapper = WriteableFieldArray if self._write_enabled else ReadOnlyFieldArray
            self._index_wrapper = wrapper(self._field, 'index')
        return self._index_wrapper

    @property
    def values(self):
        """
        Get values.
        """
        self._ensure_valid()
        if self._value_wrapper is None:
            wrapper = WriteableFieldArray if self._write_enabled else ReadOnlyFieldArray
            self._value_wrapper = wrapper(self._field, 'values')
        return self._value_wrapper

    def __len__(self):
        self._ensure_valid()
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field
        """
        self._ensure_valid()
        return ops._get_spans_for_index_string_field(self.indices[:], self.values[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, target_field)
            target_field.data[:]  # prints ['bb', 'dddd', 'eeee', 'gg']

        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_filter_to_indexed_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # ['h', 'a', 'gg', 'bb', 'fff', 'ccc', 'eeee', 'dddd', '']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_index_to_indexed_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_first(spans_to_apply, target_field)
            target_field.data[:]  # ['a', 'ccc', 'dddd', 'gg']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_last(spans_to_apply, target_field)
            target_field.data[:]  #  ['bb', 'ccc', 'fff', 'h']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_min(spans_to_apply, in_place=True)
            field.data[:]  # ['a', 'ccc', 'dddd', 'gg']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'bb', 'ccc', 'dddd', '', 'eeee', 'fff', 'gg', 'h']
            spans_to_apply = np.array([0, 2, 3, 6, 8], dtype=np.int32)

            field.apply_spans_max(spans_to_apply, in_place=True)
            field.data[:]  # ['bb', 'ccc', 'fff', 'h']

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of an IndexedStringField.
        Returns the sorted unique elements of an IndexedStringField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array
        

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


class FixedStringField(HDF5Field):
    def __init__(self, session, group, dataframe, write_enabled=False):
        super().__init__(session, group, dataframe, write_enabled=write_enabled)
        # TODO: caution; we may want to consider the issues with long-lived field instances getting
        # out of sync with their stored counterparts. Maybe a revision number of the stored field
        # is required that we can check to see if we are out of date. That or just make this a
        # property and have it always look the value up
        self._length = self._field.attrs['strlen']

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        self._ensure_valid()
        return FixedStringField(self._session, self._field, self._dataframe,
                                write_enabled=True)

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        self._ensure_valid()
        return FieldDataOps.fixed_string_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Get data.
        """
        self._ensure_valid()
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        self._ensure_valid()
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(np.char.compare_chararrays(data[:-1], data[1:], "<=", False))

    def __len__(self):
        self._ensure_valid()
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field.
        """
        self._ensure_valid()
        return ops.get_spans_for_field(self.data[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data ['a', 'b', 'c', 'd', '', 'e', 'f', 'g', 'h']
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, target_field)
            target_field.data[:]  # prints ['b', 'd', 'e', 'g']


        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data ['a', 'b', 'c', 'd', '', 'e', 'f', 'g', 'h']
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # ['h', 'a', 'g', 'b', 'f', 'c', 'e', 'd', '']


        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a FixedStringField.
        Returns the sorted unique elements of a FixedStringField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


class NumericField(HDF5Field):
    def __init__(self, session, group, dataframe, write_enabled=False):
        super().__init__(session, group, dataframe, write_enabled=write_enabled)
        self._nformat = self._field.attrs['nformat']

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        self._ensure_valid()
        return NumericField(self._session, self._field, None, write_enabled=True)

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        self._ensure_valid()
        return FieldDataOps.numeric_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Get data.
        """
        self._ensure_valid()
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        self._ensure_valid()
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(data[:-1] <= data[1:])

    def __len__(self):
        self._ensure_valid()
        return len(self.data)

    def astype(self, dtype: str, casting='unsafe'):
        """
        Convert the field data type to dtype parameter given.

        :param dtype: The new datatype, given as a str object. The dtype must be a subtype of np.number, e.g. int, float, etc.
        :param casting: Similar to the casting parameter in numpy ndarray.astype, can be 'no’, ‘equiv’, ‘safe’, ‘same_kind’, or ‘unsafe’.
        :return: The field with new datatype.
        """
        if not np.issubdtype(dtype, np.number):
            raise ValueError("The dtype to convert must be a subtype of np.number, but type {} given.".format(dtype))
        else:
            content = np.array(self.data[:]).astype(dtype, casting=casting)
            name = self.name
            del self.dataframe[name]
            fld = self.dataframe.create_numeric(name, str(dtype))
            fld.data.write(content)
            return fld

    def get_spans(self):
        """
        Get spans of field.
        """
        self._ensure_valid()
        return ops.get_spans_for_field(self.data[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, in_place=True)
            field.data[:]  # prints [22, 444, 5555, 77]


        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # [8, 1, 77, 22, 666, 333, 5555, 444, 0]


        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def __add__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_add(self._session, self, second)

    def __radd__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_add(self._session, first, self)

    def __sub__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_sub(self._session, self, second)

    def __rsub__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_sub(self._session, first, self)

    def __mul__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_mul(self._session, self, second)

    def __rmul__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_mul(self._session, first, self)

    def __truediv__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_truediv(self._session, self, second)

    def __rtruediv__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_truediv(self._session, first, self)

    def __floordiv__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_floordiv(self._session, self, second)

    def __rfloordiv__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_floordiv(self._session, first, self)

    def __mod__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_mod(self._session, self, second)

    def __rmod__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_mod(self._session, first, self)

    def __divmod__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_divmod(self._session, self, second)

    def __rdivmod__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_divmod(self._session, first, self)

    def __and__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_and(self._session, self, second)

    def __rand__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_and(self._session, first, self)

    def __xor__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_xor(self._session, self, second)

    def __rxor__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_xor(self._session, first, self)

    def __or__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_or(self._session, self, second)

    def __ror__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_or(self._session, first, self)

    def __lt__(self, value):
        self._ensure_valid()
        return FieldDataOps.less_than(self._session, self, value)

    def __le__(self, value):
        self._ensure_valid()
        return FieldDataOps.less_than_equal(self._session, self, value)

    def __eq__(self, value):
        self._ensure_valid()
        return FieldDataOps.equal(self._session, self, value)

    def __ne__(self, value):
        self._ensure_valid()
        return FieldDataOps.not_equal(self._session, self, value)

    def __gt__(self, value):
        self._ensure_valid()
        return FieldDataOps.greater_than(self._session, self, value)

    def __ge__(self, value):
        self._ensure_valid()
        return FieldDataOps.greater_than_equal(self._session, self, value)

    def __invert__(self):
        self._ensure_valid()
        return FieldDataOps.invert(self._session, self)

    def logical_not(self):
        self._ensure_valid()
        return FieldDataOps.logical_not(self._session, self)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a NumericField.
        Returns the sorted unique elements of a NumericField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


class CategoricalField(HDF5Field):
    def __init__(self, session, group, dataframe, write_enabled=False):
        super().__init__(session, group, dataframe, write_enabled=write_enabled)
        self._nformat = self._field.attrs['nformat'] if 'nformat' in self._field.attrs else 'int8'

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        self._ensure_valid()
        return CategoricalField(self._session, self._field, self._dataframe,
                                write_enabled=True)

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        self._ensure_valid()
        return FieldDataOps.categorical_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Get data.
        """
        self._ensure_valid()
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        self._ensure_valid()
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(data[:-1] <= data[1:])

    def __len__(self):
        self._ensure_valid()
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field.
        """
        self._ensure_valid()
        return ops.get_spans_for_field(self.data[:])

    @property
    def nformat(self):
        """
        Get numeric format.
        """
        self._ensure_valid()
        return self._nformat

    # Note: key is presented as value: str, even though the dictionary must be presented
    # as str: value
    @property
    def keys(self):
        """
        Get keys.
        """
        self._ensure_valid()
        if isinstance(self._field['key_values'][0], str):  # convert into bytearray to keep up with linux
            kv = [bytes(i, 'utf-8') for i in self._field['key_values']]
        else:
            kv = self._field['key_values']
        if isinstance(self._field['key_names'][0], str):
            kn = [bytes(i, 'utf-8') for i in self._field['key_names']]
        else:
            kn = self._field['key_names']
        keys = dict(zip(kv, kn))
        return keys

    def remap(self, key_map, new_key):
        """
        Remap the key names and key values.

        :param key_map: The mapping rule of convert the old key into the new key.
        :param new_key: The new key.
        :return: A CategoricalMemField with the new key.

        Example::

            cat_field = df.create_categorical('cat', 'int32', {"a": 1, "b": 2})
            cat_field.data.write([1,2,1,2])
            newfield = cat_field.remap([(1, 4), (2, 5)], {"a": 4, "b": 5})
            print(newfield.data[:])
        """
        self._ensure_valid()
        # make sure all key values are included in the key_map
        if isinstance(self._field['key_values'][0], str):  # convert into bytearray to keep up with linux
            kv = [bytes(i, 'utf-8') for i in self._field['key_values']]
        else:
            kv = self._field['key_values']
        for k in kv:
            if k not in [x[0] for x in key_map]:
                raise ValueError("Not all old key values are included in the mapping rule.")
        #remap the value
        values = self.data[:]
        new_values = np.zeros(len(values), values.dtype)
        for k in key_map:
            new_values = np.where(values == k[0], k[1], new_values)
        result = CategoricalMemField(self._session, self._nformat, new_key)
        result.data.write(new_values)
        return result

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data [1, 2, 3, 4, 0, 5, 6, 7, 8]
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, in_place=True)
            field.data[:]  # prints [2, 4, 5, 7]


        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data [1, 2, 3, 4, 0, 5, 6, 7, 8]
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # [8, 1, 7, 2, 6, 3, 5, 4, 0]


        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def __lt__(self, value):
        self._ensure_valid()
        return FieldDataOps.less_than(self._session, self, value)

    def __le__(self, value):
        self._ensure_valid()
        return FieldDataOps.less_than_equal(self._session, self, value)

    def __eq__(self, value):
        self._ensure_valid()
        return FieldDataOps.equal(self._session, self, value)

    def __ne__(self, value):
        self._ensure_valid()
        return FieldDataOps.not_equal(self._session, self, value)

    def __gt__(self, value):
        self._ensure_valid()
        return FieldDataOps.greater_than(self._session, self, value)

    def __ge__(self, value):
        self._ensure_valid()
        return FieldDataOps.greater_than_equal(self._session, self, value)
    
    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a CategoricalField.
        Returns the sorted unique elements of a CategoricalField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array
        
        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)


class TimestampField(HDF5Field):
    def __init__(self, session, group, dataframe, write_enabled=False):
        super().__init__(session, group, dataframe, write_enabled=write_enabled)

    def writeable(self):
        """
        Indicates whether this field permits write operations. By default, dataframe fields
        are read-only in order to protect accidental writes to datasets
        """
        self._ensure_valid()
        return TimestampField(self._session, self._field, self._dataframe,
                              write_enabled=True)

    def create_like(self, group=None, name=None, timestamp=None):
        """
        Creates a empty field of same type as this.

        :param group: h5group
        :param name: Name of new the field
        :param timestamp: optional - If set, the timestamp that should be given to the new field.
        :return: Indexed string field
        """
        self._ensure_valid()
        return FieldDataOps.timestamp_field_create_like(self, group, name, timestamp)

    @property
    def data(self):
        """
        Get data.
        """
        self._ensure_valid()
        if self._value_wrapper is None:
            if self._write_enabled:
                self._value_wrapper = WriteableFieldArray(self._field, 'values')
            else:
                self._value_wrapper = ReadOnlyFieldArray(self._field, 'values')
        return self._value_wrapper

    def is_sorted(self):
        """
        Returns if data in field is sorted
        :return: bool
        """
        self._ensure_valid()
        if len(self) < 2:
            return True
        data = self.data[:]
        return np.all(data[:-1] <= data[1:])

    def __len__(self):
        self._ensure_valid()
        return len(self.data)

    def get_spans(self):
        """
        Get spans of field.
        """
        self._ensure_valid()
        return ops.get_spans_for_field(self.data[:])

    def apply_filter(self, filter_to_apply, target=None, in_place=False):
        """
        Apply a boolean filter to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the filtered data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            filter_to_apply = np.array([0, 2, 0, 1, 0, 1, 0, 1, 0])

            field.apply_filter(filter_to_apply, in_place=True)
            field.data[:]  # prints [22, 444, 5555, 77]


        :param filter_to_apply: a Field or numpy array that contains the boolean filter data
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The filtered field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_filter_to_field(self, filter_to_apply, target, in_place)

    def apply_index(self, index_to_apply, target=None, in_place=False):
        """
        Apply an index to this field. This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        Example::

            field = ... # field contains data [1, 22, 333, 444, 0, 5555, 666, 77, 8]
            index_to_apply = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4], dtype=np.int32)

            field.apply_index(index_to_apply, target_field)
            target_field.data[:]  # [8, 1, 77, 22, 666, 333, 5555, 444, 0]


        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The reindexed field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_index_to_field(self, index_to_apply, target, in_place)

    def apply_spans_first(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (first). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_first(self, spans_to_apply, target, in_place)

    def apply_spans_last(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (last). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_last(self, spans_to_apply, target, in_place)

    def apply_spans_min(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (min). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_min(self, spans_to_apply, target, in_place)

    def apply_spans_max(self, spans_to_apply, target=None, in_place=False):
        """
        Apply spans (max). This operation doesn't modify the field on which it
        is called unless 'in_place is set to true'. The user can specify a 'target' field that
        the reindexed data is written to.

        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: if set, this is the field that is written to. This field must be writable.
            If 'target' is set, 'in_place' must be False.
        :param in_place: if True, perform the operation destructively on this field. This field
            must be writable. If 'in_place' is True, 'target' must be None
        :return: The respanned field. This is a new field instance unless 'target' is set, in which
            case it is the target field, or unless 'in_place' is True, in which case it is this field.
        """
        self._ensure_valid()
        return FieldDataOps.apply_spans_max(self, spans_to_apply, target, in_place)

    def __add__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_add(self._session, self, second)

    def __radd__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_add(self._session, first, self)

    def __sub__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_sub(self._session, self, second)

    def __rsub__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_sub(self._session, first, self)

    def __mul__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_mul(self._session, self, second)

    def __rmul__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_mul(self._session, first, self)

    def __truediv__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_truediv(self._session, self, second)

    def __rtruediv__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_truediv(self._session, first, self)

    def __floordiv__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_floordiv(self._session, self, second)

    def __rfloordiv__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_floordiv(self._session, first, self)

    def __mod__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_mod(self._session, self, second)

    def __rmod__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_mod(self._session, first, self)

    def __divmod__(self, second):
        self._ensure_valid()
        return FieldDataOps.numeric_divmod(self._session, self, second)

    def __rdivmod__(self, first):
        self._ensure_valid()
        return FieldDataOps.numeric_divmod(self._session, first, self)

    def __lt__(self, value):
        self._ensure_valid()
        return FieldDataOps.less_than(self._session, self, value)

    def __le__(self, value):
        self._ensure_valid()
        return FieldDataOps.less_than_equal(self._session, self, value)

    def __eq__(self, value):
        self._ensure_valid()
        return FieldDataOps.equal(self._session, self, value)

    def __ne__(self, value):
        self._ensure_valid()
        return FieldDataOps.not_equal(self._session, self, value)

    def __gt__(self, value):
        self._ensure_valid()
        return FieldDataOps.greater_than(self._session, self, value)

    def __ge__(self, value):
        self._ensure_valid()
        return FieldDataOps.greater_than_equal(self._session, self, value)

    def isin(self, test_elements:Union[list, set, np.ndarray]):
        """
        Returns a boolean array of the same length as field \
        that is True where an element of field is in test_elements and False otherwise.

        :param test_elements: The values against which to test each value of field.
        :return: a boolean array of the same length as field
        """
        return FieldDataOps.apply_isin(self, test_elements)

    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        """
        Find the unique elements of a TimestampField.
        Returns the sorted unique elements of a TimestampField. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array

        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        return FieldDataOps.apply_unique(self, return_index, return_inverse, return_counts)
      

# Operation implementations
# =========================


def as_field(data, key=None):
    if np.issubdtype(data.dtype, np.number):
        if key is None:
            r = NumericMemField(None, data.dtype)
            r.data.write(data)
            return r
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def argsort(field: Field,
            dtype: str = None):
    supported_dtypes = ('int32', 'int64', 'uint32')
    if dtype not in supported_dtypes:
        raise ValueError("If set, 'dtype' must be one of {}".format(supported_dtypes))
    indices = np.argsort(field.data[:])

    f = NumericMemField(None, dtype_to_str(indices.dtype) if dtype is None else dtype)
    f.data.write(indices)
    return f


def dtype_to_str(dtype):
    """
    Returns string name for given data type
    :param dtype: given data type
    :return: str
    """
    if isinstance(dtype, str):
        return dtype

    if dtype == bool:
        return 'bool'
    elif dtype == np.int8:
        return 'int8'
    elif dtype == np.int16:
        return 'int16'
    elif dtype == np.int32:
        return 'int32'
    elif dtype == np.int64:
        return 'int64'
    elif dtype == np.uint8:
        return 'uint8'
    elif dtype == np.uint16:
        return 'uint16'
    elif dtype == np.uint32:
        return 'uint32'
    elif dtype == np.uint64:
        return 'uint64'
    elif dtype == np.float32:
        return 'float32'
    elif dtype == np.float64:
        return 'float64'

    raise ValueError("Unsupported dtype '{}'".format(dtype))


class FieldDataOps:

    @staticmethod
    def _binary_op(session, first, second, function):
        if isinstance(first, Field):
            first_data = first.data[:]
        else:
            first_data = first

        if isinstance(second, Field):
            second_data = second.data[:]
        else:
            second_data = second

        r = function(first_data, second_data)
        f = NumericMemField(session, dtype_to_str(r.dtype))
        f.data.write(r)
        return f

    @staticmethod
    def _unary_op(session, first, function):
        if isinstance(first, Field):
            first_data = first.data[:]
        else:
            first_data = first

        r = function(first_data)
        f = NumericMemField(session, dtype_to_str(r.dtype))
        f.data.write(r)
        return f

    @classmethod
    def numeric_add(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.add)

    @classmethod
    def numeric_sub(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.sub)

    @classmethod
    def numeric_mul(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.mul)

    @classmethod
    def numeric_truediv(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.truediv)

    @classmethod
    def numeric_floordiv(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.floordiv)

    @classmethod
    def numeric_mod(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.mod)

    @classmethod
    def numeric_divmod(cls, session, first, second):
        if isinstance(first, Field):
            first_data = first.data[:]
        else:
            first_data = first

        if isinstance(second, Field):
            second_data = second.data[:]
        else:
            second_data = second

        r1, r2 = np.divmod(first_data, second_data)
        f1 = NumericMemField(session, dtype_to_str(r1.dtype))
        f1.data.write(r1)
        f2 = NumericMemField(session, dtype_to_str(r2.dtype))
        f2.data.write(r2)
        return f1, f2

    @classmethod
    def numeric_and(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.and_)

    @classmethod
    def numeric_xor(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.xor)

    @classmethod
    def numeric_or(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.or_)

    @classmethod
    def invert(cls, session, first):
        return cls._unary_op(session, first, operator.invert)

    @classmethod
    def logical_not(cls, session, first):
        def function_logical_not(first):
            return np.logical_not(first)

        return cls._unary_op(session, first, function_logical_not)

    @classmethod
    def less_than(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.lt)

    @classmethod
    def less_than_equal(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.le)

    @classmethod
    def equal(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.eq)

    @classmethod
    def not_equal(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.ne)

    @classmethod
    def greater_than(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.gt)

    @classmethod
    def greater_than_equal(cls, session, first, second):
        return cls._binary_op(session, first, second, operator.ge)

    @staticmethod
    def apply_filter_to_indexed_field(source, filter_to_apply, target=None, in_place=False):
        if in_place is True and target is not None:
            raise ValueError("if 'in_place is True, 'target' must be None")
        
        filter_to_apply_ = val.validate_filter(filter_to_apply)


        dest_indices, dest_values = \
            ops.apply_filter_to_index_values(filter_to_apply_,
                                             source.indices[:], source.values[:])

        if in_place:
            if not source._write_enabled:
                raise ValueError("This field is marked read-only. Call writeable() on it before "
                                 "performing in-place filtering")
            source.indices.clear()
            source.indices.write(dest_indices)
            source.values.clear()
            source.values.write(dest_values)
            return source

        if target is not None:
            if len(target.indices) == len(dest_indices):
                target.indices[:] = dest_indices
            else:
                target.indices.clear()
                target.indices.write(dest_indices)
            if len(target.values) == len(dest_values):
                target.values[:] = dest_values
            else:
                target.values.clear()
                target.values.write(dest_values)
            return target
        else:
            mem_field = IndexedStringMemField(source._session, source.chunksize)
            mem_field.indices.write(dest_indices)
            mem_field.values.write(dest_values)
            return mem_field

    @staticmethod
    def apply_index_to_indexed_field(source, index_to_apply, target=None, in_place=False):
        if in_place is True and target is not None:
            raise ValueError("if 'in_place is True, 'target' must be None")

        index_to_apply_ = val.array_from_field_or_lower('index_to_apply', index_to_apply)

        dest_indices, dest_values = \
            ops.apply_indices_to_index_values(index_to_apply_,
                                              source.indices[:], source.values[:])

        if in_place:
            if not source._write_enabled:
                raise ValueError("This field is marked read-only. Call writeable() on it before "
                                 "performing in-place filtering")
            source.indices.clear()
            source.indices.write(dest_indices)
            source.values.clear()
            source.values.write(dest_values)
            return source

        if target is not None:
            if len(target.indices) == len(dest_indices):
                target.indices[:] = dest_indices
            else:
                target.indices.clear()
                target.indices.write(dest_indices)
            if len(target.values) == len(dest_values):
                target.values[:] = dest_values
            else:
                target.values.clear()
                target.values.write(dest_values)
            return target
        else:
            mem_field = IndexedStringMemField(source._session, source.chunksize)
            mem_field.indices.write(dest_indices)
            mem_field.values.write(dest_values)
            return mem_field

    @staticmethod
    def apply_filter_to_field(source, filter_to_apply, target=None, in_place=False):
        """
        Apply filter to field, either in place (itself) or a target (new) field

        :param source: Field
        :param filter_to_apply: a Field or numpy array that contains the indices to filter
        :param target: Optional, Field, if set create a field like as target
        :param in_place: optional, bool, if set changes data in field
        :return: Field with filter applied
        """
        if in_place is True and target is not None:
            raise ValueError("if 'in_place is True, 'target' must be None")

        filter_to_apply_ = val.validate_filter(filter_to_apply)

        dest_data = source.data[:][filter_to_apply_]

        if in_place:
            if not source._write_enabled:
                raise ValueError("This field is marked read-only. Call writeable() on it before "
                                 "performing in-place filtering")
            source.data.clear()
            source.data.write(dest_data)
            return source

        if target is not None:
            if len(target.data) == len(dest_data):
                target.data[:] = dest_data
            else:
                target.data.clear()
                target.data.write(dest_data)
            return target
        else:
            mem_field = source.create_like()
            mem_field.data.write(dest_data)
            return mem_field

    @staticmethod
    def apply_index_to_field(source, index_to_apply, target=None, in_place=False):
        """
        Apply index to field, either in place (itself) or a target (new) field

        :param source: Field
        :param index_to_apply: a Field or numpy array that contains the indices
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field with index
        """
        if in_place is True and target is not None:
            raise ValueError("if 'in_place is True, 'target' must be None")

        index_to_apply_ = val.array_from_field_or_lower('index_to_apply', index_to_apply)

        dest_data = source.data[:][index_to_apply_]

        if in_place:
            if not source._write_enabled:
                raise ValueError("This field is marked read-only. Call writeable() on it before "
                                 "performing in-place filtering")
            source.data.clear()
            source.data.write(dest_data)
            return source

        if target is not None:
            if len(target.data) == len(dest_data):
                target.data[:] = dest_data
            else:
                target.data.clear()
                target.data.write(dest_data)
            return target
        else:
            mem_field = source.create_like()
            mem_field.data.write(dest_data)
            return mem_field

    @staticmethod
    def _apply_spans_src(source: Field,
                         predicate: Callable[[np.ndarray, np.ndarray, np.ndarray], Field],
                         spans: Union[Field, np.ndarray],
                         target: Optional[Field] = None,
                         in_place: bool = False) -> Field:
        """
        Apply spans, either in place (itself) or a target (new) field

        :param source: Field
        :param predicate: Callable[[np.ndarray, np.ndarray, np.ndarray], Field]
        :param spans: Field or ndarray
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field
        """
        if in_place is True and target is not None:
            raise ValueError("if 'in_place is True, 'target' must be None")

        spans_ = val.array_from_field_or_lower('spans', spans)
        results = predicate(spans_, source.data[:])

        if in_place is True:
            if not source._write_enabled:
                raise ValueError("This field is marked read-only. Call writeable() on it before "
                                 "performing in-place apply_span methods")
            source.data.clear()
            source.data.write(results)
            return source

        if target is None:
            result_field = source.create_like()
            result_field.data.write(results)
            return result_field
        else:
            target.data.clear()
            target.data.write(results)
            return target

    @staticmethod
    def _apply_spans_indexed_src(source: Field,
                                 predicate: Callable[[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray], Field],
                                 spans: Union[Field, np.ndarray],
                                 target: Optional[Field] = None,
                                 in_place: bool = False) -> Field:
        """
        Apply spans, either in place (itself) or a target (new) field

        :param source: Field
        :param predicate: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Field] 
        :param spans: Field or ndarray
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field
        """
        if in_place is True and target is not None:
            raise ValueError("if 'in_place is True, 'target' must be None")

        spans_ = val.array_from_field_or_lower('spans', spans)

        # step 1: get the indices through the index predicate
        results = predicate(spans_, source.indices[:], source.values[:])

        # step 2: run apply_index on the source
        return FieldDataOps.apply_index_to_indexed_field(source, results, target, in_place)

    @staticmethod
    def _apply_spans_indexed_no_src(source: Field,
                                    predicate: Callable[[np.ndarray, np.ndarray], Field],
                                    spans: Union[Field, np.ndarray],
                                    target: Optional[Field] = None,
                                    in_place: bool = False) -> Field:
        """
        Apply spans, either in place (itself) or a target (new) field

        :param source: Field
        :param predicate: Callable[[np.ndarray, np.ndarray], Field] 
        :param spans: Field or ndarray
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field
        """
        if in_place is True and target is not None:
            raise ValueError("if 'in_place is True, 'target' must be None")

        spans_ = val.array_from_field_or_lower('spans', spans)

        # step 1: get the indices through the index predicate
        results = predicate(spans_)

        # step 2: run apply_index on the source
        return FieldDataOps.apply_index_to_indexed_field(source, results, target, in_place)

    @staticmethod
    def apply_spans_first(source: Field,
                          spans: Union[Field, np.ndarray],
                          target: Optional[Field] = None,
                          in_place: bool = None) -> Field:
        """
        Apply spans first, either in place (itself) or a target (new) field

        :param source: Field
        :param spans: Field or ndarray
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field
        """
        spans_ = val.array_from_field_or_lower('spans', spans)
        if np.any(spans_[:-1] == spans_[1:]):
            raise ValueError("cannot perform 'first' on spans with empty entries")

        if source.indexed:
            return FieldDataOps._apply_spans_indexed_no_src(source,
                                                            ops.apply_spans_index_of_first,
                                                            spans_, target, in_place)
        else:
            return FieldDataOps._apply_spans_src(source, ops.apply_spans_first, spans_,
                                                 target, in_place)

    @staticmethod
    def apply_spans_last(source: Field,
                         spans: Union[Field, np.ndarray],
                         target: Optional[Field] = None,
                         in_place: bool = None) -> Field:
        """
        Apply spans last, either in place (itself) or a target (new) field

        :param source: Field
        :param spans: Field or ndarray
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field
        """
        spans_ = val.array_from_field_or_lower('spans', spans)
        if np.any(spans_[:-1] == spans_[1:]):
            raise ValueError("cannot perform 'first' on spans with empty entries")

        if source.indexed:
            return FieldDataOps._apply_spans_indexed_no_src(source,
                                                            ops.apply_spans_index_of_last,
                                                            spans_, target, in_place)
        else:
            return FieldDataOps._apply_spans_src(source, ops.apply_spans_last, spans_,
                                                 target, in_place)

    @staticmethod
    def apply_spans_min(source: Field,
                        spans: Union[Field, np.ndarray],
                        target: Optional[Field] = None,
                        in_place: bool = None) -> Field:
        """
        Apply spans min, either in place (itself) or a target (new) field

        :param source: Field
        :param spans: Field or ndarray
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field
        """
        spans_ = val.array_from_field_or_lower('spans', spans)
        if np.any(spans_[:-1] == spans_[1:]):
            raise ValueError("cannot perform 'first' on spans with empty entries")

        if source.indexed:
            return FieldDataOps._apply_spans_indexed_src(source,
                                                         ops.apply_spans_index_of_min_indexed,
                                                         spans_, target, in_place)
        else:
            return FieldDataOps._apply_spans_src(source, ops.apply_spans_min, spans_,
                                                 target, in_place)

    @staticmethod
    def apply_spans_max(source: Field,
                        spans: Union[Field, np.ndarray],
                        target: Optional[Field] = None,
                        in_place: bool = None) -> Field:
        """
        Apply spans max, either in place (itself) or a target (new) field

        :param source: Field
        :param spans: Field or ndarray
        :param target: Optional, Field, if set create a field like as target
        :param in_place: bool, if set changes data in field
        :return: Field
        """
        spans_ = val.array_from_field_or_lower('spans', spans)
        if np.any(spans_[:-1] == spans_[1:]):
            raise ValueError("cannot perform 'first' on spans with empty entries")

        if source.indexed:
            return FieldDataOps._apply_spans_indexed_src(source,
                                                         ops.apply_spans_index_of_max_indexed,
                                                         spans_, target, in_place)
        else:
            return FieldDataOps._apply_spans_src(source, ops.apply_spans_max, spans_,
                                                 target, in_place)

    @staticmethod
    def indexed_string_create_like(source, group, name, timestamp):
        """
        :param group: h5py group
        :param name: Name of indexed string field
        :param timestamp: timestamp, see: https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#timestampfield 
        :return: Indexed string field 
        """
        if group is None and name is not None:
            raise ValueError("if 'group' is None, 'name' must also be 'None'")

        ts = source.timestamp if timestamp is None else timestamp

        if group is None:
            return IndexedStringMemField(source._session, source.chunksize)

        if isinstance(group, h5py.Group):
            indexed_string_field_constructor(source._session, group, name, ts, source.chunksize)
            return IndexedStringField(source._session, group[name], None, write_enabled=True)
        else:
            return group.create_indexed_string(name, ts, source.chunksize)

    @staticmethod
    def fixed_string_field_create_like(source, group, name, timestamp):
        """
        :param group: h5py group
        :param name: str
        :param timestamp: TimestampField, see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#timestampfield 
        :return: FixedStringField or FixedStringMemField
        """
        if group is None and name is not None:
            raise ValueError("if 'group' is None, 'name' must also be 'None'")

        ts = source.timestamp if timestamp is None else timestamp
        length = source._length

        if group is None:
            return FixedStringMemField(source._session, length)

        if isinstance(group, h5py.Group):
            fixed_string_field_constructor(source._session, group, name, length, ts, source.chunksize)
            return FixedStringField(source._session, group[name], None, write_enabled=True)
        else:
            return group.create_fixed_string(name, length, ts)

    @staticmethod
    def numeric_field_create_like(source, group, name, timestamp):
        """
        :param group: h5py group
        :param name: str
        :param timestamp: TimestampField, see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#timestampfield 
        :return: NumericField or NumericMemField
        """
        if group is None and name is not None:
            raise ValueError("if 'group' is None, 'name' must also be 'None'")

        ts = source.timestamp if timestamp is None else timestamp
        nformat = source._nformat

        if group is None:
            return NumericMemField(source._session, nformat)

        if isinstance(group, h5py.Group):
            numeric_field_constructor(source._session, group, name, nformat, ts, source.chunksize)
            return NumericField(source._session, group[name], None, write_enabled=True)
        else:
            return group.create_numeric(name, nformat, ts)

    @staticmethod
    def categorical_field_create_like(source, group, name, timestamp):
        """
        :param group: h5py group
        :param name: str
        :param timestamp: timestamp
        :return: CategoricalField or CategoricalMemField
        """
        if group is None and name is not None:
            raise ValueError("if 'group' is None, 'name' must also be 'None'")

        ts = source.timestamp if timestamp is None else timestamp
        nformat = source._nformat
        keys = source.keys
        # TODO: we have to flip the keys until we fix https://github.com/KCL-BMEIS/ExeTera/issues/150
        keys = {v: k for k, v in keys.items()}

        if group is None:
            return CategoricalMemField(source._session, nformat, keys)

        if isinstance(group, h5py.Group):
            categorical_field_constructor(source._session, group, name, nformat, keys,
                                          ts, source.chunksize)
            return CategoricalField(source._session, group[name], None, write_enabled=True)
        else:
            return group.create_categorical(name, nformat, keys, ts)

    @staticmethod
    def timestamp_field_create_like(source, group, name, timestamp):
        """
        :param group: h5py group
        :param name: str
        :param timestamp: TimestampField, see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#timestampfield 
        :return: TimestampField, see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#timestampfield
        """
        if group is None and name is not None:
            raise ValueError("if 'group' is None, 'name' must also be 'None'")

        ts = source.timestamp if timestamp is None else timestamp

        if group is None:
            return TimestampMemField(source._session)

        if isinstance(group, h5py.Group):
            timestamp_field_constructor(source._session, group, name, ts, source.chunksize)
            return TimestampField(source._session, group[name], None, write_enabled=True)
        else:
            return group.create_timestamp(name, ts)


    @staticmethod
    def apply_isin(source: Field, test_elements: Union[list, set, np.ndarray]):
        """
        Apply isin operation for elements on Field

        :param source: Field
        :param test_elements: list, set or ndarray
        :return: bool
        """
        if isinstance(test_elements, set):
            test_elements = list(test_elements)

        if source.indexed:
            return ops.isin_for_indexed_string_field(test_elements, source.indices[:], source.values[:])
        else:
            return np.isin(source.data[:], test_elements)

    @staticmethod
    def apply_unique(src: Field, return_index=False, return_inverse=False, return_counts=False) -> np.ndarray:
        """
        Find unique elements in field. 
        Returns the sorted unique elements of a field. 
        There are three optional outputs in addition to the unique elements:
        (1) the indices of the input array that give the unique values
        (2) the indices of the unique array that reconstruct the input array
        (3) the number of times each unique value comes up in the input array
        
        :param src: Field
        :param return_index: boolean, if true returns index of unique elements
        :param return_inverse: boolean, if true returns result in reverse
        :param return_counts: boolean, if true returns counts of unique elements
        :return: numpy array
        """
        if src.indexed:
            return ops.unique_for_indexed_string(src.indices[:], src.values[:], return_index, return_inverse, return_counts) 
        else:
            return np.unique(src.data[:], return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
