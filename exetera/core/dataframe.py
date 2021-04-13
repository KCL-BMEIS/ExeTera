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

from exetera.core.abstract_types import AbstractSession, Dataset, DataFrame
from exetera.core import fields as fld
from exetera.core import dataset as dst
import h5py


class HDF5DataFrame(DataFrame):
    """
    DataFrame that utilising HDF5 file as storage.
    """
    def __init__(self,
                 dataset: Dataset,
                 name: str,
                 h5group: h5py.Group,
                 dataframe: dict = None):
        """
        Create a Dataframe object, user should always call from dataset.create_dataframe.

        :param name: name of the dataframe, or the group name in HDF5
        :param dataset: a dataset object, where this dataframe belongs to
        :param h5group: acquire data from h5group object directly, the h5group needs to have a
                        h5group<-group-dataset structure, the group has a 'fieldtype' attribute
                         and the dataset is named 'values'.
        :param dataframe: optional - replicate data from another dictionary
        """

        self.name = name
        self._columns = dict()
        self._dataset = dataset
        self._h5group = h5group

        if dataframe is not None:
            if isinstance(dataframe, dict):
                for k, v in dataframe.items():
                    if not isinstance(k, str) or not isinstance(v, fld.Field):
                        raise ValueError("If dataframe parameter is set, "
                                         "must be a dictionary mapping strings to fields")
                self._columns = dataframe
        for subg in h5group.keys():
            self._columns[subg] = dataset.session.get(h5group[subg])

    @property
    def columns(self):
        return dict(self._columns)

    @property
    def dataset(self):
        return self._dataset

    @property
    def h5group(self):
        return self._h5group

    def add(self, field, name=None):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("The name must be a str object.")
            else:
                self._columns[name] = field
        # note the name has '/' for hdf5 object
        self._columns[field.name[field.name.index('/', 1)+1:]] = field

    def create_group(self, name):
        """
        Create a group object in HDF5 file for field to use.

        :param name: the name of the group and field
        :return: a hdf5 group object
        """
        self._h5group.create_group(name)
        return self._h5group[name]

    def create_numeric(self, name, nformat, timestamp=None, chunksize=None):
        fld.numeric_field_constructor(self._dataset.session, self, name, nformat, timestamp, chunksize)
        field = fld.NumericField(self._dataset.session, self._h5group[name],
                                 write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_indexed_string(self, name, timestamp=None, chunksize=None):
        fld.indexed_string_field_constructor(self._dataset.session, self, name, timestamp, chunksize)
        field = fld.IndexedStringField(self._dataset.session, self._h5group[name],
                                       write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_fixed_string(self, name, length, timestamp=None, chunksize=None):
        fld.fixed_string_field_constructor(self._dataset.session, self, name, length, timestamp, chunksize)
        field = fld.FixedStringField(self._dataset.session, self._h5group[name],
                                     write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_categorical(self, name, nformat, key, timestamp=None, chunksize=None):
        fld.categorical_field_constructor(self._dataset.session, self, name, nformat, key,
                                          timestamp, chunksize)
        field = fld.CategoricalField(self._dataset.session, self._h5group[name],
                                     write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_timestamp(self, name, timestamp=None, chunksize=None):
        fld.timestamp_field_constructor(self._dataset.session, self, name, timestamp, chunksize)
        field = fld.TimestampField(self._dataset.session, self._h5group[name],
                                   write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def __contains__(self, name):
        """
        check if dataframe contains a field, by the field name
        name: the name of the field to check,return a bool
        """
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        else:
            return self._columns.__contains__(name)

    def contains_field(self, field):
        """
        check if dataframe contains a field by the field object
        field: the filed object to check, return a tuple(bool,str). The str is the name stored in dataframe.
        """
        if not isinstance(field, fld.Field):
            raise TypeError("The field must be a Field object")
        else:
            for v in self._columns.values():
                if id(field) == id(v):
                    return True
            return False

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        elif not self.__contains__(name):
            raise ValueError("Can not find the name from this dataframe.")
        else:
            return self._columns[name]

    def get_field(self, name):
        return self.__getitem__(name)

    def get_name(self, field):
        """
        Get the name of the field in dataframe.
        """
        if not isinstance(field, fld.Field):
            raise TypeError("The field argument must be a Field object.")
        for name, v in self._columns.items():
            if id(field) == id(v):
                return name
        return None

    def __setitem__(self, name, field):
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        elif not isinstance(field, fld.Field):
            raise TypeError("The field must be a Field object.")
        else:
            self._columns[name] = field
            return True

    def __delitem__(self, name):
        if not self.__contains__(name=name):
            raise ValueError("This dataframe does not contain the name to delete.")
        else:
            del self._columns[name]
            return True

    def delete_field(self, field):
        """
        Remove field from dataframe by field
        """
        name = self.get_name(field)
        if name is None:
            raise ValueError("This dataframe does not contain the field to delete.")
        else:
            self.__delitem__(name)

    def keys(self):
        return self._columns.keys()

    def values(self):
        return self._columns.values()

    def items(self):
        return self._columns.items()

    def __iter__(self):
        return iter(self._columns)

    def __next__(self):
        return next(self._columns)

    def __len__(self):
        return len(self._columns)

    def get_spans(self):
        """
        Return the name and spans of each field as a dictionary.
        """
        spans = {}
        for name, field in self._columns.items():
            spans[name] = field.get_spans()
        return spans

    def apply_filter(self, filter_to_apply, ddf=None):
        """
        Apply the filter to all the fields in this dataframe, return a dataframe with filtered fields.

        :param filter_to_apply: the filter to be applied to the source field, an array of boolean
        :param ddf: optional- the destination data frame
        :returns: a dataframe contains all the fields filterd, self if ddf is not set
        """
        if ddf is not None:
            if not isinstance(ddf, DataFrame):
                raise TypeError("The destination object must be an instance of DataFrame.")
            for name, field in self._columns.items():
                newfld = field.create_like(ddf, field.name[field.name.index('/', 1)+1:])
                ddf.add(field.apply_filter(filter_to_apply, dstfld=newfld), name=name)
            return ddf
        else:
            for field in self._columns.values():
                field.apply_filter(filter_to_apply)
            return self

    def apply_index(self, index_to_apply, ddf=None):
        """
        Apply the index to all the fields in this dataframe, return a dataframe with indexed fields.

        :param index_to_apply: the index to be applied to the fields, an ndarray of integers
        :param ddf: optional- the destination data frame
        :returns: a dataframe contains all the fields re-indexed, self if ddf is not set
        """
        if ddf is not None:
            if not isinstance(ddf, DataFrame):
                raise TypeError("The destination object must be an instance of DataFrame.")
            for name, field in self._columns.items():
                newfld = field.create_like(ddf, field.name[field.name.index('/', 1)+1:])
                idx = field.apply_index(index_to_apply, dstfld=newfld)
                ddf.add(idx, name=name)
            return ddf
        else:
            for field in self._columns.values():
                field.apply_index(index_to_apply)
            return self
