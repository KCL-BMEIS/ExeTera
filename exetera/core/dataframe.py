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
from typing import Mapping, Optional, Sequence, Tuple, Union
from collections import OrderedDict
import numpy as np
import pandas as pd

from exetera.core.abstract_types import Dataset, DataFrame
from exetera.core import fields as fld
from exetera.core import operations as ops
from exetera.core import validation as val
import h5py


class HDF5DataFrame(DataFrame):
    """
    DataFrame is the means which which you interact with an ExeTera datastore. These are created
    and loaded through `Dataset.create_dataframe`, and other methods, rather than being constructed
    directly.

    DataFrames closely resemble Pandas DataFrames, but with a number of key differences:
    1. Instead of Series, DataFrames are composed of Field objects
    2. DataFrames can store fields of differing lengths, although all fields must be of the same
    length when performing certain operations such as merges.
    3. ExeTera DataFrames do not (yet) have the ability to create filtered views onto an underlying
    DataFrame, although this functionality will be added in upcoming releases

    For a detailed explanation of DataFrame along with examples of its use, please refer to the
    wiki documentation at
    https://github.com/KCL-BMEIS/ExeTera/wiki/DataFrame-API
    
    :param name: name of the dataframe.
    :param dataset: a dataset object, where this dataframe belongs to.
    :param h5group: the h5group object to store the fields. If the h5group is not empty, acquire data from h5group
        object directly. The h5group structure is h5group<-h5group-dataset structure, the later group has a
        'fieldtype' attribute and only one dataset named 'values'. So that the structure is mapped to
        Dataframe<-Field-Field.data automatically.
    :param dataframe: optional - replicate data from another dictionary of (name:str, field: Field).
    """
    def __init__(self,
                 dataset: Dataset,
                 name: str,
                 h5group: h5py.Group):
        """
        Create a Dataframe object, that contains a dictionary of fields. User should always create dataframe by
        dataset.create_dataframe, otherwise the dataframe is not stored in the dataset.
        """

        self.name = name
        self._columns = OrderedDict()
        self._dataset = dataset
        self._h5group = h5group

        for subg in h5group.keys():
            self._columns[subg] = dataset.session.get(h5group[subg])

    @property
    def columns(self):
        """
        The columns property interface. Columns is a dictionary to store the fields by (field_name, field_object).
        The field_name is field.name without prefix '/' and HDF5 group name.
        """
        return OrderedDict(self._columns)

    @property
    def dataset(self):
        """
        The dataset property interface.
        """
        return self._dataset

    @property
    def h5group(self):
        """
        The h5group property interface, used to handle underlying storage.
        """
        return self._h5group

    def add(self,
            field: fld.Field):
        """
        Add a field to this dataframe as well as the HDF5 Group.

        :param field: field to add to this dataframe, copy the underlying dataset
        """
        dname = field.name[field.name.index('/', 1)+1:]
        nfield = field.create_like(self, dname)
        if field.indexed:
            nfield.indices.write(field.indices[:])
            nfield.values.write(field.values[:])
        else:
            nfield.data.write(field.data[:])
        self._columns[dname] = nfield

    def drop(self,
             name: str):
        del self._columns[name]
        del self._h5group[name]

    def create_group(self,
                     name: str):
        """
        Create a group object in HDF5 file for field to use. Please note, this function is for
        backwards compatibility with older scripts and should not be used in the general case.

        :param name: the name of the group and field
        :return: a hdf5 group object
        """
        self._h5group.create_group(name)
        return self._h5group[name]

    def create_indexed_string(self,
                              name: str,
                              timestamp: Optional[str] = None,
                              chunksize: Optional[int] = None):
        """
        Create a indexed string type field.
        Please see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#indexedstringfield for
        a detailed description of indexed string fields
        """
        fld.indexed_string_field_constructor(self._dataset.session, self, name,
                                             timestamp, chunksize)
        field = fld.IndexedStringField(self._dataset.session, self._h5group[name], self,
                                       write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_fixed_string(self,
                            name: str,
                            length: int,
                            timestamp: Optional[str] = None,
                            chunksize: Optional[int] = None):
        """
        Create a fixed string type field.
        Please see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#fixedstringfield for
        a detailed description of fixed string fields
        """
        fld.fixed_string_field_constructor(self._dataset.session, self, name,
                                           length, timestamp, chunksize)
        field = fld.FixedStringField(self._dataset.session, self._h5group[name], self,
                                     write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_numeric(self,
                       name: str,
                       nformat: int,
                       timestamp: Optional[str] = None,
                       chunksize: Optional[int] = None):
        """
        Create a numeric type field.
        Please see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#numericfield for
        a detailed description of numeric fields
        """
        fld.numeric_field_constructor(self._dataset.session, self, name,
                                      nformat, timestamp, chunksize)
        field = fld.NumericField(self._dataset.session, self._h5group[name], self,
                                 write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_categorical(self,
                           name: str,
                           nformat: int,
                           key: dict,
                           timestamp: Optional[str] = None,
                           chunksize: Optional[int] = None):
        """
        Create a categorical type field.
        Please see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#categoricalfield for
        a detailed description of indexed string fields
        """
        fld.categorical_field_constructor(self._dataset.session, self, name, nformat, key,
                                          timestamp, chunksize)
        field = fld.CategoricalField(self._dataset.session, self._h5group[name], self,
                                     write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def create_timestamp(self,
                         name: str,
                         timestamp: Optional[str] = None,
                         chunksize: Optional[int] = None):
        """
        Create a timestamp type field.
        Please see https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#timestampfield for
        a detailed description of timestamp fields
        """
        fld.timestamp_field_constructor(self._dataset.session, self, name,
                                        timestamp, chunksize)
        field = fld.TimestampField(self._dataset.session, self._h5group[name], self,
                                   write_enabled=True)
        self._columns[name] = field
        return self._columns[name]

    def __contains__(self, name):
        """
        check if dataframe contains a field, by the field name

        :param name: the name of the field to check
        :return: A boolean value indicating whether this DataFrame contains a Field with the
            name in question
        """
        if not isinstance(name, str):
            raise TypeError("The name must be a str object.")
        else:
            return name in self._columns

    def contains_field(self, field):
        """
        check if dataframe contains a field by the field object

        :param field: the filed object to check, return a tuple(bool,str). The str is the name stored in dataframe.
        """
        if not isinstance(field, fld.Field):
            raise TypeError("The field must be a Field object")
        else:
            for v in self._columns.values():
                if id(field) == id(v):
                    return True
            return False

    def __getitem__(self, name):
        """
        Get a field stored by the field name.

        :param name: The name of field to get.
        """
        if not isinstance(name, str):
            raise TypeError("The name must be of type str but is of type '{}'".format(str))
        elif not self.__contains__(name):
            raise ValueError("There is no field named '{}' in this dataframe".format(name))
        else:
            return self._columns[name]

    def get_field(self, name):
        """
        Get a field stored by the field name.

        :param name: The name of field to get.
        """
        return self.__getitem__(name)

    def __setitem__(self, name, field):
        if not isinstance(name, str):
            raise TypeError("The name must be of type str but is of type '{}'".format(str))
        if not isinstance(field, fld.Field):
            raise TypeError("The field must be a Field object.")
        nfield = field.create_like(self, name)
        if field.indexed:
            nfield.indices.write(field.indices[:])
            nfield.values.write(field.values[:])
        else:
            nfield.data.write(field.data[:])
        self._columns[name] = nfield

    def __delitem__(self, name):
        if not self.__contains__(name=name):
            raise ValueError("There is no field named '{}' in this dataframe".format(name))
        else:
            del self._h5group[name]
            del self._columns[name]

    def delete_field(self, field):
        """
        Remove field from dataframe by field.

        :param field: The field to delete from this dataframe.
        """
        if field.dataframe != self:
            raise ValueError("This field is owned by a different dataframe")
        name = field.name
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

    def rename(self,
               field: Union[str, Mapping[str, str]],
               field_to: Optional[str] = None) -> None:
        """
        Rename provides you with the means to rename fields within a dataframe. You can specify either
        a single field to be renamed or you can provide a dictionary with a set of fields to be
        renamed.

        Example::
        
            # rename a single field
            df.rename('a', 'b')
    
            # rename multiple fields
            df.rename({'a': 'b', 'b': 'c', 'c': 'a'})

        Field renaming can fail if the resulting set of renamed fields would have name clashes. If
        this is the case, none of the rename operations go ahead and the dataframe remains unmodified.
        
        :param field: Either a string or a dictionary of name pairs, each of which is the existing
            field name and the destination field name
        :param field_to: Optional parameter containing a string, if `field` is a string. If 'field'
            is a dictionary, parameter should not be set.
            Field references remain valid after this operation and reflect their renaming.
        :return: None
        """

        if not isinstance(field, (str, dict)):
            raise ValueError("'field' must be of type str or dict but is {}").format(type(field))

        dict_ = None
        if isinstance(field, dict):
            if field_to is not None:
                raise ValueError("'field_to' can only be set when 'field' is a single column name")
            dict_ = field
        else:
            if field_to is None:
                raise ValueError("'field_to' must be set if 'field' is a column name")
            dict_ = {field: field_to}

        # check that we aren't creating ambiguity with the sequence of renames
        # --------------------------------------------------------------------
        keys = set(self._columns.keys())

        # first, remove the keys being renamed from the keyset
        for k in dict_.keys():
            keys.remove(k)

        # second, add them in one by one to ensure that they don't clash
        clashes = set()
        for v in dict_.values():
            if v in keys:
                clashes.add(v)
            keys.add(v)

        if len(clashes) > 0:
            raise ValueError("The attempted rename cannot be performed as it creates the "
                             "following name clashes: {}".format(clashes))

        def get_unique_name(name, keys):
            while name in keys:
                name += '_'
            return name

        # from here, we know there are no name clashes, but we might still have intermediate
        # clashes, so perform two renames where necessary
        final_renames = dict()
        intermediate_columns = OrderedDict()

        for k, f in self._columns.items():
            if k in dict_:
                uname = get_unique_name(dict_[k], self._columns)
                if uname != k:
                    final_renames[uname] = dict_[k]

                self._h5group.move(k, uname)
                intermediate_columns[uname] = f
            else:
                intermediate_columns[k] = f

        final_columns = OrderedDict()

        for k, f in intermediate_columns.items():
            if k in final_renames:
                name = final_renames[k]
                f = intermediate_columns[k]
                self._h5group.move(k, name)
                final_columns[name] = f
            else:
                final_columns[k] = f

        self._columns = final_columns


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
                newfld = field.create_like(ddf, name)
                field.apply_filter(filter_to_apply, target=newfld)
            return ddf
        else:
            for field in self._columns.values():
                field.apply_filter(filter_to_apply, in_place=True)
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
                newfld = field.create_like(ddf, name)
                field.apply_index(index_to_apply, target=newfld)
            return ddf
        else:
            for field in self._columns.values():
                field.apply_index(index_to_apply, in_place=True)
            return self


def copy(field: fld.Field, dataframe: DataFrame, name: str):
    """
    Copy a field to another dataframe as well as underlying dataset.

    :param field: The source field to copy.
    :param dataframe: The destination dataframe to copy to.
    :param name: The name of field under destination dataframe.
    """
    dfield = field.create_like(dataframe, name)
    if field.indexed:
        dfield.indices.write(field.indices[:])
        dfield.values.write(field.values[:])
    else:
        dfield.data.write(field.data[:])
    dataframe.columns[name] = dfield
    return dataframe[name]


def move(field: fld.Field, dest_df: DataFrame, name: str):
    """
    Move a field to another dataframe as well as underlying dataset.

    :param src_df: The source dataframe where the field is located.
    :param field: The field to move.
    :param dest_df: The destination dataframe to move to.
    :param name: The name of field under destination dataframe.
    """
    if field.dataframe == dest_df:
        dest_df.rename(field.name, name)
        return field
    else:
        copy(field, dest_df, name)
        field.dataframe.drop(field.name)
        field._valid_reference = False
        return dest_df[name]


def merge(left: DataFrame,
          right: DataFrame,
          dest: DataFrame,
          left_on: Union[Tuple[Union[str, fld.Field]], str, fld.Field],
          right_on: Union[Tuple[Union[str, fld.Field]], str, fld.Field],
          left_fields: Optional[Sequence[str]] = None,
          right_fields: Optional[Sequence[str]] = None,
          left_suffix: str = '_l',
          right_suffix: str = '_r',
          how='left'):
    """
    Merge 'left' and 'right' DataFrames into a destination dataset. The merge is a database-style
    join operation, in any of the following modes ("left", "right", "inner", "outer"). This
    method closely follows the Pandas 'merge' functionality.

    The join is performed using the fields specified by 'left_on' and 'right_on'; these can either
    be strings or fields; if they strings then they refer to fields that must exist in the
    corresponding dataframe.

    You can optionally set 'left_fields' and / or 'right_fields' if you want to have only a subset
    of fields joined from the left and right dataframes. If you don't want any fields to be joined
    from a given dataframe, you can pass an empty list.

    Fields are written to the destination dataframe. If the field names clash, they will get
    appended with the strings specified in 'left_suffix' and 'right_suffix' respectively.

    :param left: The left dataframe
    :param right: The right dataframe
    :param left_on: The field corresponding to the left key used to perform the join. This is either the
        the name of the field, or a field object. If it is a field object, it can be from another
        dataframe but it must be the same length as the fields being joined. This can also be a tuple
        of such values when performing joins on compound keys
    :param right_on: The field corresponding to the right key used to perform the join. This is either
        the name of the field, or a field object. If it is a field object, it can be from another
        dataframe but it must be the same length as the fields being joined. This can also be a tuple
        of such values when performing joins on compound keys
    :param left_fields: Optional parameter listing which fields are to be joined from the left table. If
        this is not set, all fields from the left table are joined
    :param right_fields: Optional parameter listing which fields are to be joined from the right table.
        If this is not set, all fields from the right table are joined
    :param left_suffix: A string to be appended to fields from the left table if they clash with fields from the 
        right table.
    :param right_suffix: A string to be appended to fields from the right table if they clash with fields from the 
        left table.
    :param how: Optional parameter specifying the merge mode. It must be one of ('left', 'right',
        'inner', 'outer' or 'cross). If not set, the 'left' join is performed.

    """

    if not isinstance(left, DataFrame):
        raise ValueError("'left' must be a DataFrame but is of type '{}'".format(type(left)))

    if not isinstance(right, DataFrame):
        raise ValueError("'right' must be a DataFrame but is of type '{}'".format(type(right)))

    supported_modes = ('left', 'right', 'inner', 'outer', 'cross')
    if how not in supported_modes:
        raise ValueError("'how' must be one of {} but is {}".format(supported_modes, how))

    # check that left_on and right_on are mutually compatible
    val.validate_key_field_consistency('left_on', 'right_on', left_on, right_on)

    # check that fields are of the correct field type
    left_on_fields = val.validate_and_get_key_fields('left_on', left, left_on)
    right_on_fields = val.validate_and_get_key_fields('right_on', right, right_on)

    # check the consistency of field lengths
    left_lens = val.validate_key_lengths('left_on', left, left_on_fields)
    right_lens = val.validate_key_lengths('right_on', right, right_on_fields)

    # check consistency of fields with key lengths
    val.validate_field_lengths('left', left_lens, left, left_fields)
    val.validate_field_lengths('right', right_lens, right, right_fields)

    left_len = list(left_lens)[0]
    right_len = list(right_lens)[0]

    if left_len < (2 << 30) and right_len < (2 << 30):
        index_dtype = np.int32
    else:
        index_dtype = np.int64

    left_df_dict = {}
    right_df_dict = {}
    left_on_keys = []
    right_on_keys = []
    if isinstance(left_on_fields, tuple):
        for i_f, f in enumerate(left_on_fields):
            key = 'l_k_{}'.format(i_f)
            left_df_dict[key] = f.data[:]
            left_on_keys.append(key)
        l_key = tuple(left_on_keys)
        for i_f, f in enumerate(right_on_fields):
            key = 'r_k_{}'.format(i_f)
            right_df_dict[key] = f.data[:]
            right_on_keys.append(key)
        r_key = tuple(right_on_keys)
    else:
        l_key = 'l_k'
        left_df_dict[l_key] = left_on_fields.data[:]
        r_key = 'r_k'
        right_df_dict[r_key] = right_on_fields.data[:]

    left_df_dict['l_i'] = np.arange(left_len, dtype=index_dtype)
    right_df_dict['r_i'] = np.arange(right_len, dtype=index_dtype)
    # create the merging dataframes, using only the fields involved in the merge
    l_df = pd.DataFrame(left_df_dict)
    r_df = pd.DataFrame(right_df_dict)

    df = pd.merge(left=l_df, right=r_df, left_on=l_key, right_on=r_key, how=how)

    l_to_d_map = df['l_i'].to_numpy(dtype=np.int32)
    l_to_d_filt = np.logical_not(df['l_i'].isnull()).to_numpy()
    r_to_d_map = df['r_i'].to_numpy(dtype=np.int32)
    r_to_d_filt = np.logical_not(df['r_i'].isnull()).to_numpy()

    # perform the mapping
    left_fields_ = left.keys() if left_fields is None else left_fields
    right_fields_ = right.keys() if right_fields is None else right_fields

    for f in left_fields_:
        dest_f = f
        if f in right_fields_:
            dest_f += left_suffix
        l = left[f]
        d = l.create_like(dest, dest_f)
        if l.indexed:
            i, v = ops.safe_map_indexed_values(l.indices[:], l.values[:], l_to_d_map, l_to_d_filt)
            d.indices.write(i)
            d.values.write(v)
        else:
            v = ops.safe_map_values(l.data[:], l_to_d_map, l_to_d_filt)
            d.data.write(v)
    if np.all(l_to_d_filt) == False:
        d = dest.create_numeric('valid'+left_suffix, 'bool')
        d.data.write(l_to_d_filt)

    for f in right_fields_:
        dest_f = f
        if f in left_fields_:
            dest_f += right_suffix
        r = right[f]
        d = r.create_like(dest, dest_f)
        if r.indexed:
            i, v = ops.safe_map_indexed_values(r.indices[:], r.values[:], r_to_d_map, r_to_d_filt)
            d.indices.write(i)
            d.values.write(v)
        else:
            v = ops.safe_map_values(r.data[:], r_to_d_map, r_to_d_filt)
            d.data.write(v)
    if np.all(r_to_d_filt) == False:
        d = dest.create_numeric('valid'+right_suffix, 'bool')
        d.data.write(r_to_d_filt)
