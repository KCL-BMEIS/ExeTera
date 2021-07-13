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


import numpy as np

import h5py

from exetera.core.abstract_types import Field
from exetera.core import readerwriter as rw
from exetera.core import fields as flds


def _writer_from_writer_or_group(writer_getter, param_name, writer):
    if isinstance(writer, h5py.Group):
        return writer_getter.get_existing_writer(writer)
    elif isinstance(writer, rw.Writer):
        return writer
    else:
        msg = "'{}' must be one of (h5py.Group, Writer) but is {}"
        raise ValueError(msg.format(param_name, type(writer)))


def _check_is_appropriate_writer_if_set(reader_getter, param_name, reader, writer):
    # TODO: this method needs reworking; readers should know whether writers are compatible with
    # them
    msg = "{} must be of type {} or None but is {}"
    # if writer is not None:
    #     if isinstance(reader, np.ndarray):
    #         raise ValueError("'if 'reader' is a numpy.ndarray, 'writer' must be None")

    if isinstance(reader, h5py.Group):
        reader = reader_getter.get_reader(reader)


    if isinstance(reader, rw.IndexedStringReader):
        if not isinstance(writer, rw.IndexedStringWriter):
            raise ValueError(msg.format(param_name, rw.IndexedStringReader, writer))
    elif isinstance(reader, rw.FixedStringReader):
        if not isinstance(writer, rw.FixedStringWriter):
            raise ValueError(msg.format(param_name, rw.FixedStringReader, writer))
    elif isinstance(reader, rw.NumericReader):
        if not isinstance(writer, rw.NumericWriter):
            raise ValueError(msg.format(param_name, rw.NumericReader, writer))
    elif isinstance(reader, rw.CategoricalReader):
        if not isinstance(writer, rw.CategoricalWriter):
            raise ValueError(msg.format(param_name, rw.CategoricalReader, writer))
    elif isinstance(reader, rw.TimestampReader):
        if not isinstance(writer, rw.TimestampWriter):
            raise ValueError(msg.format(param_name, rw.TimestampReader, writer))


def _check_all_readers_valid_and_same_type(readers):
    if not isinstance(readers, (tuple, list)):
        raise ValueError("'readers' collection must be a tuple or list")

    if isinstance(readers[0], h5py.Group):
        expected_type = h5py.Group
    elif isinstance(readers[0], rw.Reader):
        expected_type = rw.Reader
    elif isinstance(readers[0], Field):
        expected_type = Field
    elif isinstance(readers[0], np.ndarray):
        expected_type = np.ndarray
    else:
        raise ValueError("'readers' collection must of the following types: "
                         "(h5py.Group, Reader, numpy.ndarray)")
    for r in readers[1:]:
        if not isinstance(r, expected_type):
            raise ValueError("'readers': all elements must be the same underlying type "
                             "(h5py.Group, Reader, numpy.ndarray")


def _check_is_reader_substitute(name, field):
    if not isinstance(field, (h5py.Group, rw.Reader, np.ndarray)):
        msg = "'{}' must be one of (h5py.Group, Reader, numpy.ndarray) but is '{}'"
        raise ValueError(msg.format(type(field)))


def _check_is_reader_or_ndarray(name, field):
    if not isinstance(field, (rw.Reader, np.ndarray)):
        raise ValueError(f"'name' must be either a Reader or an ndarray but is {type(field)}")


def _check_is_reader_or_ndarray_if_set(name, field):
    if not isinstance(field, (rw.Reader, np.ndarray)):
        raise ValueError("if set, 'name' must be either a Reader or an ndarray "
                         f"but is {type(field)}")


def _check_equal_length(name1, field1, name2, field2):
    if len(field1) != len(field2):
        msg = "'{}' must be the same length as '{}' (lengths {} and {} respectively)"
        raise ValueError(msg.format(name1, name2, len(field1), len(field2)))


def _reader_from_group_if_required(reader_source, name, reader):
    if isinstance(reader, h5py.Group):
        return reader_source.get_reader(reader)
    return reader


def ensure_valid_field(name, field):
    if not isinstance(field, Field):
        raise ValueError("'{}' is not of type '{}'; expected Field".format(name, type(field)))


def ensure_valid_field_like(name, field):
    if not isinstance(field, (h5py.Group, Field, np.ndarray)):
        raise ValueError("'{}' is of type '{}'; expected Group, Field or ndarray".format(name, type(field)))


def raw_array_from_parameter(datastore, name, field):
    if isinstance(field, h5py.Group):
        return datastore.get(field).data[:]
    elif isinstance(field, rw.Reader):
        return field[:]
    elif isinstance(field, Field):
        return field.data[:]
    elif isinstance(field, np.ndarray):
        return field
    else:
        error_str = "'{}' must be one of (Group, Reader, Field or ndarray, but is {}"
        raise ValueError(error_str.format(name, type(field)))


def array_from_parameter(session, name, field):
    if isinstance(field, h5py.Group):
        return session.get(field).data[:]
    elif isinstance(field, Field):
        if field.indexed:
            return field.indices[:], field.values[:]
        return field.data[:]
    elif isinstance(field, np.ndarray):
        return field
    else:
        error_str = "'{}' must be one of (Group, Field, or ndarray, but is {}"
        raise ValueError(error_str.format(name, type(field)))


def array_from_field_or_lower(name, field):
    if isinstance(field, Field):
        return field.data[:]
    elif isinstance(field, np.ndarray):
        return field
    else:
        error_str = "'{}' must be one of (Field, or ndarray, but is {}"
        raise ValueError(error_str.format(name, type(field)))


def field_from_parameter(session, name, field):
    if isinstance(field, h5py.Group):
        return session.get(field)
    elif isinstance(field, Field):
        return field
    else:
        error_str = "'{}' must be one of (Group, Field, or ndarray, but is {}"
        raise ValueError(error_str.format(name, type(field)))


def is_field_parameter(field):
    return isinstance(field, (Field, h5py.Group))


def all_same_basic_type(name, fields):
    msg = "'{}' cannot be mixture of groups, fields and ndarrays".format(name)
    if isinstance(fields[0], h5py.Group):
        for f in fields[1:]:
            if not isinstance(f, h5py.Group):
                raise ValueError(msg)
    if isinstance(fields[0], Field):
        for f in fields[1:]:
            if not isinstance(f, Field):
                raise ValueError(msg)
    if isinstance(fields[0], np.ndarray):
        for f in fields[1:]:
            if not isinstance(f, np.ndarray):
                raise ValueError(msg)


def validate_key_field_consistency(lname, rname, lkey, rkey):
    left_tuple = isinstance(lkey, tuple)
    right_tuple = isinstance(rkey, tuple)
    if left_tuple ^ right_tuple:
        raise ValueError("Either none or both of '{}' and '{}' "
                         "must be tuples".format(lname, rname))
    if left_tuple and len(lkey) != len(rkey):
        raise ValueError("'{}' and '{}' must be the same length, but are of length "
                         "{} and {} respectively".format(lname, rname, len(lkey), len(rkey)))


def validate_and_get_key_fields(side, df, key):
    if isinstance(key, tuple):
        fields = []
        for ik, k in enumerate(key):
            dfk = df[k] if isinstance(k, str) else df[k]
            if dfk.indexed:
                if dfk.name is None:
                    raise ValueError("'{}': field at position {} is indexed; indexed fields"
                                     " cannot be used as keys".format(side, ik))
                else:
                    raise ValueError("'{}': field '{}' at position {} is indexed; "
                                     "indexed fields cannot be used "
                                     "as keys".format(side, dfk.name, ik))
            fields.append(dfk)
        return tuple(fields)
    else:
        field = df[key] if isinstance(key, str) else df[key]
        if field.indexed:
            raise ValueError("'{}': field is indexed; indexed fields cannot be "
                             "used as keys".format(side))
        return (field,)


def validate_key_lengths(side, df, key):
    lens = set()
    if isinstance(key, tuple):
        for k in key:
            lens.add(len(k.data))
            if len(lens) > 1:
                raise ValueError("'{}' keys are consistent lengths. The following "
                                 "lengths were observed: {}".format(side, lens))
    else:
        lens.add(len(key.data))
    return lens


def validate_field_lengths(side, lens, df, names=None):
    if names is None:
        names = df.keys()
    for n in names:
        lens.add(len(df[n].data))
    if len(lens) > 1:
        raise ValueError("'{}' fields are inconsistent lengths. The following "
                         "lengths were observed: {}".format(side, lens))
    return lens

def validate_and_normalize_categorical_key(param_name, key):
    if len(key) == 0:
        raise ValueError("'{}' cannot be empty".format(param_name))
    key_types = set()
    value_types = set()
    for k, v in key.items():
        key_types.add(type(k))
        value_types.add(type(v))

    if len(key_types) > 1:
        raise ValueError("'{}' has inconsistent key types {}".format(param_name, key_types))
    if len(value_types) > 1:
        raise ValueError("'{}' has inconsistent value types {}".format(param_name, value_types))

    items = list(key.items())
    key_type = items[0][0]
    value_type = items[0][1]

    if not isinstance(key_type, (str, bytes, int)) and not np.issubdtype(key_type, np.number):
        raise ValueError("'{}': Unexpected dictionary key type; must be str, bytes or int "
                         " but is {}".format(param_name, type(key_type)))
    if not isinstance(value_type, (str, bytes, int)) and not np.issubdtype(value_type, np.number):
        raise ValueError("'{}': Unexpected dictionary value type; must be str, bytes or int "
                         " but is {}".format(param_name, type(value_type)))

    if isinstance(key_type, (str, bytes)):
        if not isinstance(value_type, int) and not np.issubdtype(value_type, np.number):
            raise ValueError("'{}': if keys are of type str or bytes then values must be of "
                             "type int but are of type {}".format(param_name, type(value_type)))
    elif isinstance(value_type, (str, bytes)):
        if not isinstance(key_type, int) and not np.issubdtype(key_type, np.number):
            raise ValueError("'{}': if values are of type str or bytes then keys must be of "
                             "type int but are of type {}".format(param_name, type(key_type)))
    if not isinstance(key_type, (str, bytes)):
        # flip the dictionary
        if isinstance(key_type, str):
            return {v: k.encode() for k, v, in key}
        else:
            return {v: k for k, v in key}
    else:
        if isinstance(value_type, str):
            return {k: v.encode() for k, v, in key}
        else:
            return key
