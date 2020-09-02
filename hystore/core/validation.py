import numpy as np

import h5py

from hystore.core import persistence as per
from hystore.core import fields as fld
from hystore.core import readerwriter as rw


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


def raw_array_from_parameter(datastore, name, field):
    if isinstance(field, h5py.Group):
        return datastore.get_reader(field)[:]
    elif isinstance(field, rw.Reader):
        return field[:]
    elif isinstance(field, fld.Field):
        return field.data[:]
    elif isinstance(field, np.ndarray):
        return field
    else:
        error_str = "'{}' must be one of (Group, Reader, Field or ndarray, but is {}"
        raise ValueError(error_str.format(name, type(field)))


def array_from_parameter(session, name, field):
    if isinstance(field, h5py.Group):
        return session.get(field).data[:]
    elif isinstance(field, fld.Field):
        return field.data[:]
    elif isinstance(field, np.ndarray):
        return field
    else:
        error_str = "'{}' must be one of (Group, Field, or ndarray, but is {}"
        raise ValueError(error_str.format(name, type(field)))


def field_from_parameter(session, name, field):
    if isinstance(field, h5py.Group):
        return session.get(field)
    elif isinstance(field, fld.Field):
        return field
    else:
        error_str = "'{}' must be one of (Group, Field, or ndarray, but is {}"
        raise ValueError(error_str.format(name, type(field)))

def is_field_parameter(field):
    return isinstance(field, (fld.Field, h5py.Group))

def all_same_basic_type(name, fields):
    msg = "'{}' cannot be mixture of groups, fields and ndarrays".format(name)
    if isinstance(fields[0], h5py.Group):
        for f in fields[1:]:
            if not isinstance(f, h5py.Group):
                raise ValueError(msg)
    if isinstance(fields[0], fld.Field):
        for f in fields[1:]:
            if not isinstance(f, fld.Field):
                raise ValueError(msg)
    if isinstance(fields[0], np.ndarray):
        for f in fields[1:]:
            if not isinstance(f, np.ndarray):
                raise ValueError(msg)
