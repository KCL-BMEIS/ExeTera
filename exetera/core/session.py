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
from datetime import datetime, timezone
import time
import warnings
import numpy as np
import pandas as pd

import h5py

from exetera.core.abstract_types import Field, AbstractSession
from exetera.core import operations
from exetera.core import persistence as per
from exetera.core import fields as fld
from exetera.core import readerwriter as rw
from exetera.core import validation as val
from exetera.core import operations as ops
from exetera.core import dataset as ds
from exetera.core import dataframe as df


class Session(AbstractSession):

    def __init__(self, chunksize=ops.DEFAULT_CHUNKSIZE,
                 timestamp=str(datetime.now(timezone.utc))):
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.chunksize = chunksize
        self.timestamp = timestamp
        self.datasets = dict()


    def __enter__(self):
        return self


    def __exit__(self, etype, evalue, etraceback):
        self.close()


    def open_dataset(self, dataset_path, mode, name):
        """
        Open a dataset with the given access mode
        :param dataset_path: the path to the dataset
        :param mode: the mode in which the dataset should be opened. This is one of "r", "r+" or "w".
        :return: The top-level dataset object
        """
        h5py_modes = {"r": "r", "r+": "r+", "w": "w"}
        if name in self.datasets:
            raise ValueError("A dataset with name '{}' is already open, and must be closed first.".format(name))

        self.datasets[name] = ds.HDF5Dataset(self, dataset_path, mode, name)
        return self.datasets[name]


    def close_dataset(self, name):
        """
        Close the dataset with the given name. If there is no dataset with that name, do nothing.
        :param name: The name of the dataset to be closed
        :return: None
        """
        if name in self.datasets:
            self.datasets[name].close()
            del self.datasets[name]


    def list_datasets(self):
        return tuple(n for n in self.datasets.keys())


    def get_dataset(self, name):
        return self.datasets[name]


    def close(self):
        """
        Close all open datasets
        :return: None
        """
        for v in self.datasets.values():
            v.close()
        self.datasets = dict()


    def get_shared_index(self, keys):
        """
        Create a shared index based on a tuple numpy arrays containing keys.
        This function generates the sorted union of a tuple of key fields and
        then maps the individual arrays to their corresponding indices in the
        sorted union.

        Example:
            key_1 = ['a', 'b', 'e', 'g', 'i']
            key_2 = ['b', 'b', 'c', 'c, 'e', 'g', 'j']
            key_3 = ['a', 'c' 'd', 'e', 'g', 'h', 'h', 'i']

            sorted_union = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j']

            key_1_index = [0, 1, 4, 5, 7]
            key_2_index = [1, 1, 2, 2, 4, 5, 8]
            key_3_index = [0, 2, 3, 4, 5, 6, 6, 7]

            :param keys: a tuple of groups, fields or ndarrays whose contents represent keys
        """
        if not isinstance(keys, tuple):
            raise ValueError("'keys' must be a tuple")
        concatted = None
        raw_keys = list()
        for k in keys:
            raw_field = val.raw_array_from_parameter(self, 'keys', k)
            raw_keys.append(raw_field)
            if concatted is None:
                concatted = pd.unique(raw_field)
            else:
                concatted = np.concatenate((concatted, raw_field), axis=0)
        concatted = pd.unique(concatted)
        concatted = np.sort(concatted)

        return tuple(np.searchsorted(concatted, k) for k in raw_keys)


    def set_timestamp(self, timestamp=str(datetime.now(timezone.utc))):
        """
        Set the default timestamp to be used when creating fields without an explicit
        timestamp specified.

        :param timestamp a string representing a valid Datetime
        :return: None
        """
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.timestamp = timestamp



    def sort_on(self, src_group, dest_group, keys,
                timestamp=datetime.now(timezone.utc), write_mode='write', verbose=True):
        """
        Sort a group (src_group) of fields by the specified set of keys, and write the
        sorted fields to dest_group.

        :param src_group: the group of fields that are to be sorted
        :param dest_group: the group into which sorted fields are written
        :param keys: fields to sort on
        :param timestamp: optional - timestamp to write on the sorted fields
        :param write_mode: optional - write mode to use if the destination fields already
        exist
        :return: None
        """
        # TODO: fields is being ignored at present
        def print_if_verbose(*args):
            if verbose:
                print(*args)

        readers = tuple(self.get(src_group[f]) for f in keys)
        t1 = time.time()
        sorted_index = self.dataset_sort_index(
            readers, np.arange(len(readers[0].data), dtype=np.uint32))
        print_if_verbose(f'sorted {keys} index in {time.time() - t1}s')

        t0 = time.time()
        for k in src_group.keys():
            t1 = time.time()
            if src_group != dest_group:
                r = self.get(src_group[k])
                w = r.create_like(dest_group, k, timestamp)
                self.apply_index(sorted_index, r, w)
                del r
                del w
            else:
                r = self.get(src_group[k]).writeable()
                if isinstance(r, fld.IndexedStringField):
                    i, v = self.apply_index(sorted_index, r)
                    r.indices[:] = i
                    r.values[:] = v
                else:
                    r.data[:] = self.apply_index(sorted_index, r)
                del r
                print_if_verbose(f"  '{k}' reordered in {time.time() - t1}s")
        print_if_verbose(f"fields reordered in {time.time() - t0}s")



    def dataset_sort_index(self, sort_indices, index=None):
        """
        Generate a sorted index based on a set of fields upon which to sort and an optional
        index to apply to the sort_indices
        :param sort_indices: a tuple or list of indices that determine the sorted order
        :param index: optional - the index by which the initial field should be permuted
        :return: the resulting index that can be used to permute unsorted fields
        """
        val._check_all_readers_valid_and_same_type(sort_indices)
        r_readers = tuple(reversed(sort_indices))

        raw_data = val.raw_array_from_parameter(self, 'readers', r_readers[0])

        if index is None:
            raw_index = np.arange(len(raw_data))
        else:
            raw_index = val.raw_array_from_parameter(self, 'index', index)

        acc_index = raw_index
        fdata = raw_data[acc_index]
        index = np.argsort(fdata, kind='stable')
        acc_index = acc_index[index]

        for r in r_readers[1:]:
            raw_data = val.raw_array_from_parameter(self, 'readers', r)
            fdata = raw_data[acc_index]
            index = np.argsort(fdata, kind='stable')
            acc_index = acc_index[index]

        return acc_index


    def apply_filter(self, filter_to_apply, src, dest=None):
        """
        Apply a filter to an a src field. The filtered field is written to dest if it set,
        and returned from the function call. If the field is an IndexedStringField, the
        indices and values are returned separately.

        :param filter_to_apply: the filter to be applied to the source field, an array of boolean
        :param src: the field to be filtered
        :param dest: optional - a field to write the filtered data to
        :return: the filtered values
        """
        filter_to_apply_ = val.array_from_parameter(self, 'index_to_apply', filter_to_apply)
        writer_ = None
        if dest is not None:
            writer_ = val.field_from_parameter(self, 'writer', dest)
        if isinstance(src, fld.IndexedStringField):
            newfld = src.apply_filter(filter_to_apply_, writer_)
            return newfld.indices, newfld.values
        elif isinstance(src,fld.Field):
            newfld = src.apply_filter(filter_to_apply_, writer_)
            return newfld.data[:]
        #elif isinstance(src, df.datafrme):
        else:
            reader_ = val.array_from_parameter(self, 'reader', src)
            result = reader_[filter_to_apply]
            if writer_:
                writer_.data.write(result)
            return result


    def apply_index(self, index_to_apply, src, dest=None):
        """
        Apply a index to an a src field. The indexed field is written to dest if it set,
        and returned from the function call. If the field is an IndexedStringField, the
        indices and values are returned separately.

        :param index_to_apply: the index to be applied to the source field, must be one of Group, Field, or ndarray
        :param src: the field to be index
        :param dest: optional - a field to write the indexed data to
        :return: the indexed values
        """
        index_to_apply_ = val.array_from_parameter(self, 'index_to_apply', index_to_apply)
        writer_ = None
        if dest is not None:
            writer_ = val.field_from_parameter(self, 'writer', dest)
        if isinstance(src, fld.IndexedStringField):
            dest_indices, dest_values = \
                ops.apply_indices_to_index_values(index_to_apply_,
                                                  src.indices[:], src.values[:])
            return dest_indices, dest_values
        elif isinstance(src,fld.Field):
            newfld = src.apply_index(index_to_apply_, writer_)
            return newfld.data[:]
        else:
            reader_ = val.array_from_parameter(self, 'reader', src)
            result = reader_[index_to_apply]
            if writer_:
                writer_.data.write(result)
            return result


    # TODO: write distinct with new readers / writers rather than deleting this
    def distinct(self, field=None, fields=None, filter=None):
        if field is None and fields is None:
            return ValueError("One of 'field' and 'fields' must be set")
        if field is not None and fields is not None:
            return ValueError("Only one of 'field' and 'fields' may be set")

        if field is not None:
            return np.unique(field)

        entries = [(f'{i}', f.dtype) for i, f in enumerate(fields)]
        unified = np.empty_like(fields[0], dtype=np.dtype(entries))
        for i, f in enumerate(fields):
            unified[f'{i}'] = f

        uniques = np.unique(unified)
        results = [uniques[f'{i}'] for i in range(len(fields))]
        return results


    def get_spans(self, field=None, fields=None):
        """
        Calculate a set of spans that indicate contiguous equal values.
        The entries in the result array correspond to the inclusive start and
        exclusive end of the span (the ith span is represented by element i and
        element i+1 of the result array). The last entry of the result array is
        the length of the source field.

        Only one of 'field' or 'fields' may be set. If 'fields' is used and more
        than one field specified, the fields are effectively zipped and the check
        for spans is carried out on each corresponding tuple in the zipped field.

        Example:
            field: [1, 2, 2, 1, 1, 1, 3, 4, 4, 4, 2, 2, 2, 2, 2]
            result: [0, 1, 3, 6, 7, 10, 15]

        """

        if fields is not None:
            if isinstance(fields[0], fld.Field):
                return ops._get_spans_for_2_fields_by_spans(fields[0].get_spans(), fields[1].get_spans())
            if isinstance(fields[0], np.ndarray):
                return ops._get_spans_for_2_fields(fields[0], fields[1])
        else:
            if isinstance(field, fld.Field):
                return field.get_spans()
            if isinstance(field, np.ndarray):
                return ops.get_spans_for_field(field)



    def _apply_spans_no_src(self, predicate, spans, dest=None):
        assert(dest is None or isinstance(dest, Field))
        if dest is not None:
            dest_f = val.field_from_parameter(self, 'dest', dest)
            results = np.zeros(len(spans) - 1, dtype=dest_f.data.dtype)
            predicate(spans, results)
            dest_f.data.write(results)
            return results
        else:
            results = np.zeros(len(spans) - 1, dtype='int64')
            predicate(spans, results)
            return results

    def _apply_spans_src(self, predicate, spans, src, dest=None):
        assert(dest is None or isinstance(dest, Field))
        src_ = val.array_from_parameter(self, 'src', src)
        if len(src) != spans[-1]:
            error_msg = ("'src' (length {}) must be one element shorter than 'spans' "
                         "(length {})")
            raise ValueError(error_msg.format(len(src_), len(spans)))

        if dest is not None:
            dest_f = val.field_from_parameter(self, 'dest', dest)
            results = np.zeros(len(spans) - 1, dtype=dest_f.data.dtype)
            predicate(spans, src_, results)
            dest_f.data.write(results)
            return results
        else:
            results = np.zeros(len(spans) - 1, dtype=src_.dtype)
            predicate(spans, src_, results)
            return results

    def apply_spans_index_of_min(self, spans, src, dest=None):
        return self._apply_spans_src(ops.apply_spans_index_of_min, spans, src, dest)

    def apply_spans_index_of_max(self, spans, src, dest=None):
        return self._apply_spans_src(ops.apply_spans_index_of_max, spans, src, dest)

    def apply_spans_index_of_first(self, spans, dest=None):
        return self._apply_spans_no_src(ops.apply_spans_index_of_first, spans, dest)

    def apply_spans_index_of_last(self, spans, dest=None):
        return self._apply_spans_no_src(ops.apply_spans_index_of_last, spans, dest)

    def apply_spans_count(self, spans, src=None, dest=None):
        return self._apply_spans_no_src(ops.apply_spans_count, spans, dest)

    def apply_spans_min(self, spans, src, dest=None):
        return self._apply_spans_src(ops.apply_spans_min, spans, src, dest)

    def apply_spans_max(self, spans, src, dest=None):
        return self._apply_spans_src(ops.apply_spans_max, spans, src, dest)

    def apply_spans_first(self, spans, src, dest=None):
        return self._apply_spans_src(ops.apply_spans_first, spans, src, dest)

    def apply_spans_last(self, spans, src, dest=None):
        return self._apply_spans_src(ops.apply_spans_last, spans, src, dest)

    def apply_spans_concat(self, spans, src, dest,
                           src_chunksize=None, dest_chunksize=None, chunksize_mult=None):
        if not isinstance(src, fld.IndexedStringField):
            raise ValueError(f"'src' must be one of 'IndexedStringField' but is {type(src)}")
        if not isinstance(dest, fld.IndexedStringField):
            raise ValueError(f"'dest' must be one of 'IndexedStringField' but is {type(dest)}")

        src_chunksize = src.chunksize if src_chunksize is None else src_chunksize
        dest_chunksize = dest.chunksize if dest_chunksize is None else dest_chunksize
        chunksize_mult = 16 if chunksize_mult is None else chunksize_mult


        src_index = src.indices[:]
        src_values = src.values[:]
        dest_index = np.zeros(src_chunksize, src_index.dtype)
        dest_values = np.zeros(dest_chunksize * chunksize_mult, src_values.dtype)

        max_index_i = src_chunksize
        max_value_i = dest_chunksize * chunksize_mult // 2

        if src_values.dtype == 'S1':
            separator = b','
            delimiter = b'"'
        elif src_values.dtype == np.uint8:
            separator = np.frombuffer(b',', dtype='S1')[0][0]
            delimiter = np.frombuffer(b'"', dtype='S1')[0][0]

        s = 0
        index_v = 0
        while s < len(spans) - 1:
            # s, index_i, index_v = per._apply_spans_concat(spans, src_index, src_values,
            #                                               dest_index, dest_values,
            #                                               max_index_i, max_value_i, s,
            #                                               separator, delimiter)
            s, index_i, index_v = per._apply_spans_concat_2(spans, src_index, src_values,
                                                            dest_index, dest_values,
                                                            max_index_i, max_value_i,
                                                            separator, delimiter, s, index_v)

            if index_i > 0 or index_v > 0:
                dest.indices.write_part(dest_index[:index_i])
                dest.values.write_part(dest_values[:index_v])
        dest.indices.complete()
        dest.values.complete()
        #         dest.write_raw(dest_index[:index_i], dest_values[:index_v])
        # dest.complete()


    def _aggregate_impl(self, predicate, index, src=None, dest=None):
        index_ = val.raw_array_from_parameter(self, "index", index)

        dest_field = None
        if dest is not None:
            dest_field = val.field_from_parameter(self, "dest", dest)

        fkey_index_spans = self.get_spans(field=index)

        # execute the predicate (note that not every predicate requires a reader)
        results = predicate(fkey_index_spans, src, dest_field)

        return dest if dest is not None else results


    def aggregate_count(self, index=None, src=None, dest=None):
        return self._aggregate_impl(self.apply_spans_count, index, src, dest)


    def aggregate_first(self, index, src=None, dest=None):
        return self.aggregate_custom(self.apply_spans_first, index, src, dest)


    def aggregate_last(self, index, src=None, dest=None):
        return self.aggregate_custom(self.apply_spans_last, index, src, dest)


    def aggregate_min(self, index, src=None, dest=None):
        return self.aggregate_custom(self.apply_spans_min, index, src, dest)


    def aggregate_max(self, index, src=None, dest=None):
        return self.aggregate_custom(self.apply_spans_max, index, src, dest)


    def aggregate_custom(self, predicate, index, src=None, dest=None):
        if src is None:
            raise ValueError("'src' must not be None")
        val.ensure_valid_field_like("src", src)
        if dest is not None:
            val.ensure_valid_field("dest", dest)

        return self._aggregate_impl(predicate, index, src, dest)


    def join(self,
             destination_pkey, fkey_indices, values_to_join,
             writer=None, fkey_index_spans=None):

        if isinstance(destination_pkey, fld.IndexedStringField):
            raise ValueError("'destination_pkey' must not be an indexed string field")
        if isinstance(fkey_indices, fld.IndexedStringField):
            raise ValueError("'fkey_indices' must not be an indexed string field")
        if isinstance(values_to_join, rw.IndexedStringReader):
            raise ValueError("Joins on indexed string fields are not supported")

        raw_fkey_indices = val.raw_array_from_parameter(self, "fkey_indices", fkey_indices)

        raw_values_to_join = val.raw_array_from_parameter(self, "values_to_join", values_to_join)

        # generate spans for the sorted key indices if not provided
        if fkey_index_spans is None:
            fkey_index_spans = self.get_spans(field=raw_fkey_indices)

        # select the foreign keys from the start of each span to get an ordered list
        # of unique id indices in the destination space that the results of the predicate
        # execution are mapped to
        unique_fkey_indices = raw_fkey_indices[fkey_index_spans[:-1]]

        # generate a filter to remove invalid foreign key indices (where values in the
        # foreign key don't map to any values in the destination space
        invalid_filter = unique_fkey_indices < operations.INVALID_INDEX
        safe_unique_fkey_indices = unique_fkey_indices[invalid_filter]

        # the predicate results are in the same space as the unique_fkey_indices, which
        # means they may still contain invalid indices, so filter those now
        safe_values_to_join = raw_values_to_join[invalid_filter]

        # now get the memory that the results will be mapped to
        #destination_space_values = writer.chunk_factory(len(destination_pkey))
        destination_space_values = np.zeros(len(destination_pkey), dtype=raw_values_to_join.dtype)


        # finally, map the results from the source space to the destination space
        destination_space_values[safe_unique_fkey_indices] = safe_values_to_join

        if writer is not None:
            writer.data.write(destination_space_values)
        else:
            return destination_space_values


    def predicate_and_join(self,
                           predicate, destination_pkey, fkey_indices,
                           reader=None, writer=None, fkey_index_spans=None):
        if reader is not None:
            if not isinstance(reader, rw.Reader):
                raise ValueError(f"'reader' must be a type of Reader but is {type(reader)}")
            if isinstance(reader, rw.IndexedStringReader):
                raise ValueError(f"Joins on indexed string fields are not supported")

        # generate spans for the sorted key indices if not provided
        if fkey_index_spans is None:
            fkey_index_spans = self.get_spans(field=fkey_indices)

        # select the foreign keys from the start of each span to get an ordered list
        # of unique id indices in the destination space that the results of the predicate
        # execution are mapped to
        unique_fkey_indices = fkey_indices[:][fkey_index_spans[:-1]]

        # generate a filter to remove invalid foreign key indices (where values in the
        # foreign key don't map to any values in the destination space
        invalid_filter = unique_fkey_indices < operations.INVALID_INDEX
        safe_unique_fkey_indices = unique_fkey_indices[invalid_filter]

        # execute the predicate (note that not every predicate requires a reader)
        if reader is not None:
            dtype = reader.dtype()
        else:
            dtype = np.uint32
        results = np.zeros(len(fkey_index_spans)-1, dtype=dtype)
        predicate(fkey_index_spans, reader, results)

        # the predicate results are in the same space as the unique_fkey_indices, which
        # means they may still contain invalid indices, so filter those now
        safe_results = results[invalid_filter]

        # now get the memory that the results will be mapped to
        destination_space_values = writer.chunk_factory(len(destination_pkey))
        # finally, map the results from the source space to the destination space
        destination_space_values[safe_unique_fkey_indices] = safe_results

        writer.write(destination_space_values)


    def get(self, field):
        if isinstance(field, Field):
            return field

        if 'fieldtype' not in field.attrs.keys():
            raise ValueError(f"'{field}' is not a well-formed field")

        fieldtype_map = {
            'indexedstring': fld.IndexedStringField,
            'fixedstring': fld.FixedStringField,
            'categorical': fld.CategoricalField,
            'boolean': fld.NumericField,
            'numeric': fld.NumericField,
            'datetime': fld.TimestampField,
            'date': fld.TimestampField,
            'timestamp': fld.TimestampField
        }

        fieldtype = field.attrs['fieldtype'].split(',')[0]
        return fieldtype_map[fieldtype](self, field)


    def create_like(self, field, dest_group, dest_name, timestamp=None, chunksize=None):
        if isinstance(field, h5py.Group):
            if 'fieldtype' not in field.attrs.keys():
                raise ValueError("{} is not a well-formed field".format(field))
            f = self.get(field)
            return f.create_like(dest_group, dest_name)
        elif isinstance(field, Field):
            return field.create_like(dest_group, dest_name)
        else:
            raise ValueError("'field' must be either a Field or a h5py.Group, but is {}".format(type(field)))


    def create_indexed_string(self, group, name, timestamp=None, chunksize=None):

        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            return fld.indexed_string_field_constructor(self, group, name, timestamp, chunksize)
        else:
            return group.create_indexed_string(name, timestamp, chunksize)


    def create_fixed_string(self, group, name, length, timestamp=None, chunksize=None):
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))
        if isinstance(group, h5py.Group):
            return fld.fixed_string_field_constructor(self, group, name, timestamp, chunksize)
        else:
            return group.create_fixed_string(name, length, timestamp, chunksize)


    def create_categorical(self, group, name, nformat, key,
                           timestamp=None, chunksize=None):
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            return fld.categorical_field_constructor(self, group, name, timestamp, chunksize)
        else:
            return group.create_categorical(name, nformat, key, timestamp, chunksize)


    def create_numeric(self, group, name, nformat, timestamp=None, chunksize=None):
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            return fld.numeric_field_constructor(self. group, name, timestamp, chunksize)
        else:
            return group.create_numeric(name, nformat, timestamp, chunksize)


    def create_timestamp(self, group, name, timestamp=None, chunksize=None):
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            return fld.categorical_field_constructor(self, group, name, timestamp, chunksize)
        else:
            return group.create_timestamp(name, timestamp, chunksize)


    def get_or_create_group(self, group, name):
        if name in group:
            return group[name]
        return group.create_group(name)


    def chunks(self, length, chunksize=None):
        if chunksize is None:
            chunksize = self.chunksize
        cur = 0
        while cur < length:
            next = min(length, cur + chunksize)
            yield cur, next
            cur = next


    def process(self, inputs, outputs, predicate):

        # TODO: modifying the dictionaries in place is not great
        input_readers = dict()
        for k, v in inputs.items():
            if isinstance(v, rw.Reader):
                input_readers[k] = v
            else:
                input_readers[k] = self.get_reader(v)
        output_writers = dict()
        output_arrays = dict()
        for k, v in outputs.items():
            if isinstance(v, rw.Writer):
                output_writers[k] = v
            else:
                raise ValueError("'outputs': all values must be 'Writers'")

        reader = next(iter(input_readers.values()))
        input_length = len(reader)
        writer = next(iter(output_writers.values()))
        chunksize = writer.chunksize
        required_chunksize = min(input_length, chunksize)
        for k, v in outputs.items():
            output_arrays[k] = output_writers[k].chunk_factory(required_chunksize)

        for c in self.chunks(input_length, chunksize):
            kwargs = dict()

            for k, v in inputs.items():
                kwargs[k] = v[c[0]:c[1]]
            for k, v in output_arrays.items():
                kwargs[k] = v[:c[1] - c[0]]
            predicate(**kwargs)

            # TODO: write back to the writer
            for k in output_arrays.keys():
                output_writers[k].write_part(kwargs[k])
        for k, v in output_writers.items():
            output_writers[k].flush()


    def get_index(self, target, foreign_key, destination=None):
        print('  building patient_id index')
        t0 = time.time()
        target_lookup = dict()
        target_ = val.raw_array_from_parameter(self, "target", target)
        for i, v in enumerate(target_):
            target_lookup[v] = i
        print(f'  target lookup built in {time.time() - t0}s')

        print('  perform initial index')
        t0 = time.time()
        foreign_key_elems = val.raw_array_from_parameter(self, "foreign_key", foreign_key)
        # foreign_key_index = np.asarray([target_lookup.get(i, -1) for i in foreign_key_elems],
        #                                    dtype=np.int64)
        foreign_key_index = np.zeros(len(foreign_key_elems), dtype=np.int64)

        current_invalid = np.int64(operations.INVALID_INDEX)
        for i_k, k in enumerate(foreign_key_elems):
            index = target_lookup.get(k, current_invalid)
            if index >= operations.INVALID_INDEX:
                current_invalid += 1
                target_lookup[k] = index
            foreign_key_index[i_k] = index
        print(f'  initial index performed in {time.time() - t0}s')

        if destination is not None:
            if val.is_field_parameter(destination):
                destination.data.write(foreign_key_index)
            else:
                destination[:] = foreign_key_index
        else:
            return foreign_key_index


    def get_trash_group(self, group):

        group_names = group.name[1:].split('/')

        while True:
            id = str(uuid.uuid4())
            try:
                result = group.create_group(f"/trash/{'/'.join(group_names[:-1])}/{id}")
                return result
            except KeyError:
                pass


    def temp_filename(self):
        uid = str(uuid.uuid4())
        while os.path.exists(uid + '.hdf5'):
            uid = str(uuid.uuid4())
        return uid + '.hdf5'


    def merge_left(self, left_on, right_on,
                   right_fields=tuple(), right_writers=None):
        l_key_raw = val.raw_array_from_parameter(self, 'left_on', left_on)
        l_index = np.arange(len(l_key_raw), dtype=np.int64)
        l_df = pd.DataFrame({'l_k': l_key_raw, 'l_index': l_index})

        r_key_raw = val.raw_array_from_parameter(self, 'right_on', right_on)
        r_index = np.arange(len(r_key_raw), dtype=np.int64)
        r_df = pd.DataFrame({'r_k': r_key_raw, 'r_index': r_index})

        df = pd.merge(left=l_df, right=r_df, left_on='l_k', right_on='r_k', how='left')
        r_to_l_map = df['r_index'].to_numpy(dtype=np.int64)
        r_to_l_filt = np.logical_not(df['r_index'].isnull()).to_numpy()

        right_results = list()
        for irf, rf in enumerate(right_fields):
            rf_raw = val.raw_array_from_parameter(self, 'right_fields[{}]'.format(irf), rf)
            joined_field = ops.safe_map(rf_raw, r_to_l_map, r_to_l_filt)
            # joined_field = per._safe_map(rf_raw, r_to_l_map, r_to_l_filt)
            if right_writers is None:
                right_results.append(joined_field)
            else:
                right_writers[irf].data.write(joined_field)

        return right_results


    def merge_right(self, left_on, right_on,
                    left_fields=None, left_writers=None):
        l_key_raw = val.raw_array_from_parameter(self, 'left_on', left_on)
        l_index = np.arange(len(l_key_raw), dtype=np.int64)
        l_df = pd.DataFrame({'l_k': l_key_raw, 'l_index': l_index})

        r_key_raw = val.raw_array_from_parameter(self, 'right_on', right_on)
        r_index = np.arange(len(r_key_raw), dtype=np.int64)
        r_df = pd.DataFrame({'r_k': r_key_raw, 'r_index': r_index})

        df = pd.merge(left=r_df, right=l_df, left_on='r_k', right_on='l_k', how='left')
        l_to_r_map = df['l_index'].to_numpy(dtype='int64')
        l_to_r_filt = np.logical_not(df['l_index'].isnull()).to_numpy()

        left_results = list()
        for ilf, lf in enumerate(left_fields):
            lf_raw = val.raw_array_from_parameter(self, 'left_fields[{}]'.format(ilf), lf)
            joined_field = ops.safe_map(lf_raw, l_to_r_map, l_to_r_filt)
            if left_writers is None:
                left_results.append(joined_field)
            else:
                left_writers[ilf].data.write(joined_field)

        return left_results


    def merge_inner(self, left_on, right_on,
                    left_fields=None, left_writers=None, right_fields=None, right_writers=None):
        l_key_raw = val.raw_array_from_parameter(self, 'left_on', left_on)
        l_index = np.arange(len(l_key_raw), dtype=np.int64)
        l_df = pd.DataFrame({'l_k': l_key_raw, 'l_index': l_index})

        r_key_raw = val.raw_array_from_parameter(self, 'right_on', right_on)
        r_index = np.arange(len(r_key_raw), dtype=np.int64)
        r_df = pd.DataFrame({'r_k': r_key_raw, 'r_index': r_index})

        df = pd.merge(left=l_df, right=r_df, left_on='l_k', right_on='r_k', how='inner')
        l_to_i_map = df['l_index'].to_numpy(dtype='int64')
        l_to_i_filt = np.logical_not(df['l_index'].isnull()).to_numpy()
        r_to_i_map = df['r_index'].to_numpy(dtype='int64')
        r_to_i_filt = np.logical_not(df['r_index'].isnull()).to_numpy()

        left_results = list()
        for ilf, lf in enumerate(left_fields):
            lf_raw = val.raw_array_from_parameter(self, 'left_fields[{}]'.format(ilf), lf)
            joined_field = ops.safe_map(lf_raw, l_to_i_map, l_to_i_filt)
            if left_writers is None:
                left_results.append(joined_field)
            else:
                left_writers[ilf].data.write(joined_field)

        right_results = list()
        for irf, rf in enumerate(right_fields):
            rf_raw = val.raw_array_from_parameter(self, 'right_fields[{}]'.format(irf), rf)
            joined_field = ops.safe_map(rf_raw, r_to_i_map, r_to_i_filt)
            if right_writers is None:
                right_results.append(joined_field)
            else:
                right_writers[irf].data.write(joined_field)

        return left_results, right_results


    def _map_fields(self, field_map, field_sources, field_sinks):
        rtn_sinks = None
        if field_sinks is None:
            left_sinks = list()
            for src in field_sources:
                src_ = val.array_from_parameter(self, 'left_field_sources', src)
                snk_ = ops.map_valid(src_, field_map)
                left_sinks.append(snk_)
            rtn_sinks = left_sinks
        elif val.is_field_parameter(field_sinks[0]):
            # groups or fields
            for src, snk in zip(field_sources, field_sinks):
                src_ = val.array_from_parameter(self, 'left_field_sources', src)
                snk_ = ops.map_valid(src_, field_map)
                snk = val.field_from_parameter(self, 'left_field_sinks', snk)
                snk.data.write(snk_)
        else:
            # raw arrays
            for src, snk in zip(field_sources, field_sinks):
                src_ = val.array_from_parameter(self, 'left_field_sources', src)
                snk_ = val.array_from_parameter(self, 'left_field_sinks', snk)
                ops.map_valid(src_, field_map, snk_)
        return None if rtn_sinks is None else tuple(rtn_sinks)


    def _streaming_map_fields(self, field_map, field_sources, field_sinks):
        # field map must be a field
        # field sources must be fields
        # field sinks must be fields
        # field sources and sinks must be the same length
        map_ = val.field_from_parameter(self, 'field_map', field_map)
        for src, snk in zip(field_sources, field_sinks):
            src_ = val.field_from_parameter(self, 'field_sources', src)
            snk_ = val.field_from_parameter(self, 'field_sinks', snk)
            ops.ordered_map_valid_stream(src_, map_, snk_)


    def ordered_merge_left(self, left_on, right_on, right_field_sources=tuple(), left_field_sinks=None,
                           left_to_right_map=None, left_unique=False, right_unique=False):
        """
        Generate the results of a left join apply it to the fields described in the tuple
        'left_field_sources'. If 'left_field_sinks' is set, the mapped values are written
        to the fields / arrays set there.
        Note: in order to achieve best scalability, you should use groups / fields rather
        than numpy arrays and provide a tuple of groups/fields to left_field_sinks, so
        that the session and compute the merge and apply the mapping in a streaming
        fashion.
        :param left_on: the group/field/numba array that contains the left key values
        :param right_on: the group/field/numba array that contains the right key values
        :param left_to_right_map: a group/field/numba array that the map is written to. If
        it is a numba array, it must be the size of the resulting merge
        :param left_field_sources: a tuple of group/fields/numba arrays that contain the
        fields to be joined
        :param left_field_sinks: optional - a tuple of group/fields/numba arrays that
        the mapped fields should be written to
        :param left_unique: a hint to indicate whether the 'left_on' field contains unique
        values
        :param right_unique: a hint to indicate whether the 'right_on' field contains
        unique values
        :return: If left_field_sinks is not set, a tuple of the output fields is returned
        """
        if left_field_sinks is not None:
            if len(right_field_sources) != len(left_field_sinks):
                msg = ("{} and {} should be of the same length but are length {} and {} "
                       "respectively")
                raise ValueError(msg.format(len(right_field_sources), len(left_field_sinks)))
        val.all_same_basic_type('left_field_sources', right_field_sources)
        if left_field_sinks and len(left_field_sinks) > 0:
            val.all_same_basic_type('left_field_sinks', left_field_sinks)

        streamable = val.is_field_parameter(left_on) and \
                     val.is_field_parameter(right_on) and \
                     val.is_field_parameter(right_field_sources[0]) and \
                     left_field_sinks is not None and \
                     val.is_field_parameter(left_field_sinks[0]) and \
                     left_to_right_map is not None

        result = None
        has_unmapped = None
        if left_unique == False:
            if right_unique == False:
                raise ValueError("Right key must not have duplicates")
            else:
                if streamable:
                    has_unmapped = \
                        ops.ordered_map_to_right_right_unique_streamed(left_on, right_on,
                                                                      left_to_right_map)
                    result = left_to_right_map
                else:
                    result = np.zeros(len(left_on), dtype=np.int64)
                    left_data = val.array_from_parameter(self, "left_on", left_on)
                    right_data = val.array_from_parameter(self, "right_on", right_on)
                    has_unmapped = \
                        ops.ordered_map_to_right_right_unique(
                            left_data, right_data, result)
        else:
            if right_unique == False:
                raise ValueError("Right key must not have duplicates")
            else:
                result = np.zeros(len(left_on), dtype=np.int64)
                left_data = val.array_from_parameter(self, "left_on", left_on)
                right_data = val.array_from_parameter(self, "right_on", right_on)
                has_unmapped = ops.ordered_map_to_right_both_unique(
                    left_data, right_data, result)

        if streamable:
            self._streaming_map_fields(result, right_field_sources, left_field_sinks)
            return None
        else:
            rtn_left_sinks = self._map_fields(result, right_field_sources, left_field_sinks)
            return rtn_left_sinks


    def ordered_merge_right(self, left_on, right_on,
                            left_field_sources=tuple(), right_field_sinks=None,
                            right_to_left_map=None, left_unique=False, right_unique=False):
        """
        Generate the results of a right join apply it to the fields described in the tuple
        'right_field_sources'. If 'right_field_sinks' is set, the mapped values are written
        to the fields / arrays set there.
        Note: in order to achieve best scalability, you should use groups / fields rather
        than numpy arrays and provide a tuple of groups/fields to right_field_sinks, so
        that the session and compute the merge and apply the mapping in a streaming
        fashion.
        :param left_on: the group/field/numba array that contains the left key values
        :param right_on: the group/field/numba array that contains the right key values
        :param right_to_left_map: a group/field/numba array that the map is written to. If
        it is a numba array, it must be the size of the resulting merge
        :param right_field_sources: a tuple of group/fields/numba arrays that contain the
        fields to be joined
        :param right_field_sinks: optional - a tuple of group/fields/numba arrays that
        the mapped fields should be written to
        :param left_unique: a hint to indicate whether the 'left_on' field contains unique
        values
        :param right_unique: a hint to indicate whether the 'right_on' field contains
        unique values
        :return: If right_field_sinks is not set, a tuple of the output fields is returned
        """
        return self.ordered_merge_left(right_on, left_on, left_field_sources, right_field_sinks,
                                       right_to_left_map, right_unique, left_unique)


    def ordered_merge_inner(self, left_on, right_on,
                            left_field_sources=tuple(), left_field_sinks=None,
                            right_field_sources=tuple(), right_field_sinks=None,
                            left_unique=False, right_unique=False):

        if left_field_sinks is not None:
            if len(left_field_sources) != len(left_field_sinks):
                msg = ("{} and {} should be of the same length but are length {} and {} "
                       "respectively")
                raise ValueError(msg.format(len(left_field_sources), len(left_field_sinks)))
        val.all_same_basic_type('left_field_sources', left_field_sources)
        if left_field_sinks and len(left_field_sinks) > 0:
            val.all_same_basic_type('left_field_sinks', left_field_sinks)

        if right_field_sinks is not None:
            if len(right_field_sources) != len(right_field_sinks):
                msg = ("{} and {} should be of the same length but are length {} and {} "
                       "respectively")
                raise ValueError(msg.format(len(right_field_sources), len(right_field_sinks)))
        val.all_same_basic_type('right_field_sources', right_field_sources)
        if right_field_sinks and len(right_field_sinks) > 0:
            val.all_same_basic_type('right_field_sinks', right_field_sinks)

        left_data = val.array_from_parameter(self, 'left_on', left_on)
        right_data = val.array_from_parameter(self, 'right_on', right_on)

        result = None
        inner_length = ops.ordered_inner_map_result_size(left_data, right_data)

        left_to_inner = np.zeros(inner_length, dtype=np.int64)
        right_to_inner = np.zeros(inner_length, dtype=np.int64)
        if left_unique is False:
            if right_unique is False:
                ops.ordered_inner_map(left_data, right_data, left_to_inner, right_to_inner)
            else:
                ops.ordered_inner_map_left_unique(right_data, left_data, right_to_inner, left_to_inner)
        else:
            if right_unique is False:
                ops.ordered_inner_map_left_unique(left_data, right_data, left_to_inner, right_to_inner)
            else:
                ops.ordered_inner_map_both_unique(left_data, right_data, left_to_inner, right_to_inner)

        rtn_left_sinks = self._map_fields(left_to_inner, left_field_sources, left_field_sinks)
        rtn_right_sinks = self._map_fields(right_to_inner, right_field_sources, right_field_sinks)

        if rtn_left_sinks:
            if rtn_right_sinks:
                return rtn_left_sinks, rtn_right_sinks
            else:
                return rtn_left_sinks
        else:
            return rtn_right_sinks
