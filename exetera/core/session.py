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

from typing import Callable, IO, Optional, Tuple, Union
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
from exetera.core import utils


class Session(AbstractSession):
    """
    Session is the top-level object that is used to create and open ExeTera Datasets. It also
    provides operations that can be performed on Fields. For a more detailed explanation of
    Session and examples of its usage, please refer to
    https://github.com/KCL-BMEIS/ExeTera/wiki/Session-API
    
    :param chunksize: Change the default chunksize that fields created with this dataset use.
        Note this is a hint parameter and future versions of Session may choose to ignore it if it
        is no longer required. In general, it should only be changed for testing.
    :param timestamp: Set the official timestamp for the Session's creation rather than taking
        the current date/time.
    """

    def __init__(self,
                 chunksize: int = ops.DEFAULT_CHUNKSIZE,
                 timestamp: str = str(datetime.now(timezone.utc))):
        """
        Create a new Session object.
        """
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.chunksize = chunksize
        self.timestamp = timestamp
        self.datasets = dict()

    def __enter__(self):
        """Context manager enter."""
        return self

    def __exit__(self, etype, evalue, etraceback):
        """Context manager exit closes any open datasets."""
        self.close()

    def open_dataset(self,
                     dataset_path: Union[str, IO[bytes]],
                     mode: str,
                     name: str):
        """
        Open a dataset with the given access mode.
        
        :param dataset_path: the path to the dataset
        :param mode: the mode in which the dataset should be opened. This is one of "r", "r+" or "w".
        :param name: the name that is associated with this dataset. This can be used to retrieve the dataset when
            calling :py:meth:`~session.Session.get_dataset`.
        :return: The top-level dataset object
        """
        h5py_modes = {"r": "r", "r+": "r+", "w": "w"}
        if name in self.datasets:
            raise ValueError("A dataset with name '{}' is already open, and must be closed first.".format(name))

        self.datasets[name] = ds.HDF5Dataset(self, dataset_path, mode, name)
        return self.datasets[name]

    def close_dataset(self,
                      name: str):
        """
        Close the dataset with the given name. If there is no dataset with that name, do nothing.
        
        :param name: The name of the dataset to be closed
        :return: None
        """
        if name in self.datasets:
            self.datasets[name].close()
            del self.datasets[name]

    def list_datasets(self):
        """
        List the open datasets for this Session object. This is returned as a tuple of strings
        rather than the datasets themselves. The individual datasets can be fetched using
        :py:meth:`~exetera.session.Session.get_dataset`.
        
        Example::
        
            names = s.list_datasets()
            datasets = [s.get_dataset(n) for n in names]
            
        :return: A tuple containing the names of the currently open datasets for this Session object
        """
        return tuple(n for n in self.datasets.keys())

    def get_dataset(self,
                    name: str):
        """
        Get the dataset with the given name. If there is no dataset with that name, raise a KeyError
        indicating that the dataset with that name is not present.
        
        :param name: Name of the dataset to be fetched. This is the name that was given to it
            when it was opened through :py:meth:`~session.Session.open_dataset`.
        :return: Dataset with that name.
        """
        return self.datasets[name]

    def close(self):
        """
        Close all open datasets.
        
        :return: None
        """
        for v in self.datasets.values():
            v.close()
        self.datasets = dict()

    def get_shared_index(self, keys: Tuple[np.ndarray]):
        """
        Create a shared index based on a tuple of numpy arrays containing keys.
        This function generates the sorted union of a tuple of key fields and
        then maps the individual arrays to their corresponding indices in the
        sorted union.

        :param keys: a tuple of groups, fields or ndarrays whose contents represent keys

        Example::
        
            key_1 = ['a', 'b', 'e', 'g', 'i']
            key_2 = ['b', 'b', 'c', 'c, 'e', 'g', 'j']
            key_3 = ['a', 'c' 'd', 'e', 'g', 'h', 'h', 'i']
            
            sorted_union = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j']
            
            key_1_index = [0, 1, 4, 5, 7]
            key_2_index = [1, 1, 2, 2, 4, 5, 8]
            key_3_index = [0, 2, 3, 4, 5, 6, 6, 7]
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

    def set_timestamp(self,
                      timestamp: str = str(datetime.now(timezone.utc))):
        """
        Set the default timestamp to be used when creating fields without specifying
        an explicit timestamp.

        :param timestamp: a string representing a valid Datetime
        :return: None
        """
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.timestamp = timestamp

    def sort_on(self,
                src_group: h5py.Group,
                dest_group: h5py.Group,
                keys: Union[tuple, list],
                timestamp=datetime.now(timezone.utc), write_mode='write', verbose=True):
        """
        Sort a group (src_group) of fields by the specified set of keys, and write the
        sorted fields to dest_group.

        :param src_group: the group of fields that are to be sorted
        :param dest_group: the group into which sorted fields are written
        :param keys: fields to sort on
        :param timestamp: optional - timestamp to write on the sorted fields
        :param write_mode: optional - write mode to use if the destination fields already exist
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
                if r.indexed:
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
        index to apply to the sort_indices.
        
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
        if isinstance(src, Field):
            newfld = src.apply_filter(filter_to_apply_, writer_)
            if src.indexed:
                return newfld.indices[:], newfld.values[:]
            else:
                return newfld.data[:]
        # elif isinstance(src, df.datafrme):
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
        if isinstance(src, Field):
            newfld = src.apply_index(index_to_apply_, writer_)
            if src.indexed:
                return newfld.indices[:], newfld.values[:]
            else:
                return newfld.data[:]
        # if src.indexed:
        #     dest_indices, dest_values = \
        #         ops.apply_indices_to_index_values(index_to_apply_,
        #                                           src.indices[:], src.values[:])
        #     return dest_indices, dest_values
        # elif isinstance(src, Field):
        #     newfld = src.apply_index(index_to_apply_, writer_)
        #     return newfld.data[:]
        else:
            reader_ = val.array_from_parameter(self, 'reader', src)
            result = reader_[index_to_apply]
            if writer_:
                writer_.data.write(result)
            return result

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

    def get_spans(self, field: Union[Field, np.ndarray] = None,
                  dest: Field = None, **kwargs):
        """
        Calculate a set of spans that indicate contiguous equal values.
        The entries in the result array correspond to the inclusive start and
        exclusive end of the span (the ith span is represented by element i and
        element i+1 of the result array). The last entry of the result array is
        the length of the source field.

        Only one of 'field' or 'fields' may be set. If 'fields' is used and more
        than one field specified, the fields are effectively zipped and the check
        for spans is carried out on each corresponding tuple in the zipped field.

        Example::
        
            field: [1, 2, 2, 1, 1, 1, 3, 4, 4, 4, 2, 2, 2, 2, 2]
            result: [0, 1, 3, 6, 7, 10, 15]

        :param field: A Field or numpy array to be evaluated for spans
        :param dest: A destination Field to store the result
        :param \*\*kwargs: See below. For parameters set in both argument and kwargs, use kwargs

        :Keyword Arguments:
            * field -- Similar to field parameter, in case user specify field as keyword
            * fields -- A tuple of Fields or tuple of numpy arrays to be evaluated for spans
            * dest -- Similar to dest parameter, in case user specify as keyword

        :return: The resulting set of spans as a numpy array
        """
        fields = []
        result = None
        if len(kwargs) > 0:
            for k in kwargs.keys():
                if k == 'field':
                    field = kwargs[k]
                elif k == 'fields':
                    fields = kwargs[k]
                elif k == 'dest':
                    dest = kwargs[k]
        if dest is not None and not isinstance(dest, Field):
            raise TypeError(f"'dest' must be one of 'Field' but is {type(dest)}")

        if field is not None:
            if isinstance(field, Field):
                result = field.get_spans()
            elif isinstance(field, np.ndarray):
                result = ops.get_spans_for_field(field)
        elif len(fields) > 0:
            if isinstance(fields[0], Field):
                result = ops._get_spans_for_2_fields_by_spans(fields[0].get_spans(), fields[1].get_spans())
            elif isinstance(fields[0], np.ndarray):
                result = ops._get_spans_for_2_fields(fields[0], fields[1])
        else:
            raise ValueError("One of 'field' and 'fields' must be set")

        if dest is not None:
            dest.data.write(result)
            return dest
        else:
            return result

    def _apply_spans_no_src(self,
                            predicate: Callable[[np.ndarray, np.ndarray], None],
                            spans: np.ndarray,
                            dest: Field = None) -> np.ndarray:
        """
        An implementation method for span applications that are carried out on the spans themselves rather than a target
        field.
        
        :param predicate: a predicate function that carries out the operation on the spans and produces the result
        :param spans: the numpy array of spans to be applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        assert (dest is None or isinstance(dest, Field))

        if dest is not None:
            dest_f = val.field_from_parameter(self, 'dest', dest)
            results = np.zeros(len(spans) - 1, dtype=dest_f.data.dtype)
            predicate(spans, results)
            dest_f.data.write(results)
            return results
        else:
            results = np.zeros(len(spans) - 1, dtype=utils.guess_dtype(length=len(spans)))
            predicate(spans, results)
            return results

    def _apply_spans_src(self,
                         predicate: Callable[[np.ndarray, np.ndarray, np.ndarray], None],
                         spans: np.ndarray,
                         target: np.ndarray,
                         dest: Field = None) -> np.ndarray:
        """
        An implementation method for span applications that are carried out on a target field.
        
        :param predicate: a predicate function that carries out the operation on the spans and a target field, and
            produces the result
        :param spans: the numpy array of spans to be applied
        :param target: the field to which the spans and predicate are applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        assert (dest is None or isinstance(dest, Field))
        target_ = val.array_from_parameter(self, 'target', target)
        if len(target) != spans[-1]:
            error_msg = ("'target' (length {}) must be one element shorter than 'spans' "
                         "(length {})")
            raise ValueError(error_msg.format(len(target_), len(spans)))

        if dest is not None:
            dest_f = val.field_from_parameter(self, 'dest', dest)
            results = np.zeros(len(spans) - 1, dtype=dest_f.data.dtype)
            predicate(spans, target_, results)
            dest_f.data.write(results)
            return results
        else:
            if 'index' in predicate.__name__:
                data_type = utils.guess_dtype(length=len(spans))
            else:
                data_type = utils.guess_dtype(src_dtype=target_.dtype)
            results = np.zeros(len(spans) - 1, dtype=data_type)
            predicate(spans, target_, results)
            return results

    def apply_spans_index_of_min(self,
                                 spans: np.ndarray,
                                 target: np.ndarray,
                                 dest: Field = None):
        """
        Finds the index of the minimum value within each span on a target field.
        
        :param spans: the numpy array of spans to be applied
        :param target: the field to which the spans are applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_src(ops.apply_spans_index_of_min, spans, target, dest)

    def apply_spans_index_of_max(self,
                                 spans: np.ndarray,
                                 target: np.ndarray,
                                 dest: Field = None):
        """
        Finds the index of the maximum value within each span on a target field.
        
        :param spans: the numpy array of spans to be applied
        :param target: the field to which the spans are applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_src(ops.apply_spans_index_of_max, spans, target, dest)

    def apply_spans_index_of_first(self,
                                   spans: np.ndarray,
                                   dest: Field = None):
        """
        Finds the index of the first entry within each span.
        
        :param spans: the numpy array of spans to be applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_no_src(ops.apply_spans_index_of_first, spans, dest)

    def apply_spans_index_of_last(self,
                                  spans: np.ndarray,
                                  dest: Field = None):
        """
        Finds the index of the last entry within each span.
        
        :param spans: the numpy array of spans to be applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_no_src(ops.apply_spans_index_of_last, spans, dest)

    def apply_spans_count(self,
                          spans: np.ndarray,
                          dest: Field = None):
        """
        Finds the number of entries within each span.
        
        :param spans: the numpy array of spans to be applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_no_src(ops.apply_spans_count, spans, dest)

    def apply_spans_min(self,
                        spans: np.ndarray,
                        target: np.ndarray,
                        dest: Field = None):
        """
        Finds the minimum value within span on a target field.
        
        :param spans: the numpy array of spans to be applied
        :param target: the field to which the spans are applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_src(ops.apply_spans_min, spans, target, dest)

    def apply_spans_max(self,
                        spans: np.ndarray,
                        target: np.ndarray,
                        dest: Field = None):
        """
        Finds the maximum value within each span on a target field.
        
        :param spans: the numpy array of spans to be applied
        :param target: the field to which the spans are applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_src(ops.apply_spans_max, spans, target, dest)

    def apply_spans_first(self,
                          spans: np.ndarray,
                          target: np.ndarray,
                          dest: Field = None):
        """
        Finds the first entry within each span on a target field.
        
        :param spans: the numpy array of spans to be applied
        :param target: the field to which the spans are applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_src(ops.apply_spans_first, spans, target, dest)

    def apply_spans_last(self,
                         spans: np.ndarray,
                         target: np.ndarray,
                         dest: Field = None):
        """
        Finds the last entry within each span on a target field.
        
        :param spans: the numpy array of spans to be applied
        :param target: the field to which the spans are applied
        :param dest: if set, the field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        return self._apply_spans_src(ops.apply_spans_last, spans, target, dest)

    def apply_spans_concat(self,
                           spans,
                           target,
                           dest,
                           src_chunksize=None,
                           dest_chunksize=None,
                           chunksize_mult=None):
        if not target.indexed:
            raise ValueError(f"'target' must be one of 'IndexedStringField' but is {type(target)}")
        if not dest.indexed:
            raise ValueError(f"'dest' must be one of 'IndexedStringField' but is {type(dest)}")

        src_chunksize = target.chunksize if src_chunksize is None else src_chunksize
        dest_chunksize = dest.chunksize if dest_chunksize is None else dest_chunksize
        chunksize_mult = 16 if chunksize_mult is None else chunksize_mult

        src_index = target.indices[:]
        src_values = target.values[:]
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

    def _aggregate_impl(self, predicate, index, target=None, dest=None):
        """
        An implementation method for aggregation of fields via various predicates. This method takes a predicate that
        defines an operation to be carried out, an 'index' array or field that determines the groupings over which the
        predicate applies, and a 'target' array or field that the operation is carried out upon, should a target be
        needed. If a 'dest' Field is supplied, the results will be written to it.
        
        :param predicate: a predicate function that carries out the operation on the spans and produces the result
        :param index: A numpy array or field representing the sub-ranges that can be aggregated
        :param target: A numpy array upon which the operation is required. This only needs to be set for certain
            operations.
        :param dest: If set, the Field to which the results are written
        :returns: A numpy array containing the resulting values
        """
        index_ = val.raw_array_from_parameter(self, "index", index)

        dest_field = None
        if dest is not None:
            dest_field = val.field_from_parameter(self, "dest", dest)

        fkey_index_spans = self.get_spans(field=index)

        # execute the predicate (note that not every predicate requires a target)
        if target is None:
            results = predicate(fkey_index_spans, dest_field)
        else:
            results = predicate(fkey_index_spans, target, dest_field)

        return dest if dest is not None else results

    def aggregate_count(self, index, dest=None):
        """
         Finds the number of entries within each sub-group of index.
         
         Example::
         
         
             Index:  a a a b b x a c c d d d
             Result: 3     2   1 1 2   3
         
         :param index: A numpy array or Field containing the index that defines the ranges over which count is applied.
         :param dest: If set, a Field to which the resulting counts are written
         :returns: A numpy array containing the resulting values
         """
        return self._aggregate_impl(self.apply_spans_count, index, None, dest)

    def aggregate_first(self, index, target=None, dest=None):
        """
        Finds the first entries within each sub-group of index.
        
        Example:
        
            Index:  a a a b b x a c c d d d
            Target: 1 2 3 4 5 6 7 8 9 0 1 2
            Result: 1     4   6 7 8   0
        
        :param index: A numpy array or Field containing the index that defines the ranges over which count is applied.
        :param target: A numpy array to which the index and predicate are applied
        :param dest: If set, a Field to which the resulting counts are written
        :returns: A numpy array containing the resulting values
        """
        return self.aggregate_custom(self.apply_spans_first, index, target, dest)

    def aggregate_last(self, index, target=None, dest=None):
        """
        Finds the first entries within each sub-group of index.
        
        Example::
        
            Index:  a a a b b x a c c d d d
            Target: 1 2 3 4 5 6 7 8 9 0 1 2
            Result: 3     5   6 7 9   2
        
        :param index: A numpy array or Field containing the index that defines the ranges over which count is applied.
        :param target: A numpy array to which the index and predicate are applied
        :param dest: If set, a Field to which the resulting counts are written
        :returns: A numpy array containing the resulting values
        """
        return self.aggregate_custom(self.apply_spans_last, index, target, dest)

    def aggregate_min(self, index, target=None, dest=None):
        """
        Finds the minimum value within each sub-group of index.
        
        Example::
        
            Index:  a a a b b x a c c d d d
            Target: 1 2 3 5 4 6 7 8 9 2 1 0
            Result: 1     4   6 7 8   0
        
        :param index: A numpy array or Field containing the index that defines the ranges over which min is applied.
        :param target: A numpy array to which the index and predicate are applied
        :param dest: If set, a Field to which the resulting counts are written
        :returns: A numpy array containing the resulting values
        """
        return self.aggregate_custom(self.apply_spans_min, index, target, dest)

    def aggregate_max(self, index, target=None, dest=None):
        """
        Finds the maximum value within each sub-group of index.
        
        Example:
        
            Index:  a a a b b x a c c d d d
            Target: 1 2 3 5 4 6 7 8 9 2 1 0
            Result: 3     5   6 7 9   2
        
        :param index: A numpy array or Field containing the index that defines the ranges over which max is applied.
        :param target: A numpy array to which the index and predicate are applied
        :param dest: If set, a Field to which the resulting counts are written
        :returns: A numpy array containing the resulting values
        """
        return self.aggregate_custom(self.apply_spans_max, index, target, dest)

    def aggregate_custom(self, predicate, index, target=None, dest=None):
        if target is None:
            raise ValueError("'src' must not be None")
        val.ensure_valid_field_like("src", target)
        if dest is not None:
            val.ensure_valid_field("dest", dest)

        return self._aggregate_impl(predicate, index, target, dest)

    def join(self,
             destination_pkey, fkey_indices, values_to_join,
             writer=None, fkey_index_spans=None):
        """
        This method is due for removal and should not be used.
        Please use the merge or ordered_merge functions instead.
        """

        if isinstance(destination_pkey, Field) and destination_pkey.indexed:
            raise ValueError("'destination_pkey' must not be an indexed string field")
        if isinstance(fkey_indices, Field) and fkey_indices.indexed:
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
        # destination_space_values = writer.chunk_factory(len(destination_pkey))
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
        """
        This method is due for removal and should not be used.
        Please use the merge or ordered_merge functions instead.
        """
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
        results = np.zeros(len(fkey_index_spans) - 1, dtype=dtype)
        predicate(fkey_index_spans, reader, results)

        # the predicate results are in the same space as the unique_fkey_indices, which
        # means they may still contain invalid indices, so filter those now
        safe_results = results[invalid_filter]

        # now get the memory that the results will be mapped to
        destination_space_values = writer.chunk_factory(len(destination_pkey))
        # finally, map the results from the source space to the destination space
        destination_space_values[safe_unique_fkey_indices] = safe_results

        writer.write(destination_space_values)

    def get(self,
            field: Union[Field, h5py.Group]):
        """
        Get a Field from a h5py Group.
        
        Example::
        
            # this code for context
            with Session() as s:
    
              # open a dataset about wildlife
              src = s.open_dataset("/my/wildlife/dataset.hdf5", "r", "src")
    
              # fetch the group containing bird data
              birds = src['birds']
    
              # get the bird decibel field
              bird_decibels = s.get(birds['decibels'])
              
        :param field: The Field or Group object to retrieve.
        """
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
        return fieldtype_map[fieldtype](self, field, None, field.name)

    def create_like(self, field, dest_group, dest_name, timestamp=None, chunksize=None):
        """
        Create a field of the same type as an existing field, in the location and with the name provided.
        
        Example::
        
            with Session as s:
              ...
              a = s.get(table_1['a'])
              b = s.create_like(a, table_2, 'a_times_2')
              b.data.write(a.data[:] * 2)

        :param field: The Field whose type is to be copied
        :param dest_group: The group in which the new field should be created
        :param dest_name: The name of the new field
        """
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
        """
        Create an indexed string field in the given DataFrame with the given name.

        :param group: The group in which the new field should be created
        :param name: The name of the new field
        :param timestamp: If set, the timestamp that should be given to the new field. If not set
            datetime.now() is used.
        :param chunksize: If set, the chunksize that should be used to create the new field. In general, this should
            not be set unless you are writing unit tests.
        """
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            fld.indexed_string_field_constructor(self, group, name, timestamp, chunksize)
            return fld.IndexedStringField(self, group[name], None, write_enabled=True)
        else:
            return group.create_indexed_string(name, timestamp, chunksize)

    def create_fixed_string(self, group, name, length, timestamp=None, chunksize=None):
        """
        Create a fixed string field in the given DataFrame, given name, and given max string length per entry.

        :param group: The group in which the new field should be created
        :param name: The name of the new field
        :param length: The maximum length in bytes that each entry can have.
        :param timestamp: If set, the timestamp that should be given to the new field. If not set
            datetime.now() is used.
        :param chunksize: If set, the chunksize that should be used to create the new field. In general, this should
            not be set unless you are writing unit tests.
        """
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))
        if isinstance(group, h5py.Group):
            fld.fixed_string_field_constructor(self, group, name, length, timestamp, chunksize)
            return fld.FixedStringField(self, group[name], None, write_enabled=True)
        else:
            return group.create_fixed_string(name, length, timestamp, chunksize)

    def create_categorical(self, group, name, nformat, key, timestamp=None, chunksize=None):
        """
        Create a categorical field in the given DataFrame with the given name. This function also takes a numerical 
        format for the numeric representation of the categories, and a key that maps numeric values to their string
        string descriptions.

        :param group: The group in which the new field should be created
        :param name: The name of the new field
        :param nformat: A numerical type in the set (int8, uint8, int16, uint18, int32, uint32, int64). It is
            recommended to use 'int8'.
        :param key: A dictionary that maps numerical values to their string representations
        :param timestamp: If set, the timestamp that should be given to the new field. If not set
            datetime.now() is used.
        :param chunksize: If set, the chunksize that should be used to create the new field. In general, this should
            not be set unless you are writing unit tests.
        """
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            fld.categorical_field_constructor(self, group, name, nformat, key, timestamp, chunksize)
            return fld.CategoricalField(self, group[name], None, write_enabled=True)
        else:
            return group.create_categorical(name, nformat, key, timestamp, chunksize)

    def create_numeric(self, group, name, nformat, timestamp=None, chunksize=None):
        """
        Create a numeric field in the given DataFrame with the given name.

        :param group: The group in which the new field should be created
        :param name: The name of the new field
        :param nformat: A numerical type in the set (int8, uint8, int16, uint18, int32, uint32, int64, uint64,
            float32, float64). It is recommended to avoid uint64 as certain operations in numpy cause conversions to
            floating point values.
        :param timestamp: If set, the timestamp that should be given to the new field. If not set
            datetime.now() is used.
        :param chunksize: If set, the chunksize that should be used to create the new field. In general, this should
            not be set unless you are writing unit tests.
        """
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            fld.numeric_field_constructor(self, group, name, nformat, timestamp, chunksize)
            return fld.NumericField(self, group[name], None, write_enabled=True)
        else:
            return group.create_numeric(name, nformat, timestamp, chunksize)

    def create_timestamp(self, group, name, timestamp=None, chunksize=None):
        """
        Create a timestamp field in the given group with the given name.
        """
        if not isinstance(group, (df.DataFrame, h5py.Group)):
            if isinstance(group, ds.Dataset):
                raise ValueError("'group' must be an ExeTera DataFrame rather than a"
                                 " top-level Dataset")
            else:
                raise ValueError("'group' must be an Exetera DataFrame but a "
                                 "{} was passed to it".format(type(group)))

        if isinstance(group, h5py.Group):
            fld.timestamp_field_constructor(self, group, name, timestamp, chunksize)
            return fld.TimestampField(self, group[name], None, write_enabled=True)
        else:
            return group.create_timestamp(name, timestamp, chunksize)

    def get_or_create_group(self,
                            group: Union[h5py.Group, h5py.File],
                            name: str):
        """
        Note: this function is deprecated, and provided only for compatibility with existing scripts.
        It will be removed in a future version.


        """
        if name in group:
            return group[name]
        return group.create_group(name)

    def chunks(self,
               length: int,
               chunksize: Optional[int] = None):
        """
        Note: this function is deprecated, and provided only for compatibility with existing scripts.
        It will be removed in a future version.

        'chunks' is a convenience method that, given an overall length and a chunksize, will yield
        a set of ranges for the chunks in question.
        ie.
        chunks(1048576, 500000) -> (0, 500000), (500000, 1000000), (1000000, 1048576)

        :param length: The range to be split into chunks
        :param chunksize: Optional parameter detailing the size of each chunk. If not set, the
            chunksize that the Session was initialized with is used.
        """
        if chunksize is None:
            chunksize = self.chunksize
        cur = 0
        while cur < length:
            next = min(length, cur + chunksize)
            yield cur, next
            cur = next

    # def process(self,
    #             inputs,
    #             outputs,
    #             predicate):
    #     """
    #     Note: this function is deprecated, and provided only for compatibility with existing scripts.
    #     It will be removed in a future version.
    #     """
    #
    #     # TODO: modifying the dictionaries in place is not great
    #     input_readers = dict()
    #     for k, v in inputs.items():
    #         if isinstance(v, fld.Field):
    #             input_readers[k] = v
    #         else:
    #             input_readers[k] = self.get(v)
    #     output_writers = dict()
    #     output_arrays = dict()
    #     for k, v in outputs.items():
    #         if isinstance(v, fld.Field):
    #             output_writers[k] = v
    #         else:
    #             raise ValueError("'outputs': all values must be 'Writers'")
    #
    #     reader = next(iter(input_readers.values()))
    #     input_length = len(reader)
    #     writer = next(iter(output_writers.values()))
    #     chunksize = writer.chunksize
    #     required_chunksize = min(input_length, chunksize)
    #     for k, v in outputs.items():
    #         output_arrays[k] = output_writers[k].chunk_factory(required_chunksize)
    #
    #     for c in self.chunks(input_length, chunksize):
    #         kwargs = dict()
    #
    #         for k, v in inputs.items():
    #             kwargs[k] = v.data[c[0]:c[1]]
    #         for k, v in output_arrays.items():
    #             kwargs[k] = v.data[:c[1] - c[0]]
    #         predicate(**kwargs)
    #
    #         # TODO: write back to the writer
    #         for k in output_arrays.keys():
    #             output_writers[k].data.write_part(kwargs[k])
    #     for k, v in output_writers.items():
    #         output_writers[k].data.complete()

    def get_index(self, target, foreign_key, destination=None):
        """
        Note: this function is deprecated, and provided only for compatibility with existing scripts.
        It will be removed in a future version.

        Please make use of Dataframe.merge functionality instead. This method can be emulated by
        adding an index (via np.arange) to a dataframe, performing a merge and then fetching the
        mapped index field.

        'get_index' maps a primary key ('target') into the space of a foreign key ('foreign_key').
        """
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

    def temp_filename(self):
        uid = str(uuid.uuid4())
        while os.path.exists(uid + '.hdf5'):
            uid = str(uuid.uuid4())
        return uid + '.hdf5'

    def merge_left(self, left_on, right_on,
                   right_fields=tuple(), right_writers=None):
        """
        Note: this function is deprecated, and provided only for compatibility with existing scripts.
        It will be removed in a future version.

        Please use DataFrame.merge instead.

        Perform a database-style left join on right_fields, outputting the result to right_writers, if set.
        
        :param left_on: The key to perform the join on on the left hand side
        :param right_on: The key to perform the join on on the right hand side
        :param right_fields: The fields to be mapped from right to left
        :param right_writers: Optional parameter providing the fields to which the mapped data should
            be written. If this is not set, the mapped data is returned as numpy arrays and lists instead.
        """
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
            if isinstance(rf, Field):
                if rf.indexed:
                    indices, values = ops.safe_map_indexed_values(rf.indices[:], rf.values[:],
                                                                  r_to_l_map, r_to_l_filt)
                    if right_writers is None:
                        result = fld.IndexedStringMemField(self)
                        result.indices.write(indices)
                        result.values.write(values)
                        right_results.append(result)
                    else:
                        right_writers[irf].indices.write(indices)
                        right_writers[irf].values.write(values)
                else:
                    values = ops.safe_map_values(rf.data[:], r_to_l_map, r_to_l_filt)
                    if right_writers is None:
                        result = rf.create_like()
                        result.data.write(values)
                        right_results.append(result)
                    else:
                        right_writers[irf].data.write(values)
            else:
                values = ops.safe_map_values(rf, r_to_l_map, r_to_l_filt)

                if right_writers is None:
                    right_results.append(values)
                else:
                    right_writers[irf].data.write(values)

        return right_results

    def merge_right(self, left_on, right_on,
                    left_fields=tuple(), left_writers=None):
        """
        Note: this function is deprecated, and provided only for compatibility with existing scripts.
        It will be removed in a future version.

        Please use DataFrame.merge instead.

        Perform a database-style right join on left_fields, outputting the result to left_writers, if set.
        
        :param left_on: The key to perform the join on on the left hand side
        :param right_on: The key to perform the join on on the right hand side
        :param left_fields: The fields to be mapped from right to left
        :param left_writers: Optional parameter providing the fields to which the mapped data should
            be written. If this is not set, the mapped data is returned as numpy arrays and lists instead.
        """

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
            if isinstance(lf, Field):
                if lf.indexed:
                    indices, values = ops.safe_map_indexed_values(lf.indices[:], lf.values[:],
                                                                  l_to_r_map, l_to_r_filt)
                    if left_writers is None:
                        result = fld.IndexedStringMemField(self)
                        result.indices.write(indices)
                        result.values.write(values)
                        left_results.append(result)
                    else:
                        left_writers[ilf].indices.write(indices)
                        left_writers[ilf].values.write(values)
                else:
                    values = ops.safe_map_values(lf.data[:], l_to_r_map, l_to_r_filt)
                    if left_writers is None:
                        result = lf.create_like()
                        result.data.write(values)
                        left_results.append(result)
                    else:
                        left_writers[ilf].data.write(values)
            else:
                values = ops.safe_map_values(lf, l_to_r_map, l_to_r_filt)

                if left_writers is None:
                    left_results.append(values)
                else:
                    left_writers[ilf].data.write(values)

        return left_results

    def merge_inner(self, left_on, right_on,
                    left_fields=None, left_writers=None, right_fields=None, right_writers=None):
        """
        Note: this function is deprecated, and provided only for compatibility with existing scripts.
        It will be removed in a future version.
        
        Please use DataFrame.merge instead.

        Perform a database-style inner join on left_fields, outputting the result to left_writers, if set.
        
        :param left_on: The key to perform the join on on the left hand side
        :param right_on: The key to perform the join on on the right hand side
        :param left_fields: The fields to be mapped from left to inner
        :param left_writers: Optional parameter providing the fields to which the mapped data should
            be written. If this is not set, the mapped data is returned as numpy arrays and lists instead.
        :param right_fields: The fields to be mapped from right to inner
        :param right_writers: Optional parameter providing the fields to which the mapped data should
            be written. If this is not set, the mapped data is returned as numpy arrays and lists instead.
        """
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
            if isinstance(lf, Field):
                if lf.indexed:
                    indices, values = ops.safe_map_indexed_values(lf.indices[:], lf.values[:],
                                                                  l_to_i_map, l_to_i_filt)
                    if left_writers is None:
                        result = fld.IndexedStringMemField(self)
                        result.indices.write(indices)
                        result.values.write(values)
                        left_results.append(result)
                    else:
                        left_writers[ilf].indices.write(indices)
                        left_writers[ilf].values.write(values)
                else:
                    values = ops.safe_map_values(lf.data[:], l_to_i_map, l_to_i_filt)
                    if left_writers is None:
                        result = lf.create_like()
                        result.data.write(values)
                        left_results.append(result)
                    else:
                        left_writers[ilf].data.write(values)
            else:
                values = ops.safe_map_values(lf, l_to_i_map, l_to_i_filt)

                if left_writers is None:
                    left_results.append(values)
                else:
                    left_writers[ilf].data.write(values)

        right_results = list()
        for irf, rf in enumerate(right_fields):
            if isinstance(rf, Field):
                if rf.indexed:
                    indices, values = ops.safe_map_indexed_values(rf.indices[:], rf.values[:],
                                                                  r_to_i_map, r_to_i_filt)
                    if right_writers is None:
                        result = fld.IndexedStringMemField(self)
                        result.indices.write(indices)
                        result.values.write(values)
                        right_results.append(result)
                    else:
                        right_writers[irf].indices.write(indices)
                        right_writers[irf].values.write(values)
                else:
                    values = ops.safe_map_values(rf.data[:], r_to_i_map, r_to_i_filt)
                    if right_writers is None:
                        result = rf.create_like()
                        result.data.write(values)
                        right_results.append(result)
                    else:
                        right_writers[irf].data.write(values)
            else:
                values = ops.safe_map_values(rf, r_to_i_map, r_to_i_filt)

                if right_writers is None:
                    right_results.append(values)
                else:
                    right_writers[irf].data.write(values)

        return left_results, right_results

    def _map_fields(self, field_map, field_sources, field_sinks, invalid):
        rtn_sinks = None
        if field_sinks is None:
            left_sinks = list()
            for src in field_sources:
                src_ = val.array_from_parameter(self, 'left_field_sources', src)
                snk_ = ops.map_valid(src_, field_map, invalid=invalid)
                left_sinks.append(snk_)
            rtn_sinks = left_sinks
        elif val.is_field_parameter(field_sinks[0]):
            # groups or fields
            for src, snk in zip(field_sources, field_sinks):
                src_ = val.array_from_parameter(self, 'left_field_sources', src)
                snk_ = ops.map_valid(src_, field_map, invalid=invalid)
                snk = val.field_from_parameter(self, 'left_field_sinks', snk)
                snk.data.write(snk_)
        else:
            # raw arrays
            for src, snk in zip(field_sources, field_sinks):
                src_ = val.array_from_parameter(self, 'left_field_sources', src)
                snk_ = val.array_from_parameter(self, 'left_field_sinks', snk)
                ops.map_valid(src_, field_map, snk_, invalid=invalid)
        return None if rtn_sinks is None else tuple(rtn_sinks)

    def _streaming_map_fields(self, field_map, field_sources, field_sinks, invalid=-1):
        # field map must be a field
        # field sources must be fields
        # field sinks must be fields
        # field sources and sinks must be the same length
        map_ = val.field_from_parameter(self, 'field_map', field_map)
        for src, snk in zip(field_sources, field_sinks):
            src_ = val.field_from_parameter(self, 'field_sources', src)
            snk_ = val.field_from_parameter(self, 'field_sinks', snk)
            ops.ordered_map_valid_stream_old(src_, map_, snk_, invalid=invalid)

    def ordered_merge_left(self, left_on, right_on, right_field_sources=tuple(), left_field_sinks=None,
                           left_to_right_map=None, left_unique=False, right_unique=False):
        """
        Generate the results of a left join and apply it to the fields described in the tuple
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
        :param left_field_sources: a tuple of group/fields/numba arrays that contain the fields to be joined
        :param left_field_sinks: optional - a tuple of group/fields/numba arrays that
            the mapped fields should be written to
        :param left_unique: a hint to indicate whether the 'left_on' field contains unique values
        :param right_unique: a hint to indicate whether the 'right_on' field contains unique values
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
                        ops.generate_ordered_map_to_left_right_unique_streamed_old(left_on, right_on,
                                                                                   left_to_right_map,
                                                                                   ops.INVALID_INDEX)
                    result = left_to_right_map
                else:
                    result = np.zeros(len(left_on), dtype=np.int64)
                    left_data = val.array_from_parameter(self, "left_on", left_on)
                    right_data = val.array_from_parameter(self, "right_on", right_on)
                    has_unmapped = \
                        ops.generate_ordered_map_to_left_right_unique(
                            left_data, right_data, result, ops.INVALID_INDEX)
        else:
            if right_unique == False:
                raise ValueError("Right key must not have duplicates")
            else:
                result = np.zeros(len(left_on), dtype=np.int64)
                left_data = val.array_from_parameter(self, "left_on", left_on)
                right_data = val.array_from_parameter(self, "right_on", right_on)
                has_unmapped = ops.generate_ordered_map_to_left_both_unique(
                    left_data, right_data, result, ops.INVALID_INDEX)

        if streamable:
            self._streaming_map_fields(result, right_field_sources, left_field_sinks,
                                       invalid=ops.INVALID_INDEX)
            return None
        else:
            rtn_left_sinks = self._map_fields(result, right_field_sources, left_field_sinks,
                                              ops.INVALID_INDEX)
            return rtn_left_sinks

    def ordered_merge_right(self, left_on, right_on,
                            left_field_sources=tuple(), right_field_sinks=None,
                            right_to_left_map=None, left_unique=False, right_unique=False):
        """
        Generate the results of a right join and apply it to the fields described in the tuple
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
        :param right_field_sources: a tuple of group/fields/numba arrays that contain the fields to be joined
        :param right_field_sinks: optional - a tuple of group/fields/numba arrays that
            the mapped fields should be written to
        :param left_unique: a hint to indicate whether the 'left_on' field contains unique values
        :param right_unique: a hint to indicate whether the 'right_on' field contains unique values
        :return: If right_field_sinks is not set, a tuple of the output fields is returned
        """
        return self.ordered_merge_left(right_on, left_on, left_field_sources, right_field_sinks,
                                       right_to_left_map, right_unique, left_unique)

    def ordered_merge_inner(self, left_on, right_on,
                            left_field_sources=tuple(), left_field_sinks=None,
                            right_field_sources=tuple(), right_field_sinks=None,
                            left_unique=False, right_unique=False):
        """
        Generate the results of an inner join and apply it to the fields described in the tuple
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
        :param right_field_sources: a tuple of group/fields/numba arrays that contain the fields to be joined
        :param right_field_sinks: optional - a tuple of group/fields/numba arrays that
            the mapped fields should be written to
        :param left_unique: a hint to indicate whether the 'left_on' field contains unique values
        :param right_unique: a hint to indicate whether the 'right_on' field contains unique values
        :return: If right_field_sinks is not set, a tuple of the output fields is returned
        """
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

        rtn_left_sinks = self._map_fields(left_to_inner, left_field_sources, left_field_sinks,
                                          invalid=ops.INVALID_INDEX)
        rtn_right_sinks = self._map_fields(right_to_inner, right_field_sources, right_field_sinks,
                                           invalid=ops.INVALID_INDEX)

        if rtn_left_sinks:
            if rtn_right_sinks:
                return rtn_left_sinks, rtn_right_sinks
            else:
                return rtn_left_sinks
        else:
            return rtn_right_sinks
