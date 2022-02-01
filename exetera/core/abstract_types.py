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

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import numpy as np

class Field(ABC):

    @property
    @abstractmethod
    def valid(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def timestamp(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataframe(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def chunksize(self):
        raise NotImplementedError()

    @abstractmethod
    def writeable(self):
        raise NotImplementedError()

    @abstractmethod
    def create_like(self, group, name, timestamp=None):
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_sorted(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def indexed(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def data(self):
        raise NotImplementedError()

    @abstractmethod
    def __bool__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def get_spans(self):
        raise NotImplementedError()

    @abstractmethod
    def unique(self, return_index=False, return_inverse=False, return_counts=False):
        raise NotImplementedError()

class Dataset(ABC):
    """
    DataSet is a container of dataframes
    """

    @property
    @abstractmethod
    def session(self):
        raise NotImplementedError()

    @abstractmethod
    def create_dataframe(self,
                         name: str,
                         dataframe: 'DataFrame'):
        raise NotImplementedError()

    @abstractmethod
    def require_dataframe(self, name: str):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def copy(self,
             field: 'Field'):
        raise NotImplementedError()

    @abstractmethod
    def __contains__(self,
                     name: str):
        raise NotImplementedError()

    @abstractmethod
    def contains_dataframe(self,
                           dataframe: 'DataFrame'):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self,
                    name: str):
        raise NotImplementedError()

    @abstractmethod
    def get_dataframe(self,
                      name: str):
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self,
                    name: str,
                    dataframe: 'DataFrame'):
        raise NotImplementedError()

    @abstractmethod
    def __delitem__(self,
                    name: str):
        raise NotImplementedError()

    @abstractmethod
    def delete_dataframe(self,
                         dataframe: 'DataFrame'):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __next__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class DataFrame(ABC):
    """
    DataFrame is a table of data that contains a list of Fields (columns)
    """

    @property
    @abstractmethod
    def columns(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataset(self):
        raise NotImplementedError()

    @abstractmethod
    def add(self, field):
        raise NotImplementedError()

    @abstractmethod
    def drop(self, name: str):
        raise NotImplementedError()

    @abstractmethod
    def create_group(self, name):
        raise NotImplementedError()

    @abstractmethod
    def create_numeric(self, name, nformat, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_indexed_string(self, name, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_fixed_string(self, name, length, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_categorical(self, name, nformat, key, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_timestamp(self, name, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def __contains__(self, name):
        raise NotImplementedError()

    @abstractmethod
    def contains_field(self, field):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, name):
        raise NotImplementedError()

    @abstractmethod
    def get_field(self, name):
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, name, field):
        raise NotImplementedError()

    @abstractmethod
    def __delitem__(self, name):
        raise NotImplementedError()

    @abstractmethod
    def delete_field(self, field):
        raise NotImplementedError()

    @abstractmethod
    def keys(self):
        raise NotImplementedError()

    @abstractmethod
    def values(self):
        raise NotImplementedError()

    @abstractmethod
    def items(self):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __next__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def apply_filter(self, filter_to_apply, ddf=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_index(self, index_to_apply, ddf=None):
        raise NotImplementedError()

    @staticmethod
    def where(cond, a, b):
        raise NotImplementedError()


class DataFrameGroupBy(ABC):
    """
    DataFrameGroupBy is an object returned after group by on dataframe
    """
    @abstractmethod
    def max(self, field):
        raise NotImplementedError()

    @abstractmethod
    def min(self, field):
        raise NotImplementedError()

    @abstractmethod
    def first(self, field):
        raise NotImplementedError()

    @abstractmethod
    def last(self, field):
        raise NotImplementedError()


class AbstractSession(ABC):

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, etype, evalue, etraceback):
        raise NotImplementedError()

    @abstractmethod
    def open_dataset(self, dataset_path, mode, name):
        raise NotImplementedError()

    @abstractmethod
    def close_dataset(self, name):
        raise NotImplementedError()

    @abstractmethod
    def list_datasets(self):
        raise NotImplementedError()

    @abstractmethod
    def get_dataset(self, name):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def get_shared_index(self, keys):
        raise NotImplementedError()

    @abstractmethod
    def set_timestamp(self, timestamp=str(datetime.now(timezone.utc))):
        raise NotImplementedError()

    @abstractmethod
    def sort_on(self, src_group, dest_group, keys, timestamp,
                write_mode='write', verbose=True):
        raise NotImplementedError()

    @abstractmethod
    def dataset_sort_index(self, sort_indices, index=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_filter(self, filter_to_apply, src, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_index(self, index_to_apply, src, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def distinct(self, field=None, fields=None, filter=None):
        raise NotImplementedError()

    @abstractmethod
    def get_spans(self, field=None, fields=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_index_of_min(self, spans, target, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_index_of_max(self, spans, target, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_index_of_first(self, spans, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_index_of_last(self, spans, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_count(self, spans, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_min(self, spans, target, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_max(self, spans, target, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_first(self, spans, target, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_last(self, spans, target, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def apply_spans_concat(self, spans, target, dest,
                           src_chunksize=None, dest_chunksize=None, chunksize_mult=None):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_count(self, index, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_first(self, index, target=None, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_last(self, index, target=None, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_min(self, index, target=None, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_max(self, index, target=None, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_custom(self, predicate, index, target=None, dest=None):
        raise NotImplementedError()

    @abstractmethod
    def join(self, destination_pkey, fkey_indices, values_to_join,
             writer=None, fkey_index_spans=None):
        raise NotImplementedError()

    @abstractmethod
    def predicate_and_join(self, predicate, destination_pkey, fkey_indices,
                           reader=None, writer=None, fkey_index_spans=None):
        raise NotImplementedError()

    @abstractmethod
    def get(self, field):
        raise NotImplementedError()

    @abstractmethod
    def create_like(self, field, dest_group, dest_name, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_indexed_string(self, group, name, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_fixed_string(self, group, name, length, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_categorical(self, group, name, nformat, key, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_numeric(self, group, name, nformat, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def create_timestamp(self, group, name, timestamp=None, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def get_or_create_group(self, group, name):
        raise NotImplementedError()

    @abstractmethod
    def chunks(self, length, chunksize=None):
        raise NotImplementedError()

    @abstractmethod
    def get_index(self, target, foreign_key, destination=None):
        raise NotImplementedError()

    @abstractmethod
    def merge_left(self, left_on, right_on, right_fields=tuple(), right_writers=None):
        raise NotImplementedError()

    @abstractmethod
    def merge_right(self, left_on, right_on, left_fields=tuple(), left_writers=None):
        raise NotImplementedError()

    @abstractmethod
    def merge_inner(self, left_on, right_on,
                    left_fields=None, left_writers=None,
                    right_fields=None, right_writers=None):
        raise NotImplementedError()

    @abstractmethod
    def ordered_merge_left(self, left_on, right_on,
                           right_field_sources=tuple(), left_field_sinks=None,
                           left_to_right_map=None, left_unique=False, right_unique=False):
        raise NotImplementedError()

    @abstractmethod
    def ordered_merge_right(self, right_on, left_on,
                            left_field_sources=tuple(), right_field_sinks=None,
                            right_to_left_map=None, right_unique=False, left_unique=False):
        raise NotImplementedError()
