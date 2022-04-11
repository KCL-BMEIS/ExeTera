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

from typing import Dict, Optional, Union
import os
import io
from datetime import datetime, MAXYEAR
from itertools import accumulate
from io import StringIO

# from exetera.core import persistence as per
from exetera.io import load_schema, parsers
from exetera.core import utils


# options:
# . close the dataset once the data is imported
# . provide a field for the dataset name as an optional parameter
# . use dataset_name if it is a string, or 'importN' where N is the lowest
#   number that doesn't clash with an existing open dataset on the session

def import_with_schema(session: 'Session',
                       dataset_filename: Union[str, io.BytesIO],
                       dataset_alias: str,
                       schema_file: Union[str, io.BytesIO, io.StringIO],
                       files: Union[str, dict],
                       overwrite: bool,
                       include: Optional[Dict] = None,
                       exclude: Optional[Dict] = None,
                       timestamp: Union[str, datetime] = None,
                       chunk_row_size: int = 1 << 20):
    """
    Imports the source data described by 'files' into a dataset specified by 'dataset_name',
    with the session alias 'dataset_alias'. The source data described by 'files' must conform to
    the schema specified in 'schema_file'.

    If 'dataset_name' refers to an existing dataset, an error will be raised unless 'overwrite' is
    set to True, otherwise, 'overwrite' doesn't do anything.

    :param session: The exetera Session object used to hold the resulting open dataset
    :param dataset_filename: A relative or absolute path and name for the dataset. If this refers
    to an existing file, and the caller has not specified 'overwrite' to be True, an error will be
    raised. Otherwise, a dataset will be created at this location. This can also be a BytesIO
    object, primarily for testing purposes.
    :param dataset_alias: An alias for the dataset in the session. This is required so that the
    dataset can be easily retrieved from the session subsequently.
    :param schema_file: The path / name of an exetera schema file that describes the data in the
    data sources specified by 'files'.
    :param include: An optional parameter that specifies fields to be included from the data
    sources. Only one of 'include' or 'exclude' may be used for each data source.
    :param exclude: An optional parameter that specifies fields to be excluded from the data
    soures. Only one of 'include' or 'exclude' may be used for each data source.
    :param timestamp: An optional parameter the specifies an official timestamp for the dataset.
    If this is not set, a timestamp will be generated using at the moment this method is called.
    :param chunk_row_size: An optional parameter that tweaks the import performance. Larger values
    use more memory but improve import speed. Typically this should be left at its default value.
    """
 
    schema = load_schema.load_schema(schema_file)

    mode = 'w' if overwrite else 'r+'
    if isinstance(dataset_filename, str) and not os.path.exists(dataset_filename):
        mode = 'w'

    ds = session.open_dataset(dataset_filename, mode, dataset_alias)

    any_parts_present = False
    for sk in schema.keys():
        if sk in files:
            any_parts_present = True
    if not any_parts_present:
        raise ValueError("none of the data sources in 'files' contain relevant data to the schema")

    # check if there's any table from the include/exclude doesn't exist in the input files
    input_file_tables = set(files.keys())
    include_tables = set(include.keys()) if include is not None else None
    exclude_tables = set(exclude.keys()) if exclude is not None else None
    if include_tables and not include_tables.issubset(input_file_tables):
        extra_tables = include_tables.difference(input_file_tables)
        raise ValueError("-n/--include: the following include table(s) are not part of any input files: {}".format(extra_tables))

    if exclude_tables and not exclude_tables.issubset(input_file_tables):
        extra_tables = exclude_tables.difference(input_file_tables)
        raise ValueError("-x/--exclude: the following exclude table(s) are not part of any input files: {}".format(extra_tables))

    reserved_column_names = ('j_valid_from', 'j_valid_to')
    ts = utils.string_to_datetime(timestamp).timestamp()
        
    for sk in files.keys():
        schema_dict = schema[sk]
        
        for key in schema_dict.keys():
            if key in reserved_column_names:
                msg = "{} is a reserved column name: reserved names are {}"
                raise ValueError(msg.format(sk, reserved_column_names))

        csv_file = files[sk]

        include_fields = include.get(sk, None) if include is not None else None
        exclude_fields = exclude.get(sk, None) if exclude is not None else None

        ddf = ds.require_dataframe(sk)

        parsers.read_csv_with_schema_dict(csv_file, ddf, schema_dict, ts, include_fields, exclude_fields, chunk_row_size=chunk_row_size)
