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
from datetime import datetime, MAXYEAR
from itertools import accumulate
from io import StringIO

from exetera.core import persistence as per
from exetera.io import load_schema, csv_parser
from exetera.core import utils


def import_with_schema(session, timestamp, dataset_name, dest_file_name, schema_file, files, overwrite, include=None, exclude=None, chunk_row_size = 1 << 20):
 
    schema = load_schema.load_schema(schema_file)

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

    mode = 'w' if overwrite else 'r+'
    if isinstance(dest_file_name, str) and not os.path.exists(dest_file_name):
        mode = 'w'

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

        ds = session.open_dataset(dest_file_name, mode, dataset_name)
        ddf = ds.require_dataframe(sk) 

        csv_parser.read_csv_with_schema_dict(csv_file, ddf, schema_dict, ts, include_fields, exclude_fields, chunk_row_size=chunk_row_size)



 