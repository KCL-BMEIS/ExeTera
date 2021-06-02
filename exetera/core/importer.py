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
import csv
from datetime import datetime, MAXYEAR
from itertools import accumulate
import time

import numpy as np
import h5py
from numba import njit,jit, prange, vectorize, float64
from numba.typed import List

from collections import Counter
from exetera.core import csvdataset as dataset
from exetera.core import persistence as per
from exetera.core import utils
from exetera.core import operations as ops
from exetera.core.load_schema import load_schema
from exetera.core.csv_reader_speedup import read_file_using_fast_csv_reader
from io import StringIO

def import_with_schema(timestamp, dest_file_name, schema_file, files, overwrite, include, exclude, chunk_row_size = 1 << 20):

    print(timestamp)
    print(schema_file)
    print(files)

    schema = None
    if isinstance(schema_file, str):
        with open(schema_file, encoding='utf-8') as sf:
            schema = load_schema(sf)
    elif isinstance(schema_file, StringIO):
        schema = load_schema(schema_file)

    any_parts_present = False
    for sk in schema.keys():
        if sk in files:
            any_parts_present = True
    if not any_parts_present:
        raise ValueError("none of the data sources in 'files' contain relevant data to the schema")

    # check if there's any table from the include/exclude doesn't exist in the input files
    input_file_tables = set(files.keys())
    include_tables, exclude_tables = set(include.keys()), set(exclude.keys())
    if include_tables and not include_tables.issubset(input_file_tables):
        extra_tables = include_tables.difference(input_file_tables)
        raise ValueError("-n/--include: the following include table(s) are not part of any input files: {}".format(extra_tables))

    if exclude_tables and not exclude_tables.issubset(input_file_tables):
        extra_tables = exclude_tables.difference(input_file_tables)
        raise ValueError("-x/--exclude: the following exclude table(s) are not part of any input files: {}".format(extra_tables))

    stop_after = {}
    reserved_column_names = ('j_valid_from', 'j_valid_to')
    datastore = per.DataStore()

    if overwrite:
        mode = 'w'
    else:
        mode = 'r+'

    if isinstance(dest_file_name, str) and not os.path.exists(dest_file_name):
        mode = 'w'
        
    with h5py.File(dest_file_name, mode) as hf:
        for sk in schema.keys():
            if sk in reserved_column_names:
                msg = "{} is a reserved column name: reserved names are {}"
                raise ValueError(msg.format(sk, reserved_column_names))

            if sk not in files:
                continue

            fields = schema[sk].fields

            with open(files[sk], encoding='utf-8') as f:
                ds = dataset.Dataset(f, stop_after=1)
            names = set([n.strip() for n in ds.names_])
            missing_names = names.difference(fields.keys())
            if len(missing_names) > 0:
                msg = "The following fields are present in {} but not part of the schema: {}"
                print("Warning:", msg.format(files[sk], missing_names))
                # raise ValueError(msg.format(files[sk], missing_names))

            # check if included/exclude fields are in the file
            include_missing_names = set(include.get(sk, [])).difference(names)
            if len(include_missing_names) > 0:
                msg = "The following include fields are not part of the {}: {}"
                raise ValueError(msg.format(files[sk], include_missing_names))

            exclude_missing_names = set(exclude.get(sk, [])).difference(names)
            if len(exclude_missing_names) > 0:
                msg = "The following exclude fields are not part of the {}: {}"
                raise ValueError(msg.format(files[sk], exclude_missing_names))


        for sk in schema.keys():
            if sk not in files:
                continue

            fields = schema[sk].fields

            DatasetImporter(datastore, files[sk], hf, sk, schema[sk], timestamp,
                            include=include, exclude=exclude,
                            stop_after=stop_after.get(sk, None),
                            chunk_row_size=chunk_row_size)

            print(sk, hf.keys())
            table = hf[sk]
            ids = datastore.get_reader(table[list(table.keys())[0]])
            jvf = datastore.get_timestamp_writer(table, 'j_valid_from')
            ftimestamp = utils.string_to_datetime(timestamp).timestamp()
            valid_froms = np.full(len(ids), ftimestamp)
            jvf.write(valid_froms)
            jvt = datastore.get_timestamp_writer(table, 'j_valid_to')
            valid_tos = np.full(len(ids), ops.MAX_DATETIME.timestamp())
            jvt.write(valid_tos)

        print(hf.keys())


class DatasetImporter:
    def __init__(self, datastore, source, hf, space, schema, timestamp,
                 include=None, exclude=None, keys=None,
                 stop_after=None, chunk_row_size = (1 << 20)) :
                 

        if space not in hf.keys():
            hf.create_group(space)
        group = hf[space]

        with open(source, encoding='utf-8') as sf:
            csvf = csv.DictReader(sf, delimiter=',', quotechar='"')

            available_keys = [k.strip() for k in csvf.fieldnames if k.strip() in schema.fields]
            if space in include and len(include[space]) > 0:
                available_keys = include[space]
            if space in exclude and len(exclude[space]) > 0:
                available_keys = [k for k in available_keys if k not in exclude[space]]

            if not keys:
                fields_to_use = available_keys
            else:
                for k in keys:
                    if k not in available_keys:
                        raise ValueError(f"key '{k}' isn't in the available keys ({keys})")
                fields_to_use = keys

            csvf_fieldnames = [k.strip() for k in csvf.fieldnames]
            index_map = [csvf_fieldnames.index(k) for k in fields_to_use]


            field_importer_list = list() # only for field_to_use          
            for i_n in range(len(fields_to_use)):
                field_name = fields_to_use[i_n]
                sch = schema.fields[field_name]
                field_importer = sch.importer(datastore, group, field_name, timestamp)
                field_importer_list.append(field_importer)

            column_offsets = np.zeros(len(csvf_fieldnames) + 1, dtype=np.int64)
            for i, field_name in enumerate(csvf_fieldnames):
                sch = schema.fields[field_name]
                column_offsets[i + 1] = column_offsets[i] + sch.field_size * chunk_row_size
        
        read_file_using_fast_csv_reader(source, chunk_row_size, column_offsets, index_map, field_importer_list, stop_after_rows=stop_after)


