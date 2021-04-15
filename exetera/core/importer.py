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

import csv
from datetime import datetime, MAXYEAR
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
from exetera.core.csv_reader_speedup import file_read_line_fast_csv

def import_with_schema(timestamp, dest_file_name, schema_file, files, overwrite, include, exclude):

    print(timestamp)
    print(schema_file)
    print(files)

    with open(schema_file) as sf:
        schema = load_schema(sf)

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

    with h5py.File(dest_file_name, mode) as hf:
        for sk in schema.keys():
            if sk in reserved_column_names:
                msg = "{} is a reserved column name: reserved names are {}"
                raise ValueError(msg.format(sk, reserved_column_names))

            if sk not in files:
                continue

            fields = schema[sk].fields

            with open(files[sk]) as f:
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
            show_every = 100000

            DatasetImporter(datastore, files[sk], hf, sk, schema[sk], timestamp,
                            include=include, exclude=exclude,
                            stop_after=stop_after.get(sk, None),
                            show_progress_every=show_every)

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
                 include=None, exclude=None,
                 keys=None,
                 stop_after=None, show_progress_every=None, filter_fn=None,
                 early_filter=None):
        # self.names_ = list()
        self.index_ = None

        #stop_after = 2000000

        file_read_line_fast_csv(source)

        time0 = time.time()

        seen_ids = set()

        if space not in hf.keys():
            hf.create_group(space)
        group = hf[space]

        with open(source) as sf:
            csvf = csv.DictReader(sf, delimiter=',', quotechar='"')

            available_keys = [k.strip() for k in csvf.fieldnames if k.strip() in schema.fields]
            if space in include and len(include[space]) > 0:
                available_keys = include[space]
            if space in exclude and len(exclude[space]) > 0:
                available_keys = [k for k in available_keys if k not in exclude[space]]

            available_keys = ['ruc11cd','ruc11']
            #available_keys = ['ruc11']

            if not keys:
                fields_to_use = available_keys
                # index_map = [csvf.fieldnames.index(k) for k in fields_to_use]
                # index_map = [i for i in range(len(fields_to_use))]
            else:
                for k in keys:
                    if k not in available_keys:
                        raise ValueError(f"key '{k}' isn't in the available keys ({keys})")
                fields_to_use = keys
                # index_map = [csvf.fieldnames.index(k) for k in fields_to_use]

            csvf_fieldnames = [k.strip() for k in csvf.fieldnames]
            index_map = [csvf_fieldnames.index(k) for k in fields_to_use]

            early_key_index = None
            if early_filter is not None:
                if early_filter[0] not in available_keys:
                    raise ValueError(
                        f"'early_filter': tuple element zero must be a key that is in the dataset")
                early_key_index = available_keys.index(early_filter[0])

            chunk_size = 1 << 20
            new_fields = dict()
            new_field_list = list()
            field_chunk_list = list()
            categorical_map_list = list()
            longest_keys = list()

            # TODO: categorical writers should use the datatype specified in the schema
            for i_n in range(len(fields_to_use)):
                field_name = fields_to_use[i_n]
                sch = schema.fields[field_name]
                writer = sch.importer(datastore, group, field_name, timestamp)
                # TODO: this list is required because we convert the categorical values to
                # numerical values ahead of adding them. We could use importers that handle
                # that transform internally instead

                string_map = sch.strings_to_values

                byte_map = None
                if sch.out_of_range_label is None and string_map:
                    #byte_map = { key : string_map[key] for key in string_map.keys()  }

                    t = [np.fromstring(x, dtype=np.uint8) for x in string_map.keys()]
                    longest_key = len(max(t, key=len))

                    byte_map = np.zeros(longest_key * len(t) , dtype=np.uint8)
                    print('string_map', string_map)
                    print("longest_key", longest_key)

                    start_pos = 0
                    for x_id, x in enumerate(t):
                        for c_id, c in enumerate(x):
                            byte_map[start_pos + c_id] = c
                        start_pos += longest_key

                    print(byte_map)
                    

                    #for key in sorted(string_map.keys()):
                    #    byte_map.append(np.fromstring(key, dtype=np.uint8))

                    #byte_map = [np.fromstring(key, dtype=np.uint8) for key in sorted(string_map.keys())]
                    #byte_map.sort()

                longest_keys.append(longest_key)
                categorical_map_list.append(byte_map)

                new_fields[field_name] = writer
                new_field_list.append(writer)
                field_chunk_list.append(writer.chunk_factory(chunk_size))

        column_ids, column_vals = file_read_line_fast_csv(source)

        print(f"CSV read {time.time() - time0}s")

        chunk_index = 0

        key_to_search = np.fromstring('Urban city and twn', dtype=np.uint8)
        #print("key to search")
        #print(key_to_search)

        print(index_map)
        for ith, i_c in enumerate(index_map):
            chunk_index = 0

            if show_progress_every:
                if i_c % 1 == 0:
                    print(f"{i_c} cols parsed in {time.time() - time0}s")

            if early_filter is not None:
                if not early_filter[1](row[early_key_index]):
                    continue

            if i_c == stop_after:
                break

            categorical_map = None
            if len(categorical_map_list) > ith:
                categorical_map = categorical_map_list[ith]

            indices = column_ids[i_c]
            values = column_vals[i_c]

            @njit
            def findFirst_basic(a, b, div):
                for i in range(0, len(a), div):
                    #i = i*longest_key
                    result = True
                    for j in range(len(b)):
                        result = result and (a[i+j] == b[j])
                        if not result:
                            break
                    if result:
                        return i
                return 0

            @njit
            def map_values(chunk, indices, cat_map, div):
                #print(indices)
                size = 0
                for row_ix in range(len(indices) - 1):
                    temp_val = values[indices[row_ix] : indices[row_ix+1]]
                    internal_val = findFirst_basic(categorical_map, temp_val, div) // div
                    chunk[row_ix] = internal_val
                    size += 1
                return size

            #print("i_c", i_c, categorical_map)
            chunk = np.zeros(chunk_size, dtype=np.uint8)

            total = []

            # NEED TO NOT WRITE THE WHOLE CHUNK.. as the counter shows too many 0!

            chunk_index = 0
            while chunk_index < len(indices):
                size = map_values(chunk, indices[chunk_index:chunk_index+chunk_size], categorical_map, longest_keys[ith])

                data = chunk[:size]

                new_field_list[ith].write_part(data)
                total.extend(data)

                chunk_index += chunk_size

            print("idx", chunk_index)

            print("i_c", i_c, Counter(total))

            if chunk_index != 0:
                new_field_list[ith].write_part(chunk[:chunk_index])
                #total.extend(chunk[:chunk_index])


            for i_df in range(len(index_map)):
                new_field_list[i_df].flush()


        print(f"Total time {time.time() - time0}s")
        #exit()

    def __ainit__(self, datastore, source, hf, space, schema, timestamp,
                 include=None, exclude=None,
                 keys=None,
                 stop_after=None, show_progress_every=None, filter_fn=None,
                 early_filter=None):
        # self.names_ = list()
        self.index_ = None

        #stop_after = 2000000

        file_read_line_fast_csv(source)

        time0 = time.time()

        seen_ids = set()

        if space not in hf.keys():
            hf.create_group(space)
        group = hf[space]

        with open(source) as sf:
            csvf = csv.DictReader(sf, delimiter=',', quotechar='"')

            available_keys = [k.strip() for k in csvf.fieldnames if k.strip() in schema.fields]
            if space in include and len(include[space]) > 0:
                available_keys = include[space]
            if space in exclude and len(exclude[space]) > 0:
                available_keys = [k for k in available_keys if k not in exclude[space]]

            available_keys = ['ruc11cd','ruc11']

            if not keys:
                fields_to_use = available_keys
                # index_map = [csvf.fieldnames.index(k) for k in fields_to_use]
                # index_map = [i for i in range(len(fields_to_use))]
            else:
                for k in keys:
                    if k not in available_keys:
                        raise ValueError(f"key '{k}' isn't in the available keys ({keys})")
                fields_to_use = keys
                # index_map = [csvf.fieldnames.index(k) for k in fields_to_use]


            csvf_fieldnames = [k.strip() for k in csvf.fieldnames]
            index_map = [csvf_fieldnames.index(k) for k in fields_to_use]

            early_key_index = None
            if early_filter is not None:
                if early_filter[0] not in available_keys:
                    raise ValueError(
                        f"'early_filter': tuple element zero must be a key that is in the dataset")
                early_key_index = available_keys.index(early_filter[0])

            chunk_size = 1 << 20
            new_fields = dict()
            new_field_list = list()
            field_chunk_list = list()
            categorical_map_list = list()

            # TODO: categorical writers should use the datatype specified in the schema
            for i_n in range(len(fields_to_use)):
                field_name = fields_to_use[i_n]
                sch = schema.fields[field_name]
                writer = sch.importer(datastore, group, field_name, timestamp)
                # TODO: this list is required because we convert the categorical values to
                # numerical values ahead of adding them. We could use importers that handle
                # that transform internally instead

                string_map = sch.strings_to_values
                if sch.out_of_range_label is None and string_map:
                    byte_map = { str.encode(key) : string_map[key] for key in string_map.keys()  }
                else:
                    byte_map = None

                categorical_map_list.append(byte_map)

                new_fields[field_name] = writer
                new_field_list.append(writer)
                field_chunk_list.append(writer.chunk_factory(chunk_size))

        column_ids, column_vals = file_read_line_fast_csv(source)

        print(f"CSV read {time.time() - time0}s")

        chunk_index = 0

        for ith, i_c in enumerate(index_map):
            chunk_index = 0

            col = column_ids[i_c]

            if show_progress_every:
                if i_c % 1 == 0:
                    print(f"{i_c} cols parsed in {time.time() - time0}s")

            if early_filter is not None:
                if not early_filter[1](row[early_key_index]):
                    continue

            if i_c == stop_after:
                break

            categorical_map = None
            if len(categorical_map_list) > ith:
                categorical_map = categorical_map_list[ith]

            a = column_vals[i_c].copy()

            for row_ix in range(len(col) - 1):
                val = a[col[row_ix] : col[row_ix+1]].tobytes()

                if categorical_map is not None:
                    if val not in categorical_map:
                        #print(i_c, row_ix)
                        error = "'{}' not valid: must be one of {} for field '{}'"
                        raise KeyError(
                            error.format(val, categorical_map, available_keys[i_c]))
                    val = categorical_map[val]

                field_chunk_list[ith][chunk_index] = val

                chunk_index += 1

                if chunk_index == chunk_size:
                    new_field_list[ith].write_part(field_chunk_list[ith])

                    chunk_index = 0

            #print(f"Total time {time.time() - time0}s")

        if chunk_index != 0:
            for ith in range(len(index_map)):
                new_field_list[ith].write_part(field_chunk_list[ith][:chunk_index])

        for ith in range(len(index_map)):
            new_field_list[ith].flush()

        print(f"Total time {time.time() - time0}s")


    def __ainit__(self, datastore, source, hf, space, schema, timestamp,
                 include=None, exclude=None,
                 keys=None,
                 stop_after=None, show_progress_every=None, filter_fn=None,
                 early_filter=None):
        # self.names_ = list()
        self.index_ = None

        time0 = time.time()

        seen_ids = set()

        if space not in hf.keys():
            hf.create_group(space)
        group = hf[space]

        with open(source) as sf:
            csvf = csv.DictReader(sf, delimiter=',', quotechar='"')
            # self.names_ = csvf.fieldnames

            available_keys = [k.strip() for k in csvf.fieldnames if k.strip() in schema.fields]
            if space in include and len(include[space]) > 0:
                available_keys = include[space]
            if space in exclude and len(exclude[space]) > 0:
                available_keys = [k for k in available_keys if k not in exclude[space]]

            available_keys = ['ruc11']
            available_keys = ['ruc11cd','ruc11']

            # available_keys = csvf.fieldnames

            if not keys:
                fields_to_use = available_keys
                # index_map = [csvf.fieldnames.index(k) for k in fields_to_use]
                # index_map = [i for i in range(len(fields_to_use))]
            else:
                for k in keys:
                    if k not in available_keys:
                        raise ValueError(f"key '{k}' isn't in the available keys ({keys})")
                fields_to_use = keys
                # index_map = [csvf.fieldnames.index(k) for k in fields_to_use]

            csvf_fieldnames = [k.strip() for k in csvf.fieldnames]
            index_map = [csvf_fieldnames.index(k) for k in fields_to_use]

            early_key_index = None
            if early_filter is not None:
                if early_filter[0] not in available_keys:
                    raise ValueError(
                        f"'early_filter': tuple element zero must be a key that is in the dataset")
                early_key_index = available_keys.index(early_filter[0])

            chunk_size = 1 << 20
            new_fields = dict()
            new_field_list = list()
            field_chunk_list = list()
            categorical_map_list = list()

            # TODO: categorical writers should use the datatype specified in the schema
            for i_n in range(len(fields_to_use)):
                field_name = fields_to_use[i_n]
                sch = schema.fields[field_name]
                writer = sch.importer(datastore, group, field_name, timestamp)
                # TODO: this list is required because we convert the categorical values to
                # numerical values ahead of adding them. We could use importers that handle
                # that transform internally instead
                categorical_map_list.append(
                    sch.strings_to_values if sch.out_of_range_label is None else None)
                new_fields[field_name] = writer
                new_field_list.append(writer)
                field_chunk_list.append(writer.chunk_factory(chunk_size))

            csvf = csv.reader(sf, delimiter=',', quotechar='"')
            ecsvf = iter(csvf)

            chunk_index = 0
            try:
                total = []
                for i_r, row in enumerate(ecsvf):
                    if show_progress_every:
                        if i_r % show_progress_every == 0:
                            print(f"{i_r} rows parsed in {time.time() - time0}s")

                    if early_filter is not None:
                        if not early_filter[1](row[early_key_index]):
                            continue

                    if i_r == stop_after:
                        break

                    if not filter_fn or filter_fn(i_r):
                        for i_df, i_f in enumerate(index_map):
                            f = row[i_f]
                            categorical_map = categorical_map_list[i_df]
                            if categorical_map is not None:
                                if f not in categorical_map:
                                    error = "'{}' not valid: must be one of {} for field '{}'"
                                    raise KeyError(
                                        error.format(f, categorical_map, available_keys[i_f]))
                                f = categorical_map[f]
                            field_chunk_list[i_df][chunk_index] = f
                        chunk_index += 1
                        if chunk_index == chunk_size:
                            for i_df in range(len(index_map)):
                                # with utils.Timer("writing to {}".format(self.names_[i_df])):
                                #     new_field_list[i_df].write_part(field_chunk_list[i_df])
                                total.extend(field_chunk_list[i_df])

                                new_field_list[i_df].write_part(field_chunk_list[i_df])
                            chunk_index = 0

            except Exception as e:
                msg = "row {}: caught exception {}\nprevious row {}"
                print(msg.format(i_r + 1, e, row))
                raise

            if chunk_index != 0:
                for i_df in range(len(index_map)):
                    new_field_list[i_df].write_part(field_chunk_list[i_df][:chunk_index])
                    total.extend(field_chunk_list[i_df][:chunk_index])

            print("i_df", i_df, Counter(total))
            for i_df in range(len(index_map)):
                new_field_list[i_df].flush()

            print(f"{i_r} rows parsed in {time.time() - time0}s")

        print(f"Total time {time.time() - time0}s")

def get_cell(row, col, column_inds, column_vals):
    start_row_index = column_inds[col][row]
    end_row_index = column_inds[col][row+1]
    return column_vals[col][start_row_index:end_row_index].tobytes()
