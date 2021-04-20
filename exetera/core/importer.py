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

        old = False
        if old:
            self.old(datastore, source, hf, space, schema, timestamp,
                 include, exclude,
                 keys,
                 stop_after, show_progress_every, filter_fn,
                 early_filter)
        else:
            self.nnnn(datastore, source, hf, space, schema, timestamp,
                 include, exclude,
                 keys,
                 stop_after, show_progress_every, filter_fn,
                 early_filter)

    def nnnn(self, datastore, source, hf, space, schema, timestamp,
                 include=None, exclude=None,
                 keys=None,
                 stop_after=None, show_progress_every=None, filter_fn=None,
                 early_filter=None):
        # self.names_ = list()
        self.index_ = None

        #stop_after = 2000000

        file_read_line_fast_csv(source)
        #exit()

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
                    # sort by length of key first, and then sort alphabetically
                    sorted_string_map = {k: v for k, v in sorted(string_map.items(), key=lambda item: (len(item[0]), item[0]))}
                    sorted_string_key = [(len(k), np.frombuffer(k.encode(), dtype=np.uint8), v) for k, v in sorted_string_map.items()]
                    sorted_string_values = list(sorted_string_map.values())
                    
                    # assign byte_map_key_lengths, byte_map_value
                    byte_map_key_lengths = np.zeros(len(sorted_string_map), dtype=np.uint8)
                    byte_map_value = np.zeros(len(sorted_string_map), dtype=np.uint8)

                    for i, (length, _, v)  in enumerate(sorted_string_key):
                        byte_map_key_lengths[i] = length
                        byte_map_value[i] = v

                    # assign byte_map_keys, byte_map_key_indices
                    byte_map_keys = np.zeros(sum(byte_map_key_lengths), dtype=np.uint8)
                    byte_map_key_indices = np.zeros(len(sorted_string_map)+1, dtype=np.uint8)
                    
                    idx_pointer = 0
                    for i, (_, b_key, _) in enumerate(sorted_string_key):   
                        for b in b_key:
                            byte_map_keys[idx_pointer] = b
                            idx_pointer += 1

                        byte_map_key_indices[i + 1] = idx_pointer  


                    byte_map = [byte_map_keys, byte_map_key_lengths, byte_map_key_indices, byte_map_value]

                categorical_map_list.append(byte_map)


                new_fields[field_name] = writer
                new_field_list.append(writer)
                field_chunk_list.append(writer.chunk_factory(chunk_size))

        column_ids, column_vals = file_read_line_fast_csv(source)

        print(f"CSV read {time.time() - time0}s")

        chunk_index = 0

        total_col = []

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
                cat_keys, cat_key_len, cat_index, cat_values = categorical_map_list[ith]

            @njit           
            def my_fast_categorical_mapper(chunk, chunk_index, chunk_size, cat_keys, cat_key_len, cat_index, cat_values):
                error_row_idx = -1
                for row_idx in range(chunk_size):
                    # Finds length, which we use to lookup potential matches
                    key_start = column_ids[i_c, chunk_index + row_idx]
                    key_end = column_ids[i_c, chunk_index + row_idx + 1]
                    key_len = key_end - key_start
                                        
                    # start_idx = np.searchsorted(cat_key_len, key_len, "left")
                    # stop_idx = np.searchsorted(cat_key_len, key_len, "right")

                    # print('key_start', key_start, 'key_end', key_end)
                    # print('start_idx', start_idx, 'stop_idx', stop_idx)

                    for i in range(len(cat_index) - 1):
                        sc_key_len = cat_index[i + 1] - cat_index[i]

                        if key_len != sc_key_len:
                            continue

                        index = i
                        for j in range(key_len):
                            entry_start = cat_index[i]
                            if column_vals[i_c, key_start + j] != cat_keys[entry_start + j]:
                                index = -1
                                break

                        if index != -1:
                            chunk[row_idx] = cat_values[index]

                return error_row_idx

            
            total = []
            chunk_index = 0
            indices_len = len(column_ids[i_c])

            # print('@@@@@')
            # print('column_ids', 'i_c', i_c, column_ids)
            # print('column_vals', 'i_c', i_c, column_vals)
            # print('@@@@@')
            while chunk_index < indices_len:
                if chunk_index + chunk_size > indices_len:
                    chunk_size = indices_len - chunk_index 

                #print('chunk_size', chunk_size)

                chunk = np.zeros(chunk_size, dtype=np.uint8)
                
                my_fast_categorical_mapper(chunk, chunk_index, chunk_size, cat_keys, cat_key_len, cat_index, cat_values)

                new_field_list[ith].write_part(chunk)
                total.extend(chunk)
                chunk_index += chunk_size

            total_col.append(total)

            print("i_c", i_c, Counter(total))


            if chunk_index != 0:
                new_field_list[ith].write_part(chunk[:chunk_index])
                #total.extend(chunk[:chunk_index])


            for i_df in range(len(index_map)):
                new_field_list[i_df].flush()
    

        print(f"Total time {time.time() - time0}s")
        #exit()

    
    def old(self, datastore, source, hf, space, schema, timestamp,
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

            available_keys = ['ruc11cd']
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
                total = [[],[]]
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
                                total[i_df].extend(field_chunk_list[i_df])

                                new_field_list[i_df].write_part(field_chunk_list[i_df])
                            chunk_index = 0

            except Exception as e:
                msg = "row {}: caught exception {}\nprevious row {}"
                print(msg.format(i_r + 1, e, row))

                
                raise

            if chunk_index != 0:
                for i_df in range(len(index_map)):
                    new_field_list[i_df].write_part(field_chunk_list[i_df][:chunk_index])
                    total[i_df].extend(field_chunk_list[i_df][:chunk_index])

                    print("i_df", i_df, Counter(total[i_df]))
                    
                print('ruc == ruc11cd', total[0] == total[1])
                
            for i_df in range(len(index_map)):
                new_field_list[i_df].flush()

            print(f"{i_r} rows parsed in {time.time() - time0}s")

        print(f"Total time {time.time() - time0}s")

def get_cell(row, col, column_inds, column_vals):
    start_row_index = column_inds[col][row]
    end_row_index = column_inds[col][row+1]
    return column_vals[col][start_row_index:end_row_index].tobytes()
