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
import time


class DatasetImporter:
    def __init__(self, datastore, source, hf, space,
                 writer_factory, writers, field_entries, timestamp,
                 keys=None, # field_descriptors=None,
                 stop_after=None, show_progress_every=None, filter_fn=None,
                 early_filter=None):
        self.names_ = list()
        self.index_ = None

        time0 = time.time()

        seen_ids = set()

        # keys = ('id', 'created_at', 'updated_at')
        if space not in hf.keys():
            hf.create_group(space)
        group = hf[space]

        with open(source) as sf:
            csvf = csv.DictReader(sf, delimiter=',', quotechar='"')
            self.names_ = csvf.fieldnames

            available_keys = csvf.fieldnames
            if not keys:
                fields_to_use = available_keys
                index_map = [i for i in range(len(fields_to_use))]
            else:
                for k in keys:
                    if k not in available_keys:
                        raise ValueError(f"key '{k}' isn't in the available keys ({keys})")
                fields_to_use = keys
                index_map = [available_keys.index(k) for k in keys]


            transforms_by_index = list()
            for i_n, n in enumerate(available_keys):
                if field_entries and n in field_entries:
                    transforms_by_index.append(field_entries[n])
                else:
                    transforms_by_index.append(None)

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
                if writers[field_name] == 'categoricaltype':
                    str_to_vals = field_entries[field_name].strings_to_values
                    writer =\
                        writer_factory[writers[field_name]](
                            datastore, group, field_name, str_to_vals, timestamp)
                    categorical_map_list.append(str_to_vals)
                elif writers[field_name] == 'leakycategoricaltype':
                    str_to_vals = field_entries[field_name].strings_to_values
                    out_of_range = field_entries[field_name].out_of_range_label
                    writer =\
                        writer_factory[writers[field_name]](
                            datastore, group, field_name, str_to_vals, out_of_range, timestamp)
                    # categorical_map_list.append(str_to_vals)
                    categorical_map_list.append(None)
                else:
                    writer = writer_factory[writers[field_name]](
                        datastore, group, field_name, timestamp)
                    categorical_map_list.append(None)
                new_fields[field_name] = writer
                new_field_list.append(writer)
                field_chunk_list.append(writer.chunk_factory(chunk_size))

            csvf = csv.reader(sf, delimiter=',', quotechar='"')
            ecsvf = iter(csvf)

            chunk_index = 0
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
                                    error.format(f, categorical_map, self.names_[i_f]))
                            f = categorical_map[f]
                        # t = transforms_by_index[i_f]
                        field_chunk_list[i_df][chunk_index] = f
                        # new_field_list[i_df].append(f)
                    chunk_index += 1
                    if chunk_index == chunk_size:
                        for i_df in range(len(index_map)):
                            new_field_list[i_df].write_part(field_chunk_list[i_df])
                        chunk_index = 0

            if chunk_index != 0:
                for i_df in range(len(index_map)):
                    new_field_list[i_df].write_part(field_chunk_list[i_df][:chunk_index])

            for i_df in range(len(index_map)):
                new_field_list[i_df].flush()


from hystore.core import session, fields

class NewDatasetImporter:
    def __init__(self, datastore, source, hf, space,
                 writer_factory, writers, field_entries, timestamp,
                 keys=None, # field_descriptors=None,
                 stop_after=None, show_progress_every=None, filter_fn=None,
                 early_filter=None):
        self.names_ = list()
        self.index_ = None

        time0 = time.time()

        seen_ids = set()

        # keys = ('id', 'created_at', 'updated_at')
        if space not in hf.keys():
            hf.create_group(space)
        group = hf[space]

        with open(source) as sf:
            csvf = csv.DictReader(sf, delimiter=',', quotechar='"')
            self.names_ = csvf.fieldnames

            available_keys = csvf.fieldnames
            if not keys:
                fields_to_use = available_keys
                index_map = [i for i in range(len(fields_to_use))]
            else:
                for k in keys:
                    if k not in available_keys:
                        raise ValueError(f"key '{k}' isn't in the available keys ({keys})")
                fields_to_use = keys
                index_map = [available_keys.index(k) for k in keys]


            transforms_by_index = list()
            for i_n, n in enumerate(available_keys):
                if field_entries and n in field_entries:
                    transforms_by_index.append(field_entries[n])
                else:
                    transforms_by_index.append(None)

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
                if writers[field_name] == 'categoricaltype':
                    str_to_vals = field_entries[field_name].strings_to_values
                    writer =\
                        writer_factory[writers[field_name]](
                            datastore, group, field_name, 'int8', str_to_vals)
                    categorical_map_list.append(str_to_vals)
                elif writers[field_name] == 'leakycategoricaltype':
                    str_to_vals = field_entries[field_name].strings_to_values
                    out_of_range = field_entries[field_name].out_of_range_label
                    writer =\
                        writer_factory[writers[field_name]](
                            datastore, group, field_name, 'int8', str_to_vals, out_of_range)
                    # categorical_map_list.append(str_to_vals)
                    categorical_map_list.append(None)
                else:
                    writer = writer_factory[writers[field_name]](datastore, group, field_name)
                    categorical_map_list.append(None)
                new_fields[field_name] = writer
                new_field_list.append(writer)
                field_chunk_list.append(writer.chunk_factory(chunk_size))

            csvf = csv.reader(sf, delimiter=',', quotechar='"')
            ecsvf = iter(csvf)

            chunk_index = 0
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
                                    error.format(f, categorical_map, self.names_[i_f]))
                            f = categorical_map[f]
                        # t = transforms_by_index[i_f]
                        try:
                            field_chunk_list[i_df][chunk_index] = f
                        except Exception as e:
                            print(i_df, self.names_[i_f], chunk_index, f, e)
                            raise
                        # new_field_list[i_df].append(f)
                    chunk_index += 1
                    if chunk_index == chunk_size:
                        for i_df in range(len(index_map)):
                            new_field_list[i_df].write_part(field_chunk_list[i_df])
                        chunk_index = 0

            if chunk_index != 0:
                for i_df in range(len(index_map)):
                    new_field_list[i_df].write_part(field_chunk_list[i_df][:chunk_index])

            for i_df in range(len(index_map)):
                new_field_list[i_df].complete()
