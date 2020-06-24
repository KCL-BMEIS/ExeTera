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
from datetime import datetime, timezone
import time
import csv
import h5py
import numpy as np

import dataset
import data_schemas
import parsing_schemas
import persistence
import utils

# TODO:
"""
 * field source
   * original - quote name of file from which the data was imported
   * derived - algorithm that was run on the field
     * what about storing the actual algorithm, vs just the name and version
 * filters
   * non-destructive stored filters
     * from data cleaning
     * user-defined
     * unfiltered-count
     * soft filter format
       * boolean flags
       * indices
       * run length encoding
   * destructive filters to generate new spaces
   * built-in filters for fields that require them
 * sorting
   * soft-sort - generate a set of indices and permute at the point it is required
   * hard-sort - re-sort data
   * reference sort for hard-sorted data
 * readers
   * fetch all
   * yield-based - IN PROGRESS
 * cleaning
   * running filters
   * augmenting new tests with old tests
   * correcting/flagging age/height/weight/bmi
   * flagging healthy/not-healthy & symptoms
"""
class DatasetImporter:
    def __init__(self, source, hf, space,
                 writer_factory, writers, field_entries, timestamp,
                 keys=None, field_descriptors=None,
                 stop_after=None, show_progress_every=None, filter_fn=None,
                 early_filter=None):
        self.names_ = list()
        self.index_ = None

        time0 = time.time()

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
                if field_descriptors and n in field_descriptors:
                    transforms_by_index.append(field_descriptors[n])
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
            for i_n in range(len(fields_to_use)):
                field_name = fields_to_use[i_n]
                if writers[field_name] != 'categoricaltype':
                    writer = writer_factory[writers[field_name]](
                        group, chunk_size, field_name, timestamp)
                    categorical_map_list.append(None)
                else:
                    str_to_vals = field_entries[field_name].strings_to_values
                    writer =\
                        writer_factory[writers[field_name]](
                            group, chunk_size, field_name, timestamp, str_to_vals)
                    categorical_map_list.append(str_to_vals)
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


def import_to_hdf5(timestamp, dest_file_name, data_schema,
                   p_file_name=None, a_file_name=None, t_file_name=None,
                   territories=None):

    early_filter = None
    if territories is not None:
        early_filter = ('country_code', lambda x: x in tuple(territories.split(',')))

    with h5py.File(dest_file_name, 'w') as hf:
        writer_factory = data_schema.field_writers

        show_every = 100000
        import_patients = True
        import_assessments = True
        import_tests = True

        if import_patients:
            patient_maps = data_schema.patient_categorical_maps
            patient_writers = data_schema.patient_field_types
            p_categorical_fields = set([a[0] for a in patient_writers.items() if a[1] == 'categoricaltype'])
            p_mapped_fields = set([a[0] for a in patient_maps.items()])
            print(p_categorical_fields)
            print(p_mapped_fields)
            print(p_categorical_fields.difference(p_mapped_fields))
            with open(p_file_name) as f:
                pds = dataset.Dataset(f, stop_after=1)
            p_names = set(pds.names_)
            print(p_names.difference(patient_writers.keys()))


            p_show_progress_every = show_every
            p_stop_after = None
            p_keys = None
            DatasetImporter(p_file_name, hf, 'patients',
                            writer_factory, patient_writers, patient_maps, timestamp,
                            keys=p_keys, field_descriptors=patient_maps,
                            show_progress_every=p_show_progress_every, stop_after=p_stop_after,
                            early_filter=early_filter)
            print("patients done")


        if import_assessments:
            assessment_maps = data_schema.assessment_categorical_maps
            assessment_writers = data_schema.assessment_field_types
            a_categorical_fields = set([a[0] for a in assessment_writers.items() if a[1] == 'categoricaltype'])
            a_mapped_fields = set([a[0] for a in assessment_maps.items()])
            print(a_categorical_fields)
            print(a_mapped_fields)
            print(a_categorical_fields.difference(a_mapped_fields))
            a_show_progress_every = show_every
            a_stop_after = None
            a_keys = None
            DatasetImporter(a_file_name, hf, 'assessments',
                            writer_factory, assessment_writers, assessment_maps, timestamp,
                            keys=a_keys, field_descriptors=assessment_maps,
                            show_progress_every=a_show_progress_every, stop_after=a_stop_after,
                            early_filter=early_filter)
            print("assessments_done")


        if import_tests:
            test_maps = data_schema.test_categorical_maps
            test_writers = data_schema.test_field_types
            t_categorical_fields = set([t[0] for t in test_writers.items() if t[1] == 'categoricaltype'])
            t_mapped_fields = set([t[0] for t in test_maps.items()])
            print(t_categorical_fields)
            print(t_mapped_fields)
            print(t_categorical_fields.difference(t_mapped_fields))
            t_show_progress_every = show_every
            t_stop_after = None
            t_keys = None
            DatasetImporter(t_file_name, hf, 'tests',
                            writer_factory, test_writers, test_maps, timestamp,
                            keys=t_keys, field_descriptors=test_maps,
                            show_progress_every=t_show_progress_every, stop_after=t_stop_after,
                            early_filter=early_filter)
            print("test_done")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='v0.2.0')
    parser.add_argument('-te', '--territories', default=None,
                        help='the territory/territories to filter the dataset on (runs on all territories if not set)')
    parser.add_argument('-p', '--patient_data',
                        help='the location and name of the patient data csv file')
    parser.add_argument('-a', '--assessment_data',
                        help='the location and name of the assessment data csv file')
    parser.add_argument('-t', '--test_data',
                        help='the location and name of the assessment data csv file')
    parser.add_argument('-c', '--consent_data', default=None,
                        help='the location and name of the consent data csv file')
    parser.add_argument('-o', '--output_hdf5',
                        help='the location and name of the output hdf5 file')
    parser.add_argument('-d', '--data_schema', default=1, type=int,
                        help='the schema number to use for parsing and cleaning data')
    parser.add_argument('-ts', '--timestamp', default=str(datetime.now(timezone.utc)),
                        help='override for the import datetime (')
    args = parser.parse_args()

    errors = False
    if not os.path.isfile(args.patient_data):
        print('-p/--patient_data argument must be an existing file')
        errors = True
    if not os.path.isfile(args.assessment_data):
        print('-a/--assessment_data argument must be an existing file')
        errors = True
    if not os.path.isfile(args.test_data):
        print('-t/--test_data argument must be an existing file')
        errors = True

    if errors:
        exit(-1)

    data_schema_version = 1
    data_schema = data_schemas.DataSchema(data_schema_version)
    data_schema = data_schemas.DataSchema(1)
    import_to_hdf5(args.timestamp, args.output_hdf5, data_schema,
                   args.patient_data, args.assessment_data, args.test_data,
                   args.territories)
