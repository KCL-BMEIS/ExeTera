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

import h5py

from hystore.core import dataset, persistence

# TODO:
from hystore.core.importer import DatasetImporter

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
def import_to_hdf5(timestamp, dest_file_name, data_schema,
                   p_file_name=None, a_file_name=None, t_file_name=None,
                   territories=None):

    early_filter = None
    if territories is not None:
        early_filter = ('country_code', lambda x: x in tuple(territories.split(',')))

    datastore = persistence.DataStore()
    with h5py.File(dest_file_name, 'w') as hf:
        writer_factory = data_schema.field_writers

        show_every = 100000
        import_patients = True
        import_assessments = True
        import_tests = True

        patient_maps = data_schema.patient_categorical_maps
        patient_writers = data_schema.patient_field_types
        p_categorical_fields = set([a[0] for a in patient_writers.items() if a[1] == 'categoricaltype'])
        p_mapped_fields = set([a[0] for a in patient_maps.items()])

        if import_patients:
            print(p_categorical_fields)
            print(p_mapped_fields)
            print(p_categorical_fields.difference(p_mapped_fields))
            with open(p_file_name) as f:
                pds = dataset.Dataset(f, stop_after=1)
            p_names = set(pds.names_)
            print(p_names.difference(patient_writers.keys()))

        assessment_maps = data_schema.assessment_categorical_maps
        assessment_writers = data_schema.assessment_field_types
        a_categorical_fields = set([a[0] for a in assessment_writers.items() if a[1] == 'categoricaltype'])
        a_mapped_fields = set([a[0] for a in assessment_maps.items()])

        if import_assessments:
            print(a_categorical_fields)
            print(a_mapped_fields)
            print(a_categorical_fields.difference(a_mapped_fields))
            with open(a_file_name) as f:
                ads = dataset.Dataset(f, stop_after=1)
            a_names = set(ads.names_)
            print(a_names.difference(assessment_writers.keys()))

        test_maps = data_schema.test_categorical_maps
        test_writers = data_schema.test_field_types
        t_categorical_fields = set([t[0] for t in test_writers.items() if t[1] == 'categoricaltype'])
        t_mapped_fields = set([t[0] for t in test_maps.items()])
        if import_tests:
            print(t_categorical_fields)
            print(t_mapped_fields)
            print(t_categorical_fields.difference(t_mapped_fields))
            with open(t_file_name) as f:
                tds = dataset.Dataset(f, stop_after=1)
            t_names = set(tds.names_)
            print(t_names.difference(test_writers.keys()))

        # perform the imports

        if import_patients:
            p_show_progress_every = show_every
            p_stop_after = 100000
            p_keys = None
            DatasetImporter(datastore, p_file_name, hf, 'patients',
                            writer_factory, patient_writers, patient_maps, timestamp,
                            keys=p_keys, field_descriptors=patient_maps,
                            show_progress_every=p_show_progress_every, stop_after=p_stop_after,
                            early_filter=early_filter)
            print("patients done")


        if import_assessments:
            a_show_progress_every = show_every
            a_stop_after = 1200000
            a_keys = None
            DatasetImporter(datastore, a_file_name, hf, 'assessments',
                            writer_factory, assessment_writers, assessment_maps, timestamp,
                            keys=a_keys, field_descriptors=assessment_maps,
                            show_progress_every=a_show_progress_every, stop_after=a_stop_after,
                            early_filter=early_filter)
            print("assessments_done")


        if import_tests:
            t_show_progress_every = show_every
            t_stop_after = None
            t_keys = None
            DatasetImporter(datastore, t_file_name, hf, 'tests',
                            writer_factory, test_writers, test_maps, timestamp,
                            keys=t_keys, field_descriptors=test_maps,
                            show_progress_every=t_show_progress_every, stop_after=t_stop_after,
                            early_filter=early_filter)
            print("test_done")


# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--version', action='version', version='v0.2.0')
#     parser.add_argument('-te', '--territories', default=None,
#                         help='the territory/territories to filter the dataset on (runs on all territories if not set)')
#     parser.add_argument('-p', '--patient_data',
#                         help='the location and name of the patient data csv file')
#     parser.add_argument('-a', '--assessment_data',
#                         help='the location and name of the assessment data csv file')
#     parser.add_argument('-t', '--test_data',
#                         help='the location and name of the assessment data csv file')
#     parser.add_argument('-c', '--consent_data', default=None,
#                         help='the location and name of the consent data csv file')
#     parser.add_argument('-o', '--output_hdf5',
#                         help='the location and name of the output hdf5 file')
#     parser.add_argument('-d', '--data_schema', default=1, type=int,
#                         help='the schema number to use for parsing and cleaning data')
#     parser.add_argument('-ts', '--timestamp', default=str(datetime.now(timezone.utc)),
#                         help='override for the import datetime (')
#     args = parser.parse_args()
#
#     errors = False
#     if not os.path.isfile(args.patient_data):
#         print('-p/--patient_data argument must be an existing file')
#         errors = True
#     if not os.path.isfile(args.assessment_data):
#         print('-a/--assessment_data argument must be an existing file')
#         errors = True
#     if not os.path.isfile(args.test_data):
#         print('-t/--test_data argument must be an existing file')
#         errors = True
#
#     if errors:
#         exit(-1)
#
#     data_schema_version = 1
#     data_schema = data_schemas.DataSchema(data_schema_version)
#     data_schema = data_schemas.DataSchema(1)
#     import_to_hdf5(args.timestamp, args.output_hdf5, data_schema,
#                    args.patient_data, args.assessment_data, args.test_data,
#                    args.territories)
