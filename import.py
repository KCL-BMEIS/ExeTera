from datetime import datetime, timezone
import csv
import h5py
import numpy as np

import dataset
import data_schemas
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
    def __init__(self, source, dest, space,
                 writer_factory, writers, field_entries, timestamp,
                 keys=None, field_descriptors=None,
                 stop_after=None, show_progress_every=None, filter_fn=None):
        self.names_ = list()
        self.index_ = None

        with h5py.File(dest, 'a') as hf:
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

                chunk_size = 1 << 18
                new_fields = dict()
                new_field_list = list()
                for i_n in range(len(fields_to_use)):
                    field_name = fields_to_use[i_n]
                    if writers[field_name] != 'categoricaltype':
                        writer = writer_factory[writers[field_name]](
                            group, chunk_size, field_name, timestamp)
                    else:
                        str_to_vals = field_entries[field_name].strings_to_values
                        writer =\
                            writer_factory[writers[field_name]](
                                group, chunk_size, field_name, timestamp, str_to_vals)
                    new_fields[field_name] = writer
                    new_field_list.append(writer)

                csvf = csv.reader(sf, delimiter=',', quotechar='"')
                ecsvf = iter(csvf)

                for i_r, row in enumerate(ecsvf):
                    if show_progress_every:
                        if i_r % show_progress_every == 0:
                            print(i_r)

                    if i_r == stop_after:
                        break

                    if not filter_fn or filter_fn(i_r):
                        for i_df, i_f in enumerate(index_map):
                            f = row[i_f]
                            # t = transforms_by_index[i_f]
                            try:
                                new_field_list[i_df].append(f)
                            except KeyError as k:
                                print(new_field_list[i_df].name, k)
                                raise

                for i_df in range(len(index_map)):
                    new_field_list[i_df].flush()


timestamp = str(datetime.now(timezone.utc))
p_file_name = '/home/ben/covid/patients_export_geocodes_20200604030001.csv'
a_file_name = '/home/ben/covid/assessments_export_20200604030001.csv'
t_file_name = '/home/ben/covid/covid_test_export_20200604030001.csv'
dest_file_name = '/home/ben/covid/dataset_test_indexed.hdf5'
data_schema = data_schemas.DataSchema(1)
writer_factory = data_schema.field_writers

import_patients = True
import_assessments = True
import_tests = True

if import_patients:
    patient_maps = data_schema.patient_categorical_maps
    patient_writers = data_schema.patient_field_types
    patient_field_entries = data_schema.patient_categorical_maps
    patient_maps = data_schema.patient_categorical_maps
    patient_writers = data_schema.patient_field_types
    patient_field_entries = data_schema.patient_categorical_maps
    p_categorical_fields = set([a[0] for a in patient_writers.items() if a[1] == 'categoricaltype'])
    p_mapped_fields = set([a[0] for a in patient_maps.items()])
    print(p_categorical_fields)
    print(p_mapped_fields)
    print(p_categorical_fields.difference(p_mapped_fields))
    with open(p_file_name) as f:
        pds = dataset.Dataset(f, stop_after=1)
    p_names = set(pds.names_)
    print(p_names.difference(patient_writers.keys()))


    p_show_progress_every = 10000
    p_stop_after = None
    p_keys = None # ('id', 'patient_id', 'updated_at', 'created_at', 'tested_covid_positive')
    pdi = DatasetImporter(p_file_name, dest_file_name, 'patients',
                         writer_factory, patient_writers, patient_maps, timestamp,
                         keys=p_keys, field_descriptors=patient_maps,
                         show_progress_every=p_show_progress_every, stop_after=p_stop_after)
    print("patients done")


if import_assessments:
    assessment_maps = data_schema.assessment_categorical_maps
    assessment_writers = data_schema.assessment_field_types
    a_categorical_fields = set([a[0] for a in assessment_writers.items() if a[1] == 'categoricaltype'])
    a_mapped_fields = set([a[0] for a in assessment_maps.items()])
    print(a_categorical_fields)
    print(a_mapped_fields)
    print(a_categorical_fields.difference(a_mapped_fields))
    a_show_progress_every = 10000
    a_stop_after = None
    a_keys = None # ('id', 'patient_id', 'updated_at', 'created_at', 'tested_covid_positive')
    adi = DatasetImporter(a_file_name, dest_file_name, 'assessments',
                         writer_factory, assessment_writers, assessment_maps, timestamp,
                         keys=a_keys, field_descriptors=assessment_maps,
                         show_progress_every=a_show_progress_every, stop_after=a_stop_after)
    print("assessments_done")


if import_tests:
    test_maps = data_schema.test_categorical_maps
    test_writers = data_schema.test_field_types
    t_categorical_fields = set([t[0] for t in test_writers.items() if t[1] == 'categoricaltype'])
    t_mapped_fields = set([t[0] for t in test_maps.items()])
    print(t_categorical_fields)
    print(t_mapped_fields)
    print(t_categorical_fields.difference(t_mapped_fields))
    t_show_progress_every = 10000
    t_stop_after = None
    t_keys = None
    tdi = DatasetImporter(t_file_name, dest_file_name, 'tests',
                          writer_factory, test_writers, test_maps, timestamp,
                          keys=t_keys, field_descriptors=test_maps,
                          show_progress_every=t_show_progress_every, stop_after=t_stop_after)
    print("test_done")
