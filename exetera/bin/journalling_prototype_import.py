import argparse
import os
import sys

import h5py

from exetera.io import importer
# from exetera.core.importer import DatasetImporter
# from exetera.core.persistence import DataStore
from exetera.core.session import Session
# from exetera.covidspecific import data_schemas


def consolidate(datastore, existing_group, new_group):

    # investigative - just append and sort
    # * get sorted index on pid, import_timestamp
    if existing_group is None:
        """Just make the initial group from the existing group"""
    else:
        """Add the existing group to the end of the initial group with a new timestamp"""

    # proper:
    # * index of last entry for each patient
    # * map between those indices and patients in the new dataset
    #   * for each field
    #     * if latest patient values differ from new patient values


def import_and_consolidate(datastore, dataset, source_file, data_schema, timestamp):
    writer_factory = data_schema.field_writers
    field_types = data_schema.patient_field_types
    categorical_maps = data_schema.patient_categorical_maps

    keys = None
    show_progress_every = 1000000
    stop_after = None
    DatasetImporter(datastore, source_file, dataset, 'new_patients',
                    writer_factory, field_types, categorical_maps, timestamp,
                    keys=keys, show_progress_every=show_progress_every, stop_after=stop_after)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--schema', help='The path and name of the schema file')
    parser.add_argument('--source_dir', help='The directory containing the source files')
    parser.add_argument('--pattern', help="The pattern that identifies files of interest in '--source_dir'")
    parser.add_argument('--dest', help='The path and name of the datatset to be created or appended to')

    # if len(sys.argv) != 4:
    #     print("Usage: check_for_duplicates.py <datastore> <directory> <pattern>")
    #     exit(1)

    args = parser.parse_args()


    show_progress_every = 500000

    filenames = sorted(fn for fn in os.listdir(sys.argv[2]) if sys.argv[3] in fn)

    with Session() as s:
        dataset = s.open_dataset(sys.argv[1], 'w', 'dataset')

        for fn in filenames:
            with open(fn) as src:
                import_and_consolidate(s, dataset, src, data_schema)