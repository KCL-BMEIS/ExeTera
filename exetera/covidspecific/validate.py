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
import argparse
import os
import datetime

from exetera.covidspecific import data_schemas, parsing_schemas, pipeline
from exetera.core import dataset, utils

equivalence_map = {
    'na': ('', 'na'),
    '': ('', 'na')
}

class Logger(object):
    def __init__(self):
        self.report = ''

    def log(self, text, print_to_console=False):
        if isinstance(text, list):
            for item in text:
                self.report += str(item)
                self.report += '\n'
                if print_to_console:
                    print(item)
        else:
            self.report += text
            self.report += '\n'
            if print_to_console:
                print(text)

    def write(self, destination, append=False):
        if os.path.isfile(destination) and append is False:
            mode = 'w'
        else:
            mode = 'a'
        with open(destination, mode) as f:
            f.write(self.report)




def compare_row(expected_index, expected_row, actual_index, actual_row, keys):
    all_true = True
    mismatches = None
    for k in keys:
        expected_value = expected_row.value_from_fieldname(expected_index, k)
        actual_value = actual_row.value_from_fieldname(actual_index, k)
        if expected_value in equivalence_map:
            if actual_value not in equivalence_map[expected_value]:
                all_true = False
                if mismatches is None:
                    mismatches = list()
                mismatches.append(k)
            # all_true = all_true and actual_value in equivalence_map[expected_value]
        else:
            if expected_value != actual_value:
                all_true = False
                if mismatches is None:
                    mismatches = list()
                mismatches.append(k)
            # all_true = all_true and expected_value == actual_value
    return all_true, mismatches


def compare(expected, actual, verbose):
    logger = Logger()
    logger.log(f'\nComparing expected {expected} and current {actual}', print_to_console=True)

    with open(expected) as f:
        expected_ds = dataset.Dataset(f)
    expected_field_names = expected_ds.names_
    with open(actual) as f:
        actual_ds = dataset.Dataset(f)
    actual_field_names = actual_ds.names_

    # compare field names
    if expected_field_names == actual_field_names:
        logger.log('field names are identical')
    else:
        logger.log(['WARNING: fieldnames only in expected:', set(expected_field_names).difference(set(actual_field_names))], print_to_console = True)
        logger.log(['WARNING: fieldnames only in actual:', set(actual_field_names).difference(set(expected_field_names))], print_to_console = True)

    common_fields = set(expected_field_names).intersection(set(actual_field_names))
    if verbose:
        logger.log(['common fields:', common_fields,'\n'], print_to_console = True)

    # compare data content
    if expected_ds.row_count() == actual_ds.row_count():
        logger.log(f'both files have {expected_ds.row_count()} rows', print_to_console=True)
    else:
        logger.log(f'WARNING: expected file has {expected_ds.row_count()} rows, actual file has {actual_ds.row_count()} rows', print_to_console=True)


    rows_that_differ = []
    for i in range(expected_ds.row_count()):
        matched, mismatches = compare_row(i, expected_ds, i, actual_ds, common_fields)
        if not matched:
            rows_that_differ.append((i, mismatches))


    if len(rows_that_differ) > 0:
        logger.log(f'WARNING: a total of {len(rows_that_differ)} rows do not match', print_to_console=True)
        if verbose:
            for row in rows_that_differ:
                logger.log('expected value | actual_value')
                for field in row[1]:# common_fields:
                    logger.log(f'{expected_ds.value_from_fieldname(row[0], field)} | {actual_ds.value_from_fieldname(row[0], field)}')
    else:
        logger.log('row contents are equal \n \n', print_to_console=True)
    return logger


def validate(expected_patients_file_name, actual_patients_file_name,
             expected_assessments_file_name, actual_assessments_file_name,
             verbose=False, report_dest=None):
    log = compare(expected_patients_file_name, actual_patients_file_name, verbose)
    if report_dest is not None:
        log.write(report_dest)
    log = compare(expected_assessments_file_name, actual_assessments_file_name, verbose)
    if report_dest is not None:
        log.write(report_dest, append=True)


def validate_full():
    filename = '/home/ben/covid/assessments_20200413050002_clean.csv'

    with open(filename) as asmt:
        csvr = csv.DictReader(asmt)
        fieldnames = csvr.fieldnames
        print(fieldnames)

        for ir, r in enumerate(csvr):
            if r['tested_covid_positive'] != '':
                print(ir, r['id'], r['patient_id'], r['tested_covid_positive'])


def show_rows(filename, fields_to_show):
    with open(filename) as f:
        ds = dataset.Dataset(f)
    # ds.parse_file()
    ds.sort(('patient_id', 'updated_at'))
    for i_f, f in enumerate(ds.fields_):
        if ds.value_from_fieldname(i_f, 'patient_id') == 'e88bfabe5b16897866f91deb8a7a90f2':
            utils.print_diagnostic_row(f'{i_f}:', ds, ds.fields_, i_f, fields_to_show)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pe', '--patients_output_expected',
                        help='the location of the patient file produced by a correct version (currently v.0.1.2) of the pipeline')
    parser.add_argument('-pi', '--patients_input',
                        help='the location of the patient input file to be processed')
    parser.add_argument('-ae', '--assessments_output_expected',
                        help='the location of the assessments file produced by a correct version (currently v.0.1.2) of the pipeline')
    parser.add_argument('-ai', '--assessments_input',
                        help='the location of the assessments input file to be processed')
    parser.add_argument('-r', '--report_destination',
                        help='the location for the report .txt to be saved',
                        default=None)
    parser.add_argument('-ps', '--parsing_schema', default=1,
                        help='the schema number to use for parsing and cleaning data')
    parser.add_argument('-y', '--year', default=datetime.datetime.now().year, type=int)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='increase verbosity')
    args = parser.parse_args()

    data_schema_version = 1
    data_schema = data_schemas.DataSchema(data_schema_version)
    parsing_schema_version = args.parsing_schema
    parsing_schema = parsing_schemas.ParsingSchema(parsing_schema_version)
    # run the current pipeline
    pipeline_output = pipeline.pipeline(patient_filename=args.patients_input,
                                        assessment_filename=args.assessments_input,
                                        data_schema=data_schema, parsing_schema=parsing_schema,
                                        year=args.year)
    pipeline.save_csv(pipeline_output,
                      patient_data_out='patients_output_test.csv',
                      assessment_data_out='assessments_output_test.csv',
                      data_schema=data_schema)

    # compare to expected pipeline output
    validate(expected_patients_file_name=args.patients_output_expected,
             actual_patients_file_name='patients_output_test.csv',
             expected_assessments_file_name=args.assessments_output_expected,
             actual_assessments_file_name='assessments_output_test.csv',
             report_dest=args.report_destination,
             verbose=args.verbose)