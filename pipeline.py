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
from collections import defaultdict
import numpy as np

import dataset
import data_schemas
import parsing_schemas


def read_header_and_n_lines(filename, n):
    with open(filename) as f:
        print(f.readline())
        for i in range(n):
            print(f.readline())


def build_histogram(dataset, field_index, filtered_records=None, tx=None):
    if False:
        # TODO: memory_efficiency: test and replace defaultdict with this code when tested
        dataset = sorted(dataset, dataset.field_index)
        histogram = list()
        histogram.append((dataset[0][1], 0))
        for r in dataset:
            if histogram[-1][0] != r[1]:
                histogram.append((r[1], 1))
            else:
                histogram[-1] = (histogram[-1][0], histogram[-1][1] + 1)
    else:
        histogram = defaultdict(int)
        for ir, r in enumerate(dataset):
            if not filtered_records or not filtered_records[ir]:
                if tx is not None:
                    value = tx(r[field_index])
                else:
                    value = r[field_index]
                histogram[value] += 1
        hlist = list(histogram.items())
        del histogram
        return hlist


def build_histogram_from_list(dataset, filtered_records=None, tx=None):
    # TODO: memory_efficiency: see build_histogram function
    histogram = defaultdict(int)
    for ir, r in enumerate(dataset):
        if not filtered_records or not filtered_records[ir]:
            if tx is not None:
                value = tx(r)
            else:
                value = r
            histogram[value] += 1
    hlist = list(histogram.items())
    del histogram
    return hlist


def is_int(value):
    try:
        int(float(value))
        return True
    except:
        return False


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False


def to_int(value):
    try:
        fvalue = float(value)
    except ValueError as e:
        raise ValueError(f'{value} cannot be converted to float')

    try:
        ivalue = int(fvalue)
    except ValueError as e:
        raise ValueError(f'{fvalue} cannot be converted to int')

    return ivalue


def to_float(value):
    try:
        fvalue = float(value)
    except ValueError as e:
        raise ValueError(f'{value} cannot be converted to float')

    return fvalue


def replace_if_invalid(replacement):
    def inner_(value):
        if value is '':
            return replacement
        else:
            return float(value)
    return inner_


def count_flag_set(flags, flag_to_test):
    count = 0
    for f in flags:
        if f & flag_to_test:
            count += 1
    return count


def clear_set_flag(values, to_clear):
    for v in range(len(values)):
        values[v] &= ~to_clear
    return values


def valid_range_fac(f_min, f_max, default_value=''):
    def inner_(x):
        return x == default_value or x > f_min and x < f_max
    return inner_


def valid_range_fac_inc(f_min, f_max, default_value=''):
    def inner_(x):
        return x == default_value or x >= f_min and x <= f_max
    return inner_


def filter_fields(fields, filter_list, index, f_missing, f_bad, is_type_fn, type_fn, valid_fn):
    for ir, r in enumerate(fields):
        if not is_type_fn(r[index]):
            if f_missing != 0:
                filter_list[ir] |= f_missing
        else:
            value = type_fn(r[index])
            if not valid_fn(value):
                if f_bad != 0:
                    filter_list[ir] |= f_bad


def filter_list(fields, filter_list, f_missing, f_bad, is_type_fn, type_fn, valid_fn):
    for ir, r in enumerate(fields):
        if not is_type_fn(r):
            if f_missing != 0:
                filter_list[ir] |= f_missing
        else:
            value = type_fn(r)
            if not valid_fn(value):
                if f_bad != 0:
                    filter_list[ir] |= f_bad


def sort_mixed_list(values, check_fn, sort_fn):
    # pass to find the single entry that fails check_fn
    for iv in range(len(values)):
        checked_item = None
        if not check_fn(values[iv]):
            #swap the current item with the last if it isn't last
            found_checked_item = True
            if iv != len(values) - 1:
                values[iv], values[-1] = values[-1], values[iv]
                checked_item = values.pop()
        break

    list.sort(values, key=sort_fn)
    if found_checked_item:
        values.append(checked_item)

    return values


def to_categorical(fields, field_index, desttype, mapdict):
    results = np.ndarray((len(fields),), dtype=desttype)
    for ir, r in enumerate(fields):
        v = r[field_index]
        results[ir] = mapdict[v]
    return results


def map_patient_ids(geoc, asmt, map_fn):
    g = 0
    a = 0
    while g < len(geoc) and a < len(asmt):
        gpid = geoc[g][0]
        apid = asmt[a][1]
        if gpid < apid:
            g += 1
        elif apid < gpid:
            a += 1
        else:
            map_fn(geoc, asmt, g, a)
            a += 1


def iterate_over_patient_assessments(fields, filter_status, visitor):
    cur_id = fields[0][1]
    cur_start = 0
    cur_end = 0
    i = 1
    while i < len(fields):
        while fields[i][1] == cur_id:
            cur_end = i
            i += 1
            if i >= len(fields):
                break;

        visitor(fields, filter_status, cur_start, cur_end)

        if i < len(fields):
            cur_start = i
            cur_end = cur_start
            cur_id = fields[i][1]


def datetime_to_seconds(dt):
    return f'{dt[0:4]}-{dt[5:7]}-{dt[8:10]} {dt[11:13]}:{dt[14:16]}:{dt[17:19]}'


def print_diagnostic_row(preamble, ds, fields, ir, keys, fns=None):
    if fns is None:
        fns = dict()
    indices = [ds.field_to_index(k) for k in keys]
    indexed_fns = [None if k not in fns else fns[k] for k in keys]
    values = [None] * len(indices)
    for ii, i in enumerate(indices):
        if indexed_fns[ii] is None:
            values[ii] = fields[ir][i]
        else:
            values[ii] = indexed_fns[ii](fields[ir][i])
    print(f'{preamble}: {values}')


#patient limits
MIN_YOB = 1930
MAX_YOB = 2004
MIN_HEIGHT = 110
MAX_HEIGHT = 220
MIN_WEIGHT = 40
MAX_WEIGHT = 200
MIN_BMI = 15
MAX_BMI = 55

MIN_TEMP = 35
MAX_TEMP = 42

# patient filter values
PFILTER_OTHER_TERRITORY = 0x1
PFILTER_NO_ASSESSMENTS = 0x2
PFILTER_ONE_ASSESSMENT = 0x4
FILTER_MISSING_YOB = 0x8
FILTER_BAD_YOB = 0x10
FILTER_MISSING_HEIGHT = 0x20
FILTER_BAD_HEIGHT = 0x40
FILTER_MISSING_WEIGHT = 0x80
FILTER_BAD_WEIGHT = 0x100
FILTER_MISSING_BMI = 0x200
FILTER_BAD_BMI = 0x400
FILTER_NOT_IN_FINAL_ASSESSMENTS = 0x800
FILTERP_ALL = 0xffffffff
patient_flag_descs = {
    0x1: 'other_territory',
    0x2: 'no_assessments', 0x4: 'one_assessment',
    0x8: 'missing_year_of_birth', 0x10: 'out_of_range_year_of_birth',
    0x20: 'missing_height', 0x40: 'out_of_range_height',
    0x80: 'missing_weight', 0x100: 'out_of_range_weight',
    0x200: 'missing_bmi', 0x400: 'out_of_range_bmi',
    0x800: 'not_in_final_assessments',
    0xffffffff: 'all_flags'
}

# assessment filter values
AFILTER_INVALID_PATIENT_ID = 0x1
AFILTER_PATIENT_FILTERED = 0x2
FILTER_MISSING_TEMP = 0x0
FILTER_BAD_TEMP = 0x8
FILTER_INCONSISTENT_NOT_TESTED = 0x10
FILTER_INCONSISTENT_TESTED = 0x20
FILTER_INCONSISTENT_SYMPTOMS = 0x40
FILTER_INCONSISTENT_NO_SYMPTOMS = 0x80
FILTER_INVALID_COVID_PROGRESSION = 0X100
FILTERA_ALL = 0xffffffff


assessment_flag_descs = {
    0x1: 'invalid_patient_id',
    0x2: 'patient_filtered',
    0x8: 'out_of_range_temperature',
    0x10: 'inconsistent_testing_not_tested', 0x20: 'inconsistent_testing_was_tested',
    0x40: 'inconsistent_symptoms', 0x80: 'inconsistent_no_symptoms',
    0x100: 'invalid_covid_progression',
    0xffffffff: 'all_flags'
}

symptomatic_fields = ["fatigue", "shortness_of_breath", "abdominal_pain", "chest_pain",
                      "delirium", "diarrhoea", "fever", "headache",
                      "hoarse_voice", "loss_of_smell", "persistent_cough",  "skipped_meals",
                      "sore_throat", "unusual_muscle_pains"]
flattened_fields = [("fatigue", "fatigue_binary"), ("shortness_of_breath", "shortness_of_breath_binary")]
exposure_fields = ["always_used_shortage", "have_used_PPE", "never_used_shortage", "sometimes_used_shortage",
                   "treated_patients_with_covid"]
miscellaneous_fields = ['location', 'level_of_isolation', 'had_covid_test']


def pipeline(patient_filename, assessment_filename, data_schema, parsing_schema, territory=None):

    print(); print();
    print('load patients')
    print('-------------')
    # geoc_fieldnames = enumerate_fields(patient_filename)
    # geoc_countdict = {'id': False, 'patient_id': False}
    with open(patient_filename) as f:
        geoc_ds = dataset.Dataset(f)
    # with open(patient_filename) as f:
    #         geoc_ds.parse_file(f)
    geoc_ds.sort(('id',))
    geoc_ds.show()


    print(); print()
    print('load assessments')
    print('----------------')
    # asmt_fieldnames = enumerate_fields(assessment_filename)
    # asmt_countdict = {'id': False, 'patient_id': False}
    with open(assessment_filename) as f:
        asmt_ds = dataset.Dataset(f)
    # with open(assessment_filename) as f:
    #     asmt_ds.parse_file(f)
    asmt_ds.sort(('patient_id', 'updated_at'))
    asmt_ds.show()


    print(); print()
    print('generate dataset indices')
    print("------------------------")
    symptomatic_indices = [asmt_ds.field_to_index(c) for c in symptomatic_fields]
    print(symptomatic_indices)
    exposure_indices = [asmt_ds.field_to_index(c) for c in exposure_fields]
    print(exposure_indices)


    print(); print()
    print("pre-sort by patient id")
    print("----------------------")
    print("pre-sort patient data")
    geoc_fields = sorted(geoc_ds.fields_, key=lambda r: r[0])
    geoc_filter_status = [0] * len(geoc_fields)

    print("pre-sort assessment data")
    asmt_fields = sorted(asmt_ds.fields_, key=lambda r: (r[1], r[3]))
    asmt_filter_status = [0] * len(asmt_fields)


    if territory is not None:
        print(); print();
        print("filter patients from outside the territory of interest")
        print("------------------------------------------------------")

        country_code = geoc_ds.field_to_index('country_code')
        for ir, r in enumerate(geoc_fields):
            if r[country_code] != territory:
                geoc_filter_status[ir] |= PFILTER_OTHER_TERRITORY
        print(f'other territories: filtered {count_flag_set(geoc_filter_status, PFILTER_OTHER_TERRITORY)} missing values')


    # print(count_flag_set(geoc_filter_status, PFILTER_NO_ASSESSMENTS))
    patient_ids = set()
    for r in geoc_fields:
        patient_ids.add(r[0])

    print('patients:', len(geoc_filter_status))
    print('patients with no assessments:',
          count_flag_set(geoc_filter_status, PFILTER_NO_ASSESSMENTS))
    print('patients with one assessment:',
          count_flag_set(geoc_filter_status, PFILTER_ONE_ASSESSMENT))
    print('patients with sufficient assessments:', geoc_filter_status.count(0))

    print(); print()
    print("patients")
    print("--------")

    ptnt_dest_fields = dict()

    print(); print("checking yob")

    filter_fields(geoc_fields, geoc_filter_status, geoc_ds.field_to_index('year_of_birth'),
                  FILTER_MISSING_YOB, FILTER_BAD_YOB, is_int, to_int, valid_range_fac_inc(MIN_YOB, MAX_YOB))
    print(f'yob: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_YOB)} missing values')
    print(f'yob: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_YOB)} bad values')
    yobs = build_histogram(geoc_fields, geoc_ds.field_to_index('year_of_birth'))
    print(f'yob: {len(yobs)} unique values')
    print(geoc_filter_status.count(0))
    age = np.zeros((len(geoc_fields),), dtype=np.int16)
    yob_index = geoc_ds.field_to_index('year_of_birth')
    for ir, r in enumerate(geoc_fields):
        if geoc_filter_status[ir] & (FILTER_MISSING_YOB | FILTER_BAD_YOB):
            age[ir] = 0
        else:
            age[ir] = 2020 - to_int(r[yob_index])
    ptnt_dest_fields['age'] = age

    print(); print("checking height")
    filter_fields(geoc_fields, geoc_filter_status, geoc_ds.field_to_index('height_cm'),
                  FILTER_MISSING_HEIGHT, FILTER_BAD_HEIGHT, is_float, to_float, valid_range_fac_inc(MIN_HEIGHT, MAX_HEIGHT))
    print(f'height: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_HEIGHT)} missing values')
    print(f'height: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_HEIGHT)} bad values')
    heights = build_histogram(geoc_fields, geoc_ds.field_to_index('height_cm')) #, tx=replace_if_invalid(-1.0))
    print(f'height: {len(heights)} unique values')
    print(geoc_filter_status.count(0))

    print(); print("checking weight")
    filter_fields(geoc_fields, geoc_filter_status, geoc_ds.field_to_index('weight_kg'),
                  FILTER_MISSING_WEIGHT, FILTER_BAD_WEIGHT, is_float, to_float, valid_range_fac_inc(MIN_WEIGHT, MAX_WEIGHT))
    print(f'weight: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_WEIGHT)} missing values')
    print(f'weight: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_WEIGHT)} bad values')
    weights = build_histogram(geoc_fields, geoc_ds.field_to_index('weight_kg')) #, tx=replace_if_invalid(-1.0))
    print(f'weight: {len(weights)} unique values')
    print(geoc_filter_status.count(0))

    print(); print("checking bmi")
    filter_fields(geoc_fields, geoc_filter_status, geoc_ds.field_to_index('bmi'), FILTER_MISSING_BMI, FILTER_BAD_BMI,
                 is_float, to_float, valid_range_fac_inc(MIN_BMI, MAX_BMI))
    print(f'bmi: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_BMI)} missing values')
    print(f'bmi: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_BMI)} bad values')
    bmis = build_histogram(geoc_fields, geoc_ds.field_to_index('bmi')) #, tx=replace_if_invalid(-1.0))
    print(f'bmi: {len(bmis)} unique values')

    print(); print('unfiltered patients:', geoc_filter_status.count(0))


    print(); print()
    print("assessments")
    print("-----------")

    asmt_dest_fields = dict()
    asmt_dest_keys = dict()

    clear_set_flag(asmt_filter_status, FILTERA_ALL)
    # print(); print("removing assessments for filtered patients")
    # def filter_assessments_on_patient_ids(geoc, asmt, g, a):
    #     if geoc_filter_status[g] != 0:
    #         asmt_filter_status[a] |= AFILTER_PATIENT_FILTERED

    # map_patient_ids(geoc_fields, asmt_fields, filter_assessments_on_patient_ids)

    patient_ids = set()
    for ir, r in enumerate(geoc_fields):
        if geoc_filter_status[ir] == 0:
            patient_ids.add(r[0])
    for ir, r in enumerate(asmt_fields):
        if r[1] not in patient_ids:
            asmt_filter_status[ir] |= AFILTER_PATIENT_FILTERED

    print('assessments filtered due to patient filtering:',
          count_flag_set(asmt_filter_status, FILTERA_ALL))

    print(); print("checking temperature")
    # convert temperature to C if F
    temperature_c = np.zeros((len(asmt_fields),), dtype=np.float)
    temperature_index = asmt_ds.field_to_index('temperature')
    for ir, r in enumerate(asmt_fields):
        t = r[temperature_index]
        if is_float(t):
            t = float(t)
            temperature_c[ir] = (t - 32) / 1.8 if t > MAX_TEMP else t
        else:
            temperature_c[ir] = 0.0

    filter_list(temperature_c, asmt_filter_status, FILTER_MISSING_TEMP, FILTER_BAD_TEMP,
                  is_float, to_float, valid_range_fac(MIN_TEMP, MAX_TEMP, 0.0))
    print(f'temperature: filtered {count_flag_set(asmt_filter_status, FILTER_MISSING_TEMP)} missing values')
    print(f'temperature: filtered {count_flag_set(asmt_filter_status, FILTER_BAD_TEMP)} bad values')
    asmt_dest_fields['temperature_C'] = temperature_c

    indices = [asmt_ds.field_to_index(c) for c in ('had_covid_test', 'tested_covid_positive')]

    for ir, r in enumerate(asmt_fields):
        had_test = asmt_fields[ir][indices[0]]
        test_result = asmt_fields[ir][indices[1]]
        if had_test != 'True' and test_result != '':
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_NOT_TESTED
        if had_test == 'True' and test_result == '':
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_TESTED

    print(f'inconsistent_not_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_NOT_TESTED)} missing values')
    print(f'inconsistent_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_TESTED)} missing values')
    print(); print('unfiltered assessments:', asmt_filter_status.count(0))


    print(); print()
    print("convert symptomatic, exposure, flattened and miscellaneous fields to bool")
    print("-------------------------------------------------------------------------")

    any_symptoms = np.zeros((len(asmt_fields),), dtype=np.bool)
    for s in symptomatic_fields:
        print('symptomatic_field', s)
        cv = data_schema[s].strings_to_values
        print(f"symptomatic_field '{s}' to categorical")
        asmt_dest_fields[s] = to_categorical(asmt_fields, asmt_ds.field_to_index(s), np.uint8, cv)
        any_symptoms |= asmt_dest_fields[s] > 1
        print(np.count_nonzero(asmt_dest_fields[s] == True))
        print(np.count_nonzero(any_symptoms == True))

    print(build_histogram(asmt_fields, asmt_ds.field_to_index('tested_covid_positive')))

    for f in flattened_fields:
        cv = data_schema[f[1]].strings_to_values
        print(f[1], cv)
        print(f"flattened_field '{f[0]}' to categorical field '{f[1]}'")
        asmt_dest_fields[f[1]] = to_categorical(asmt_fields, asmt_ds.field_to_index(f[0]), np.uint8, cv)
        any_symptoms |= asmt_dest_fields[f[1]] > 1

    for e in exposure_fields:
        cv = data_schema[e].strings_to_values
        print(f"exposure_field '{e}' to categorical")
        asmt_dest_fields[e] = to_categorical(asmt_fields, asmt_ds.field_to_index(e), np.uint8, cv)

    for m in miscellaneous_fields:
        cv = data_schema[m].strings_to_values
        print(f"miscellaneous_field '{m}' to categorical")
        print(build_histogram(asmt_fields, asmt_ds.field_to_index(m)))
        asmt_dest_fields[m] = to_categorical(asmt_fields, asmt_ds.field_to_index(m), np.uint8, cv)

    print(); print()
    print("filter inconsistent health status")
    print("---------------------------------")
    health_status_index = asmt_ds.field_to_index('health_status')
    health_status = build_histogram(asmt_fields, health_status_index)

    for ir, r in enumerate(asmt_fields):
        if r[health_status_index] == 'healthy' and any_symptoms[ir]:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_SYMPTOMS
        elif r[health_status_index] == 'not_healthy' and not any_symptoms[ir]:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_NO_SYMPTOMS

    for f in (FILTER_INCONSISTENT_SYMPTOMS, FILTER_INCONSISTENT_NO_SYMPTOMS):
        print(f'{assessment_flag_descs[f]}: {count_flag_set(asmt_filter_status, f)}')

    print(); print('unfiltered assessments:', asmt_filter_status.count(0))

    # print('fatigue_binary')
    # print(build_histogram_from_list(asmt_dest_fields['fatigue_binary']))

    # validate assessments per patient
    print(); print()
    print("validate covid progression")
    print("--------------------------")

    sanitised_covid_results = np.ndarray((len(asmt_fields),), dtype=np.uint8)
    sanitised_covid_results_key = data_schema['tested_covid_positive'].values_to_strings[:]

    fn_fac = parsing_schemas.ParsingSchema(1).class_entries['clean_covid_progression']
    fn = fn_fac(asmt_ds, asmt_filter_status, sanitised_covid_results_key, sanitised_covid_results,
                FILTER_INVALID_COVID_PROGRESSION)
    # fn = ValidateCovidTestResultsFac(asmt_ds, asmt_filter_status, sanitised_covid_results_key, sanitised_covid_results)
    iterate_over_patient_assessments(asmt_fields, asmt_filter_status, fn)

    print(f'{assessment_flag_descs[FILTER_INVALID_COVID_PROGRESSION]}:',
          count_flag_set(asmt_filter_status, FILTER_INVALID_COVID_PROGRESSION))

    asmt_dest_fields['tested_covid_positive'] = sanitised_covid_results
    asmt_dest_keys['tested_covid_positive'] = sanitised_covid_results_key

    print('remaining assessments before squashing', asmt_filter_status.count(0))

    # create a new assessment space with only unfiltered rows
    print(); print()
    print("discard all filtered assessments")
    print("--------------------------------")
    remaining_asmt_fields = list()
    remaining_dest_fields = dict()

    print('asmt_fields:', len(asmt_fields))
    for ir, r in enumerate(asmt_fields):
        if not asmt_filter_status[ir]:
            remaining_asmt_fields.append(r)
    print('remaining_asmt_fields:', len(remaining_asmt_fields))

    for dk, dv in asmt_dest_fields.items():
        remaining_dest_fields[dk] = np.zeros((len(remaining_asmt_fields), ), dtype=dv.dtype)
        rdf = remaining_dest_fields[dk]
        rdindex = 0
        for ir in range(len(asmt_fields)):

            if not asmt_filter_status[ir]:
                rdf[rdindex] = dv[ir]
                rdindex += 1

    print(len(remaining_asmt_fields))
    remaining_asmt_filter_status = [0] * len(remaining_asmt_fields)
    print(len(remaining_asmt_filter_status))

    print(); print()
    print("quantise assessments by day")
    print("---------------------------")
    merged_row_count = 0

    def calculate_merged_field_count_fac():

        def inner_(fields, dummy, start, end):
            nonlocal merged_row_count
            results = list()
            for i in range(start + 1, end + 1):
                last_date_str = fields[i-1][3]
                last_date = (last_date_str[0:4], last_date_str[5:7], last_date_str[8:10])
                cur_date_str = fields[i][3]
                cur_date = (cur_date_str[0:4], cur_date_str[5:7], cur_date_str[8:10])
                if last_date == cur_date:
                    merged_row = fields[i].copy()
                    merged_row_count += 1

        return inner_


    class MergeAssessmentRows:
        def __init__(self, resulting_fields, created_fields, existing_field_indices):
            print(created_fields.keys())
            print(created_fields['tested_covid_positive'].dtype, created_fields['tested_covid_positive'])
            print(resulting_fields['tested_covid_positive'].dtype, resulting_fields['tested_covid_positive'])
            self.rfindex = 0
            self.resulting_fields = resulting_fields
            self.created_fields = created_fields
            self.existing_field_indices = existing_field_indices

        def populate_row(self, source_fields, source_index):
            source_row = source_fields[source_index]
            for e in self.existing_field_indices:
                self.resulting_fields[e[0]][self.rfindex] = source_row[e[1]]
            for ck, cv in self.created_fields.items():
                self.resulting_fields[ck][self.rfindex] =\
                    max(self.resulting_fields[ck][self.rfindex], cv[source_index])

        def __call__(self, fields, dummy, start, end):
            rfstart = self.rfindex

            # write the first row to the current resulting field index
            prev_asmt = fields[start]
            prev_date_str = prev_asmt[3]
            prev_date = (prev_date_str[0:4], prev_date_str[5:7], prev_date_str[8:10])
            self.populate_row(fields, start)

            for i in range(start + 1, end + 1):
                cur_asmt = fields[i]
                cur_date_str = cur_asmt[3]
                cur_date = (cur_date_str[0:4], cur_date_str[5:7], cur_date_str[8:10])
                if cur_date != prev_date:
                    self.rfindex += 1
                if i % 1000000 == 0 and i > 0:
                    print('.')
                self.populate_row(fields, i)

                prev_asmt = cur_asmt
                prev_date_str = cur_date_str
                prev_date = cur_date

            # finally, update the resulting field index one more time
            self.rfindex += 1

    fn = calculate_merged_field_count_fac()
    print(len(remaining_asmt_fields))
    print(len(remaining_asmt_filter_status))
    iterate_over_patient_assessments(remaining_asmt_fields, remaining_asmt_filter_status, fn)
    remaining_asmt_row_count = len(remaining_asmt_fields) - merged_row_count
    print(f'{len(remaining_asmt_fields)} - {merged_row_count} = {remaining_asmt_row_count}')

    existing_fields = ('id', 'patient_id', 'created_at', 'updated_at', 'version',
                       'country_code', 'health_status')
    existing_field_indices = [(f, asmt_ds.field_to_index(f)) for f in existing_fields]

    resulting_fields = dict()
    for e in existing_fields:
        resulting_fields[e] = [None] * remaining_asmt_row_count
    for dk, dv in remaining_dest_fields.items():
        resulting_fields[dk] = np.zeros((remaining_asmt_row_count, ), dtype=dv.dtype)

    resulting_field_keys = dict()
    for dk, dv in asmt_dest_keys.items():
        resulting_field_keys[dk] = dv

    print(); print('fatigue_binary before merge')
    print(build_histogram_from_list(remaining_dest_fields['fatigue_binary']))

    unique_patients = set()
    for ir, r in enumerate(remaining_asmt_fields):
        unique_patients.add(r[1])
    print('unique patents in remaining assessments:', len(unique_patients))

    # sanity check
    print('remaining_field_len:', len(remaining_asmt_fields))
    print('remaining_dest_len:', len(remaining_dest_fields['fatigue_binary']))
    for ir, r in enumerate(remaining_asmt_fields):
        f = r[asmt_ds.field_to_index('fatigue')]
        fb = remaining_dest_fields['fatigue_binary'][ir]
        # if (f not in ('mild', 'severe') and fb == True) or (f in ('mild', 'severe') and fb == False):
        #    print('empty' if f == '' else f, fb)
    print('resulting_fields:', len(resulting_fields['patient_id']))

    print(build_histogram_from_list(remaining_dest_fields['tested_covid_positive']))
    merge = MergeAssessmentRows(resulting_fields, remaining_dest_fields, existing_field_indices)
    iterate_over_patient_assessments(remaining_asmt_fields, remaining_asmt_filter_status, merge)
    print(merge.rfindex)

    unique_patients = defaultdict(int)
    for ir, r in enumerate(remaining_asmt_fields):
        unique_patients[r[1]] += 1
    print('unique patents in remaining assessments:', len(unique_patients))

#    print(); print()
#    print("filter patients with insufficient assessments")
#    print("---------------------------------------------")
#    patient_assessment_counts = defaultdict(int)
#    for a in asmt_fields:
#        patient_assessment_counts[a[1]] += 1
#    patient_assessments = list(patient_assessment_counts.items())
#
#    for ir, r in enumerate(geoc_fields):
#        pid = r[0]
#        if pid not in patient_assessment_counts:
#            geoc_filter_status[ir] |= PFILTER_NO_ASSESSMENTS
#        elif patient_assessment_counts[pid] == 1:
#            geoc_filter_status[ir] |= PFILTER_ONE_ASSESSMENT
#    del patient_assessment_counts

#    print('filter patients with only zero or one rows')
#    patient_ids = set()
#    for ir, r in enumerate(geoc_fields):
#        if geoc_filter_status[ir] == 0:
#            patient_ids.add(r[0])
#    for ir, r in enumerate(asmt_fields):
#        if r[1] not in patient_ids:
#            asmt_filter_status[ir] |= AFILTER_PATIENT_FILTERED
#
#    print('assessments filtered due to patient filtering:',
#          count_flag_set(asmt_filter_status, FILTERA_ALL))


    print(); print()
    print("filter summaries")
    print("----------------")

    print(); print('patient flags set')
    for v in patient_flag_descs.keys():
        print(f'{patient_flag_descs[v]}: {count_flag_set(geoc_filter_status, v)}')

    print(); print('assessment flags set')
    for v in assessment_flag_descs.keys():
        print(f'{assessment_flag_descs[v]}: {count_flag_set(asmt_filter_status, v)}')

    return (geoc_ds, geoc_fields, geoc_filter_status, ptnt_dest_fields,
            asmt_ds, asmt_fields, asmt_filter_status,
            remaining_asmt_fields, remaining_asmt_filter_status,
            resulting_fields, resulting_field_keys)


def regression_test_assessments(old_assessments, new_assessments):
    with open(old_assessments) as f:
        r_a_ds = dataset.Dataset(f)
        # r_a_ds.parse_file(f)
    r_a_ds.sort(('patient_id', 'id'))
    with open(new_assessments) as f:
        p_a_ds = dataset.Dataset(f)
        # p_a_ds.parse_file(f)
    p_a_ds.sort(('patient_id', 'id'))

    r_a_fields = r_a_ds.fields_
    p_a_fields = p_a_ds.fields_

    r_a_keys = set(r_a_ds.names_)
    p_a_keys = set(p_a_ds.names_)
    print('diff:', r_a_keys.difference(p_a_keys))
    print(r_a_keys)
    print(p_a_keys)
    r_a_fields = sorted(r_a_fields, key=lambda r: (r[2], r[1]))
    p_a_fields = sorted(p_a_fields, key=lambda p: (p[1], p[0]))

    diagnostic_row_keys = ['id', 'patient_id', 'created_at', 'updated_at', 'fatigue', 'fatigue_binary', 'tested_covid_positive']
    r_fns = {'created_at': datetime_to_seconds, 'updated_at': datetime_to_seconds}

    patients_with_disparities = set()
    r = 0
    p = 0
    while r < len(r_a_fields) and p < len(p_a_fields):
        rkey = (r_a_fields[r][2], r_a_fields[r][1])
        pkey = (p_a_fields[p][1], p_a_fields[p][0])
        if rkey < pkey:
            print(f'{r}, {p}: {rkey} not in python dataset')
            print_diagnostic_row('', r_a_ds, r_a_fields, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p_a_fields, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[r][2])
            patients_with_disparities.add(p_a_fields[p][1])
            r += 1
        elif pkey < rkey:
            print(f'{r}, {p}: {pkey} not in r dataset')
            print_diagnostic_row('', r_a_ds, r_a_fields, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p_a_fields, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[r][2])
            patients_with_disparities.add(p_a_fields[p][1])
            p += 1
        else:
            r += 1
            p += 1

        if r < len(r_a_fields):
            treatment = r_a_fields[r][r_a_ds.field_to_index('treatment')]
            if treatment not in ('NA', '', 'none'):
                print(r, treatment)

    r_a_fields = sorted(r_a_fields, key=lambda r: (r[2], r[4]))
    p_a_fields = sorted(p_a_fields, key=lambda p: (p[1], p[3]))

    for pd in patients_with_disparities:
        print(); print(pd)
        for ir, r in enumerate(r_a_fields):
            if r[2] == pd:
                print_diagnostic_row(f'r[ir]', r_a_ds, r_a_fields, ir, diagnostic_row_keys, fns=r_fns)
        for ip, p in enumerate(p_a_fields):
            if p[1] == pd:
                print_diagnostic_row(f'p[ip]', p_a_ds, p_a_fields, ip, diagnostic_row_keys)

    print('done')

def regression_test_patients(old_patients, new_patients):
    print(); print('regression test patients')
    print('old_patients:', old_patients)
    print('new_patients:', new_patients)
    with open(old_patients) as f:
        r_a_ds = dataset.Dataset(f)
        # r_a_ds.parse_file(f)
    r_a_ds.sort(('id',))
    with open(new_patients) as f:
        p_a_ds = dataset.Dataset(f)
        # p_a_ds.parse_file(f)
    p_a_ds.sort(('id',))

    r_a_fields = r_a_ds.fields_
    p_a_fields = p_a_ds.fields_

    r_a_keys = set(r_a_ds.names_)
    p_a_keys = set(p_a_ds.names_)
    print('r_a_keys:', r_a_keys)
    print('p_a_keys:', p_a_keys)

    r_a_fields = sorted(r_a_fields, key=lambda r: r[1])
    p_a_fields = sorted(p_a_fields, key=lambda p: p[0])

    print('checking for disparities')
    patients_with_disparities = set()
    r = 0
    p = 0
    while r < len(r_a_fields) and p < len(p_a_fields):
        rkey = r_a_fields[r][1]
        pkey = p_a_fields[p][0]
        if rkey < pkey:
            print(f'{r}, {p}: {rkey} not in python dataset')
            patients_with_disparities.add(r_a_fields[r][1])
            r += 1
        elif pkey < rkey:
            print(f'{r}, {p}: {pkey} not in r dataset')
            patients_with_disparities.add(p_a_fields[p][0])
            p += 1
        else:
            age_same = r_a_fields[r][r_a_ds.field_to_index('age')] == p_a_fields[p][p_a_ds.field_to_index('age')]
            print(r, p, age_same)
            r += 1
            p += 1

    for pd in patients_with_disparities:
        print(); print(pd)

def save_csv(pipeline_output, patient_data_out, assessment_data_out, data_schema):
    p_ds, p_fields, p_status, p_dest_fields, a_ds, a_fields, a_status, ra_fields, ra_status, res_fields, res_keys \
        = pipeline_output
    remaining_patients = set()
    for p in res_fields['patient_id']:
        remaining_patients.add(p)
    for ip, p in enumerate(p_fields):
        if p[0] not in remaining_patients:
            p_status[ip] |= FILTER_NOT_IN_FINAL_ASSESSMENTS

    print();
    print(f'writing patient data to {patient_data_out}')
    tstart = time.time()
    with open(patient_data_out, 'w') as f:
        dest_keys = list(p_dest_fields.keys())
        values = [None] * (len(p_ds.names_) + len(dest_keys))
        csvw = csv.writer(f)
        csvw.writerow(p_ds.names_ + dest_keys)
        for ir, r in enumerate(p_fields):
            if p_status[ir] == 0:
                for iv, v in enumerate(r):
                    values[iv] = v
                for iv in range(len(dest_keys)):
                    values[len(p_ds.names_) + iv] = p_dest_fields[dest_keys[iv]][ir]
                csvw.writerow(values)
    print(f'written to {patient_data_out} in {time.time() - tstart} seconds')

    functor_fields = {'created_at': datetime_to_seconds, 'updated_at': datetime_to_seconds}

    print(f'writing assessment data to {assessment_data_out}')
    tstart = time.time()
    with open(assessment_data_out, 'w') as f:
        csvw = csv.writer(f)
        headers = list(res_fields.keys())
        # TODO: constructed fields should be in their own collection; the ['day'] and +1 stuff is a temporary hack
        csvw.writerow(headers + ['day'])
        row_field_count = len(res_fields)
        row_values = [None] * (row_field_count + 1)
        for ir in range(len(res_fields['id'])):
            if ra_status[ir] == 0:
                for irh, rh in enumerate(headers):
                    if rh in functor_fields:
                        row_values[irh] = functor_fields[rh](res_fields[rh][ir])
                    elif rh in data_schema:
                        v_to_s = data_schema[rh].values_to_strings
                        if res_fields[rh][ir] >= len(v_to_s):
                            print(f'{res_fields[rh][ir]} is out of range for {v_to_s}')
                        try:
                            row_values[irh] = v_to_s[res_fields[rh][ir]]
                        except:
                            print('que?')
                    else:
                        row_values[irh] = res_fields[rh][ir]
                updated = res_fields['updated_at']
                row_values[-1] = f"{updated[ir][0:4]}-{updated[ir][5:7]}-{updated[ir][8:10]}"
                csvw.writerow(row_values)
                for irv in range(len(row_values)):
                    row_values[irv] = None
    print(f'written to {assessment_data_out} in {time.time() - tstart} seconds')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--regression_test', action='store_true')
    parser.add_argument('-t', '--territory', default=None,
                        help='the territory to filter the dataset on (runs on all territories if not set)')
    parser.add_argument('-p', '--patient_data',
                        help='the location and name of the patient data csv file')
    parser.add_argument('-po', '--patient_data_out',
                        help='the location and name of the output patient data csv file')
    parser.add_argument('-a', '--assessment_data',
                        help='the location and name of the assessment data csv file')
    parser.add_argument('-ao', '--assessment_data_out',
                        help='the location and name of the output assessment data csv file')
    args = parser.parse_args()
    if args.regression_test:
        regression_test_assessments('assessments_cleaned_short.csv', args.assessment_data)
        regression_test_patients('patients_cleaned_short.csv', args.patient_data)
    else:
        print(); print(f'cleaning')
        tstart = time.time()

        data_schema_version = 1
        data_schema = data_schemas.get_categorical_maps(data_schema_version)
        parsing_schema_version = 1
        parsing_schema = parsing_schemas.ParsingSchema(parsing_schema_version)
        pipeline_output = pipeline(args.patient_data, args.assessment_data,
                                   data_schema, parsing_schema,
                                   territory=args.territory)
        print(f'cleaning completed in {time.time() - tstart} seconds')

        save_csv(pipeline_output, args.patient_data_out, args.assessment_data_out, data_schema)
