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
from collections import defaultdict
import numpy as np


def enumerate_fields(filename):
    with open(filename) as f:
        csvf = csv.DictReader(f, delimiter=',', quotechar='"')
        fieldnames = csvf.fieldnames
        field_count = len(fieldnames) # len(line.split(','))
    return fieldnames


def parse_file(filename, strings=None, functor=None):
    if strings is None:
        strings = list()
    newline_at = 10
    lines_per_dot = 100000
    with open(filename) as f:
        csvf = csv.DictReader(f, delimiter=',', quotechar='"')
        fieldnames = csvf.fieldnames
        field_count = len(fieldnames) # len(line.split(','))
        print('field count =', field_count)
        print('field names =', fieldnames)

        csvf = csv.reader(f, delimiter=',', quotechar='"')

        for i, fields in enumerate(csvf):
            if len(fields) != field_count:
                print(len(fields), fields)
            if len(fields) == 1:
                print(f'warning: line {i} skipped as it is not data ({fields})')
                continue
            if functor is not None:
                functor(i, fields)
            if i > 0 and i % lines_per_dot == 0:
                if i % (lines_per_dot * newline_at) == 0:
                    print(f'. {i}')
                else:
                    print('.', end= '')
        if i % (lines_per_dot * newline_at) != 0:
            print(f' {i}')

    return strings


def read_header_and_n_lines(filename, n):
    with open(filename) as f:
        print(f.readline())
        for i in range(n):
            print(f.readline())


def build_histogram(dataset, field_index, filtered_records=None, tx=None):
#     dataset = sorted(dataset, dataset.field_index)
#     histogram = list()
#     histogram.append((dataset[0][1][1], 0))
#     for r in dataset:
#         if histogram[-1][0] != r[1][1]:
#             histogram.append((r[1][1], 1))
#         else:
#             histogram[-1] = (histogram[-1][0], histogram[-1][1] + 1)
    histogram = defaultdict(int)
    for ir, r in enumerate(dataset):
        if not filtered_records or not filtered_records[ir]:
            if tx is not None:
                value = tx(r[1][field_index])
            else:
                value = r[1][field_index]
            histogram[value] += 1
    hlist = list(histogram.items())
    del histogram
    return hlist


def build_histogram_from_list(dataset, filtered_records=None, tx=None):
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
    print('foo')
    def inner_(x):
        return x == default_value or x > f_min and x < f_max
    return inner_


def valid_range_fac_inc(f_min, f_max, default_value=''):
    def inner_(x):
        return x == default_value or x >= f_min and x <= f_max
    return inner_


def filter_fields(fields, filter_list, index, f_missing, f_bad, is_type_fn, type_fn, valid_fn):
    for ir, r in enumerate(fields):
        if not is_type_fn(r[1][index]):
            if f_missing != 0:
                filter_list[ir] |= f_missing
        else:
            value = type_fn(r[1][index])
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
        v = r[1][field_index]
        results[ir] = mapdict[v]
    return results


def field_to_index(dataset, field_name):
    return dataset.names_.index(field_name)


class PatientDataset:
    def __init__(self, fieldnames):
        self.names_ = fieldnames
        self.fields_ = list()

    def __call__(self, index, fields):
        if len(fields) < 2:
            print(f'{index}: fields is badly formatted ({fields})')
        self.fields_.append((index, fields))

    def show(self):
        for ir, r in enumerate(self.names_):
            print(f'{ir}-{r}')


def map_patient_ids(geoc, asmt, map_fn):
    g = 0
    a = 0
    while g < len(geoc) and a < len(asmt):
        gpid = geoc[g][1][0]
        apid = asmt[a][1][1]
        if gpid < apid:
            g += 1
        elif apid < gpid:
            a += 1
        else:
            map_fn(geoc, asmt, g, a)
            a += 1


def iterate_over_patient_assessments(fields, filter_status, visitor):
    cur_id = fields[0][1][1]
    cur_start = 0
    cur_end = 0
    i = 1
    while i < len(fields):
        while fields[i][1][1] == cur_id:
            cur_end = i
            i += 1
            if i >= len(fields):
                break;

        visitor(fields, filter_status, cur_start, cur_end)

        if i < len(fields):
            cur_start = i
            cur_end = cur_start
            cur_id = fields[i][1][1]


def datetime_to_seconds(dt):
   return f'{dt[0:4]}-{dt[5:7]}-{dt[8:10]} {dt[11:13]}:{dt[14:16]}:{dt[17:19]}'

def print_diagnostic_row(preamble, ds, fields, ir, keys, fns=None):
    if fns is None:
        fns = dict()
    indices = [field_to_index(ds, k) for k in keys]
    indexed_fns = [None if k not in fns else fns[k] for k in keys]
    values = [None] * len(indices)
    for ii, i in enumerate(indices):
        if indexed_fns[ii] is None:
            values[ii] = fields[ir][1][i]
        else:
            values[ii] = indexed_fns[ii](fields[ir][1][i])
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

categorical_maps = {
    'fatigue': {'': 0, 'no': 1, 'mild': 2, 'severe': 3},
    'shortness_of_breath': {'': 0, 'no': 1, 'mild': 2, 'significant': 3, 'severe': 4},
    'abdominal_pain': {'': 0, 'False': 1, 'True': 2},
    'chest_pain': {'': 0, 'False': 1, 'True': 2},
    'delirium': {'': 0, 'False': 1, 'True': 2},
    'diarrhoea': {'': 0, 'False': 1, 'True': 2},
    'fever': {'': 0, 'False': 1, 'True': 2},
    'headache': {'': 0, 'False': 1, 'True': 2},
    'hoarse_voice': {'': 0, 'False': 1, 'True': 2},
    'loss_of_smell': {'': 0, 'False': 1, 'True': 2},
    'persistent_cough': {'': 0, 'False': 1, 'True': 2},
    'skipped_meals': {'': 0, 'False': 1, 'True': 2},
    'sore_throat': {'': 0, 'False': 1, 'True': 2},
    'unusual_muscle_pains': {'': 0, 'False': 1, 'True': 2},
    'always_used_shortage': {'': 0, 'all_needed': 1, 'reused': 2},
    'have_used_PPE': {'': 0, 'never': 1, 'sometimes': 2, 'always': 3},
    'never_used_shortage': {'': 0, 'not_needed': 1, 'not_available': 2},
    'sometimes_used_shortage': {'': 0, 'all_needed': 1, 'reused': 2, 'not_enough': 3},
    'treated_patients_with_covid': {'': 0, 'no': 1, 'yes_suspected': 2,
                                    'yes_documented_suspected': 3, 'yes_documented': 4},
    'fatigue_binary': {'': 0, 'no': 1, 'mild': 2, 'severe': 2},
    'shortness_of_breath_binary': {'': 0, 'no': 1, 'mild': 2, 'significant': 2, 'severe': 2},
    'location': {'': 0, 'home': 1, 'hospital': 2, 'back_from_hospital': 3},
    'level_of_isolation': {'': 0, 'not_left_the_house': 1, 'rarely_left_the_house': 2, 'often_left_the_house': 3},
    'had_covid_test': {'': 0, 'False': 1, 'True': 2}
}

boolean_inv_map = ['na', 'False', 'True']
categorical_inv_maps = {
    'fatigue': ['na', 'no', 'mild', 'severe'],
    'shortness_of_breath': ['', 'no', 'mild', 'significant', 'severe'],
    'abdominal_pain': boolean_inv_map,
    'chest_pain': boolean_inv_map,
    'delirium': boolean_inv_map,
    'diarrhoea': boolean_inv_map,
    'fever': boolean_inv_map,
    'headache': boolean_inv_map,
    'hoarse_voice': boolean_inv_map,
    'loss_of_smell': boolean_inv_map,
    'persistent_cough': boolean_inv_map,
    'skipped_meals': boolean_inv_map,
    'sore_throat': boolean_inv_map,
    'unusual_muscle_pains': boolean_inv_map,
    'always_used_shortage': ['na', 'all_needed', 'reused'],
    'have_used_PPE': ['na', 'never', 'sometimes', 'always'],
    'never_used_shortage': ['na', 'not_needed', 'not_available'],
    'sometimes_used_shortage': ['na', 'all_needed', 'reused', 'not_enough'],
    'treated_patients_with_covid': ['na', 'no', 'yes_suspected',
                                    'yes_documented_suspected', 'yes_documented'],
    'fatigue_binary': ['na', 'False', 'True'],
    'shortness_of_breath_binary': ['na', 'False', 'True'],
    'location': ['na', 'home', 'hospital'],
    'level_of_isolation': ['na', 'not_left_the_house', 'rarely_left_the_house', 'often_left_the_house'],
    'had_covid_test': boolean_inv_map
}


def pipeline(patient_filename, assessment_filename, territory=None):

    print(); print();
    print('load patients')
    print('-------------')
    geoc_fieldnames = enumerate_fields(patient_filename)
    geoc_countdict = {'id': False, 'patient_id': False}
    geoc_ds = PatientDataset(geoc_fieldnames)
    parse_file(patient_filename, functor=geoc_ds)
    geoc_ds.show()


    print(); print()
    print('load assessments')
    print('----------------')
    asmt_fieldnames = enumerate_fields(assessment_filename)
    asmt_countdict = {'id': False, 'patient_id': False}
    asmt_ds = PatientDataset(asmt_fieldnames)
    parse_file(assessment_filename, functor=asmt_ds)
    asmt_ds.show()


    print(); print()
    print('generate dataset indices')
    print("------------------------")
    symptomatic_indices = [field_to_index(asmt_ds, c) for c in symptomatic_fields]
    print(symptomatic_indices)
    exposure_indices = [field_to_index(asmt_ds, c) for c in exposure_fields]
    print(exposure_indices)


    print(); print()
    print("pre-sort by patient id")
    print("----------------------")
    print("pre-sort patient data")
    geoc_fields = sorted(geoc_ds.fields_, key=lambda r: r[1][0])
    geoc_filter_status = [0] * len(geoc_fields)

    print("pre-sort assessment data")
    asmt_fields = sorted(asmt_ds.fields_, key=lambda r: (r[1][1], r[1][3]))
    asmt_filter_status = [0] * len(asmt_fields)


    if territory is not None:
        print(); print();
        print("filter patients from outside the territory of interest")
        print("------------------------------------------------------")

        country_code = field_to_index(geoc_ds, 'country_code')
        print(build_histogram(geoc_fields, country_code))
        for ir, r in enumerate(geoc_fields):
            if r[1][country_code] != territory:
                geoc_filter_status[ir] |= PFILTER_OTHER_TERRITORY
        print(f'other territories: filtered {count_flag_set(geoc_filter_status, PFILTER_OTHER_TERRITORY)} missing values')


    # print(); print()
    # print("filter patients with insufficient assessments")
    # print("---------------------------------------------")
    # patient_assessment_counts = defaultdict(int)
    # for a in asmt_fields:
    #     patient_assessment_counts[a[1][1]] += 1
    # patient_assessments = list(patient_assessment_counts.items())
    #
    # for ir, r in enumerate(geoc_fields):
    #     pid = r[1][0]
    #     if pid not in patient_assessment_counts:
    #         geoc_filter_status[ir] |= PFILTER_NO_ASSESSMENTS
    #     elif patient_assessment_counts[pid] == 1:
    #         geoc_filter_status[ir] |= PFILTER_ONE_ASSESSMENT
    # del patient_assessment_counts

#    def check_assessment_counts():
#        abp = defaultdict(int)
#        for ir in range(len(asmt_fields)):
#            abp[asmt_fields[ir][1][1]] += 1
#        abpt = list(abp.items())
#        count_multi = 0
#        print(len(abpt))
#        for it in abpt:
#            if it[1] > 1:
#                count_multi += 1
#        print('multiple assessments:', count_multi)
#    check_assessment_counts()
#
#    Restore once properly refactored and unit tested
#    i = 0
#    j = 0
#    missing_patient_ids = list()
#    while i < len(geoc_fields) and j < len(patient_assessments):
#        # print(i, j, geoc_fields[i][1][0], patient_assessments[j][0])
#        if geoc_fields[i][1][0] < patient_assessments[j][0]:
#            # patient has no assessments
#            geoc_filter_status[i] |= PFILTER_NO_ASSESSMENTS
#            i += 1
#
#        elif geoc_fields[i][1][0] > patient_assessments[j][0]:
#            # assessment patient id not in patient list
#            missing_patient_ids.append(patient_assessments[j][0])
#            j += 1
#        else:
#            # patient has assessments; but needs multiples
#            if patient_assessments[j][1] == 1:
#                geoc_filter_status[i] |= PFILTER_ONE_ASSESSMENT
#            i += 1
#            j += 1
#    # print(missing_patient_ids)
#    while i < len(geoc_fields):
#        geoc_filter_status[i] |= PFILTER_NO_ASSESSMENTS
#        i += 1
#
#    while j < len(patient_assessments):
#        missing_patient_ids.append(patient_assessments[j][0])
#        j += 1

    # print(count_flag_set(geoc_filter_status, PFILTER_NO_ASSESSMENTS))
    patient_ids = set()
    for r in geoc_fields:
        patient_ids.add(r[1][0])

    print('patients:', len(geoc_filter_status))
    print('patients with no assessments:',
          count_flag_set(geoc_filter_status, PFILTER_NO_ASSESSMENTS))
    print('patients with one assessment:',
          count_flag_set(geoc_filter_status, PFILTER_ONE_ASSESSMENT))
    print('patients with sufficient assessments:', geoc_filter_status.count(0))

    print(); print()
    print("patients")
    print("--------")

    print(); print("checking yob")

    filter_fields(geoc_fields, geoc_filter_status, field_to_index(geoc_ds, 'year_of_birth'),
                  FILTER_MISSING_YOB, FILTER_BAD_YOB, is_int, to_int, valid_range_fac_inc(MIN_YOB, MAX_YOB))
    print(f'yob: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_YOB)} missing values')
    print(f'yob: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_YOB)} bad values')
    yobs = build_histogram(geoc_fields, field_to_index(geoc_ds, 'year_of_birth'))
    print(f'yob: {len(yobs)} unique values')
    print(geoc_filter_status.count(0))

    print(); print("checking height")
    filter_fields(geoc_fields, geoc_filter_status, field_to_index(geoc_ds, 'height_cm'),
                  FILTER_MISSING_HEIGHT, FILTER_BAD_HEIGHT, is_float, to_float, valid_range_fac_inc(MIN_HEIGHT, MAX_HEIGHT))
    print(f'height: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_HEIGHT)} missing values')
    print(f'height: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_HEIGHT)} bad values')
    heights = build_histogram(geoc_fields, field_to_index(geoc_ds, 'height_cm')) #, tx=replace_if_invalid(-1.0))
    print(f'height: {len(heights)} unique values')
    print(geoc_filter_status.count(0))

    print(); print("checking weight")
    filter_fields(geoc_fields, geoc_filter_status, field_to_index(geoc_ds, 'weight_kg'),
                  FILTER_MISSING_WEIGHT, FILTER_BAD_WEIGHT, is_float, to_float, valid_range_fac_inc(MIN_WEIGHT, MAX_WEIGHT))
    print(f'weight: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_WEIGHT)} missing values')
    print(f'weight: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_WEIGHT)} bad values')
    weights = build_histogram(geoc_fields, field_to_index(geoc_ds, 'weight_kg')) #, tx=replace_if_invalid(-1.0))
    print(f'weight: {len(weights)} unique values')
    print(geoc_filter_status.count(0))

    print(); print("checking bmi")
    filter_fields(geoc_fields, geoc_filter_status, field_to_index(geoc_ds, 'bmi'), FILTER_MISSING_BMI, FILTER_BAD_BMI,
                 is_float, to_float, valid_range_fac_inc(MIN_BMI, MAX_BMI))
    print(f'bmi: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_BMI)} missing values')
    print(f'bmi: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_BMI)} bad values')
    bmis = build_histogram(geoc_fields, field_to_index(geoc_ds, 'bmi')) #, tx=replace_if_invalid(-1.0))
    print(f'bmi: {len(bmis)} unique values')

    print(); print('unfiltered patients:', geoc_filter_status.count(0))


    print(); print()
    print("assessments")
    print("-----------")

    dest_fields = dict()
    dest_keys = dict()

    clear_set_flag(asmt_filter_status, FILTERA_ALL)
    # print(); print("removing assessments for filtered patients")
    # def filter_assessments_on_patient_ids(geoc, asmt, g, a):
    #     if geoc_filter_status[g] != 0:
    #         asmt_filter_status[a] |= AFILTER_PATIENT_FILTERED

    # map_patient_ids(geoc_fields, asmt_fields, filter_assessments_on_patient_ids)

    patient_ids = set()
    for ir, r in enumerate(geoc_fields):
        if geoc_filter_status[ir] == 0:
            patient_ids.add(r[1][0])
    for ir, r in enumerate(asmt_fields):
        if r[1][1] not in patient_ids:
            asmt_filter_status[ir] |= AFILTER_PATIENT_FILTERED

    print('assessments filtered due to patient filtering:',
          count_flag_set(asmt_filter_status, FILTERA_ALL))

    print(); print("checking temperature")
    # convert temperature to C if F
    temperature_c = np.zeros((len(asmt_fields),), dtype=np.float)
    temperature_index = field_to_index(asmt_ds, 'temperature')
    for ir, r in enumerate(asmt_fields):
        t = r[1][temperature_index]
        if is_float(t):
            t = float(t)
            temperature_c[ir] = (t - 32) / 1.8 if t > MAX_TEMP else t
        else:
            temperature_c[ir] = 0.0

    filter_list(temperature_c, asmt_filter_status, FILTER_MISSING_TEMP, FILTER_BAD_TEMP,
                  is_float, to_float, valid_range_fac(MIN_TEMP, MAX_TEMP, 0.0))
    print(f'temperature: filtered {count_flag_set(asmt_filter_status, FILTER_MISSING_TEMP)} missing values')
    print(f'temperature: filtered {count_flag_set(asmt_filter_status, FILTER_BAD_TEMP)} bad values')
    dest_fields['temperature_C'] = temperature_c


    # print('level_of_isolation', build_histogram(asmt_fields, field_to_index(asmt_ds, 'level_of_isolation')))
    # print('had_covid_test', build_histogram(asmt_fields, field_to_index(asmt_ds, 'had_covid_test')))

    # had_covid_test
    # tested_covid_positive
    indices = [field_to_index(asmt_ds, c) for c in ('had_covid_test', 'tested_covid_positive')]
#    for ix in indices:
#        print(build_histogram(asmt_fields, ix))
    # for c in ('had_covid_test', 'tested_covid_positive'):
    #     print(f'{c}:', build_histogram(asmt_fields, field_to_index(asmt_ds, c)))

    for ir, r in asmt_fields:
        had_test = asmt_fields[ir][1][indices[0]]
        test_result = asmt_fields[ir][1][indices[1]]
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
    # for ck, cv in categorical_maps.items():
        print('symptomatic_field', s)
        cv = categorical_maps[s]
        print(f"symptomatic_field '{s}' to categorical")
        dest_fields[s] = to_categorical(asmt_fields, field_to_index(asmt_ds, s), np.uint8, cv)
        any_symptoms |= dest_fields[s] > 1
        print(np.count_nonzero(dest_fields[s] == True))
        print(np.count_nonzero(any_symptoms == True))

    for f in flattened_fields:
        cv = categorical_maps[f[1]]
        print(f[1], categorical_maps[f[1]])
        print(f"flattened_field '{f[0]}' to categorical field '{f[1]}'")
        dest_fields[f[1]] = to_categorical(asmt_fields, field_to_index(asmt_ds, f[0]), np.uint8, cv)
        any_symptoms |= dest_fields[f[1]] > 1

    for e in exposure_fields:
        cv = categorical_maps[e]
        print(f"exposure_field '{e}' to categorical")
        dest_fields[e] = to_categorical(asmt_fields, field_to_index(asmt_ds, e), np.uint8, cv)

    for m in miscellaneous_fields:
        cv = categorical_maps[m]
        print(f"miscellaneous_field '{m}' to categorical")
        print(build_histogram(asmt_fields, field_to_index(asmt_ds, m)))
        dest_fields[m] = to_categorical(asmt_fields, field_to_index(asmt_ds, m), np.uint8, cv)

    print(); print()
    print("filter inconsistent health status")
    print("---------------------------------")
    health_status_index = field_to_index(asmt_ds, 'health_status')
    health_status = build_histogram(asmt_fields, health_status_index)
    print(health_status)

    for ir, r in enumerate(asmt_fields):
        if r[1][health_status_index] == 'healthy' and any_symptoms[ir]:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_SYMPTOMS
        elif r[1][health_status_index] == 'not_healthy' and not any_symptoms[ir]:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_NO_SYMPTOMS

    for f in (FILTER_INCONSISTENT_SYMPTOMS, FILTER_INCONSISTENT_NO_SYMPTOMS):
        print(f'{assessment_flag_descs[f]}: {count_flag_set(asmt_filter_status, f)}')

    print(); print('unfiltered assessments:', asmt_filter_status.count(0))

    print('fatigue_binary')
    print(build_histogram_from_list(dest_fields['fatigue_binary']))

    # validate assessments per patient
    print(); print()
    print("validate covid progression")
    print("--------------------------")
    def validate_and_sanitise_covid_test_results_fac(results_key, results):
        valid_transitions = {
            '': ('', 'waiting', 'yes', 'no'),
            'waiting': ('', 'waiting', 'yes', 'no'),
            'no': ('', 'no'),
            'yes': ('', 'yes')
        }
        upgrades = {
            '': ('waiting', 'yes', 'no'),
            'waiting': ('yes', 'no'),
            'no': (),
            'yes':()
        }
        key_to_value = {
            '': 0,
            'waiting': 1,
            'no': 2,
            'yes': 3
        }

        tcp_index = field_to_index(asmt_ds, 'tested_covid_positive')

        def inner_(fields, filter_status, start, end):
            raw_results = list()
            for s in range(start, end+1):
                raw_results.append(fields[s][1][tcp_index])

            # validate the subrange
            invalid = False
            max_value = ''
            for j in range(start, end + 1):
                # allowable transitions
                #print(asmt_ds.names_)
                value = fields[j][1][tcp_index]
                if not value in valid_transitions[max_value]:
                    invalid = True
                    break
                if value in upgrades[max_value]:
                    max_value = value
                sanitised_covid_results[j] = key_to_value[max_value]

            if invalid:
                for j in range(start, end + 1):
                    sanitised_covid_results[j] = key_to_value[fields[j][1][tcp_index]]
                    filter_status[j] |= FILTER_INVALID_COVID_PROGRESSION

#            refined_results = sanitised_covid_results[start:end+1]
#            if refined_results.sum() > 0 or invalid:
#                print(raw_results, refined_results, filter_status[start] & FILTER_INVALID_COVID_PROGRESSION)

        return inner_

    sanitised_covid_results = np.ndarray((len(asmt_fields),), dtype=np.uint8)
    sanitised_covid_results_key = ['', 'waiting', 'no', 'yes']

    fn = validate_and_sanitise_covid_test_results_fac(sanitised_covid_results_key, sanitised_covid_results)
    iterate_over_patient_assessments(asmt_fields, asmt_filter_status, fn)

    print(f'{assessment_flag_descs[FILTER_INVALID_COVID_PROGRESSION]}:',
          count_flag_set(asmt_filter_status, FILTER_INVALID_COVID_PROGRESSION))

    dest_fields['tested_covid_positive'] = sanitised_covid_results
    dest_keys['tested_covid_positive'] = sanitised_covid_results_key

    print('fatigue_binary')
    print(build_histogram_from_list(dest_fields['fatigue_binary']))

    # sanity check
    print('field_len:', len(asmt_fields))
    print('dest_len:', len(dest_fields['fatigue_binary']))
    for ir, r in enumerate(asmt_fields):
        f = r[1][field_to_index(asmt_ds, 'fatigue')]
        fb = dest_fields['fatigue_binary'][ir]
        # if (f not in ('mild', 'severe') and fb == True) or (f in ('mild', 'severe') and fb == False):
        #    print('empty' if f == '' else f, fb)

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

    for dk, dv in dest_fields.items():
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

    print(build_histogram_from_list(sanitised_covid_results))

    print('fatigue_binary')
    print(build_histogram_from_list(remaining_dest_fields['fatigue_binary']))

    print(); print()
    print("quantise assessments by day")
    print("---------------------------")
    merged_row_count = 0

    def calculate_merged_field_count_fac():

        def inner_(fields, dummy, start, end):
            nonlocal merged_row_count
            results = list()
            for i in range(start + 1, end + 1):
                last_date_str = fields[i-1][1][3]
                last_date = (last_date_str[0:4], last_date_str[5:7], last_date_str[8:10])
                cur_date_str = fields[i][1][3]
                cur_date = (cur_date_str[0:4], cur_date_str[5:7], cur_date_str[8:10])
                if last_date == cur_date:
                    merged_row = fields[i][1].copy()
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

        def populate_row(self, source_fields, source_index, show_work=False):
            source_row = source_fields[source_index]
            if show_work:
                print(source_row, self.created_fields['fatigue_binary'][source_index])
                print(self.existing_field_indices)
                print(self.created_fields.keys())
            for e in self.existing_field_indices:
                self.resulting_fields[e[0]][self.rfindex] = source_row[1][e[1]]
            for ck, cv in self.created_fields.items():
                if show_work and ck == 'fatigue_binary':
                    if cv[source_index] != self.resulting_fields[ck][self.rfindex]:
                        print(self.resulting_fields[ck][self.rfindex], cv[source_index])
                self.resulting_fields[ck][self.rfindex] =\
                    max(self.resulting_fields[ck][self.rfindex], cv[source_index])
                if show_work and ck == 'fatigue_binary':
                    print(self.resulting_fields[ck][self.rfindex])
                #print(self.rfindex, cv[source_index], self.resulting_fields[ck][self.rfindex])

        def __call__(self, fields, dummy, start, end):
            show_work = fields[start][1][1][0:8] == '53bb6fb6'
            if show_work:
                for ir in range(start, end+1):
                    print(fields[ir][1][0], fields[ir][1][1], fields[ir][1][2], fields[ir][1][3],
                          fields[ir][1][field_to_index(asmt_ds, 'fatigue')], self.created_fields['fatigue_binary'][ir])

            rfstart = self.rfindex

            # write the first row to the current resulting field index
            prev_asmt = fields[start]
            prev_date_str = prev_asmt[1][3]
            prev_date = (prev_date_str[0:4], prev_date_str[5:7], prev_date_str[8:10])
            self.populate_row(fields, start, show_work)

            for i in range(start + 1, end + 1):
                cur_asmt = fields[i]
                cur_date_str = cur_asmt[1][3]
                cur_date = (cur_date_str[0:4], cur_date_str[5:7], cur_date_str[8:10])
                if cur_date != prev_date:
                    self.rfindex += 1
                if i % 1000000 == 0 and i > 0:
                    print('.')
                self.populate_row(fields, i, show_work)

                prev_asmt = cur_asmt
                prev_date_str = cur_date_str
                prev_date = cur_date

#            print(self.resulting_fields['patient_id'][rfstart], self.resulting_fields['patient_id'][self.rfindex],
#                  self.resulting_fields['tested_covid_positive'][rfstart:self.rfindex+1])
            if show_work:
                for ir in range(rfstart, self.rfindex+1):
                    print(self.resulting_fields['id'][ir], self.resulting_fields['patient_id'][ir],
                          self.resulting_fields['created_at'][ir], self.resulting_fields['updated_at'][ir],
                          self.resulting_fields['fatigue'][ir], self.resulting_fields['fatigue_binary'][ir])
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
    existing_field_indices = [(f, field_to_index(asmt_ds, f)) for f in existing_fields]

    resulting_fields = dict()
    for e in existing_fields:
        resulting_fields[e] = [None] * remaining_asmt_row_count
    for dk, dv in remaining_dest_fields.items():
        resulting_fields[dk] = np.zeros((remaining_asmt_row_count, ), dtype=dv.dtype)

    resulting_field_keys = dict()
    for dk, dv in dest_keys.items():
        resulting_field_keys[dk] = dv

    print(); print('fatigue_binary before merge')
    print(build_histogram_from_list(remaining_dest_fields['fatigue_binary']))

    unique_patients = set()
    for ir, r in enumerate(remaining_asmt_fields):
        unique_patients.add(r[1][1])
    print('unique patents in remaining assessments:', len(unique_patients))

    # sanity check
    print('remaining_field_len:', len(remaining_asmt_fields))
    print('remaining_dest_len:', len(remaining_dest_fields['fatigue_binary']))
    for ir, r in enumerate(remaining_asmt_fields):
        f = r[1][field_to_index(asmt_ds, 'fatigue')]
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
        unique_patients[r[1][1]] += 1
    print('unique patents in remaining assessments:', len(unique_patients))

#    print(); print()
#    print("filter patients with insufficient assessments")
#    print("---------------------------------------------")
#    patient_assessment_counts = defaultdict(int)
#    for a in asmt_fields:
#        patient_assessment_counts[a[1][1]] += 1
#    patient_assessments = list(patient_assessment_counts.items())
#
#    for ir, r in enumerate(geoc_fields):
#        pid = r[1][0]
#        if pid not in patient_assessment_counts:
#            geoc_filter_status[ir] |= PFILTER_NO_ASSESSMENTS
#        elif patient_assessment_counts[pid] == 1:
#            geoc_filter_status[ir] |= PFILTER_ONE_ASSESSMENT
#    del patient_assessment_counts

#    print('filter patients with only zero or one rows')
#    patient_ids = set()
#    for ir, r in enumerate(geoc_fields):
#        if geoc_filter_status[ir] == 0:
#            patient_ids.add(r[1][0])
#    for ir, r in enumerate(asmt_fields):
#        if r[1][1] not in patient_ids:
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

    print('done!')
    print('geoc_fields[0]:', geoc_fields[0])
    return (geoc_ds, geoc_fields, geoc_filter_status,
            asmt_ds, asmt_fields, asmt_filter_status,
            remaining_asmt_fields, remaining_asmt_filter_status,
            resulting_fields, resulting_field_keys)


def regression_test_assessments(old_assessments, new_assessments):
    r_a_fieldnames = enumerate_fields(old_assessments)
    p_a_fieldnames = enumerate_fields(new_assessments)
    r_a_ds = PatientDataset(r_a_fieldnames)
    p_a_ds = PatientDataset(p_a_fieldnames)
    parse_file(old_assessments, functor=r_a_ds)
    parse_file(new_assessments, functor=p_a_ds)

    r_a_fields = r_a_ds.fields_
    p_a_fields = p_a_ds.fields_

    r_a_keys = set(r_a_ds.names_)
    p_a_keys = set(p_a_ds.names_)
    print('diff:', r_a_keys.difference(p_a_keys))
    print(r_a_keys)
    print(p_a_keys)
    r_a_fields = sorted(r_a_fields, key=lambda r: (r[1][2], r[1][1]))
    p_a_fields = sorted(p_a_fields, key=lambda p: (p[1][1], p[1][0]))

    diagnostic_row_keys = ['id', 'patient_id', 'created_at', 'updated_at', 'fatigue', 'fatigue_binary']
    r_fns = {'created_at': datetime_to_seconds, 'updated_at': datetime_to_seconds}

    patients_with_disparities = set()
    r = 0
    p = 0
    while r < len(r_a_fields) and p < len(p_a_fields):
        #rkey = (r_a_fields[r][1][2], r_a_fields[r][1][4])
        #pkey = (p_a_fields[p][1][1], p_a_fields[p][1][3])
        rkey = (r_a_fields[r][1][2], r_a_fields[r][1][1])
        pkey = (p_a_fields[p][1][1], p_a_fields[p][1][0])
        if rkey < pkey:
            print(f'{r}, {p}: {rkey} not in python dataset')
            print_diagnostic_row('', r_a_ds, r_a_fields, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p_a_fields, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[r][1][2])
            patients_with_disparities.add(p_a_fields[p][1][1])
            r += 1
        elif pkey < rkey:
            print(f'{r}, {p}: {pkey} not in r dataset')
            print_diagnostic_row('', r_a_ds, r_a_fields, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p_a_fields, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[r][1][2])
            patients_with_disparities.add(p_a_fields[p][1][1])
            p += 1
        else:
            print(r, p,
                  r_a_fields[r][1][field_to_index(r_a_ds, 'fatigue_binary')],
                  p_a_fields[p][1][field_to_index(p_a_ds, 'fatigue_binary')])
            r += 1
            p += 1

    r_a_fields = sorted(r_a_fields, key=lambda r: (r[1][2], r[1][4]))
    p_a_fields = sorted(p_a_fields, key=lambda p: (p[1][1], p[1][3]))

    for pd in patients_with_disparities:
        print(); print(pd)
        for ir, r in enumerate(r_a_fields):
            if r[1][2] == pd:
                print_diagnostic_row(f'r[ir]', r_a_ds, r_a_fields, ir, diagnostic_row_keys, fns=r_fns)
        for ip, p in enumerate(p_a_fields):
            if p[1][1] == pd:
                print_diagnostic_row(f'p[ip]', p_a_ds, p_a_fields, ip, diagnostic_row_keys)


def regression_test_patients(old_patients, new_patients):
    print(); print('regression test patients')
    print('old_patients:', old_patients)
    print('new_patients:', new_patients)
    r_a_fieldnames = enumerate_fields(old_patients)
    p_a_fieldnames = enumerate_fields(new_patients)
    r_a_ds = PatientDataset(r_a_fieldnames)
    p_a_ds = PatientDataset(p_a_fieldnames)
    parse_file(old_patients, functor=r_a_ds)
    parse_file(new_patients, functor=p_a_ds)

    r_a_fields = r_a_ds.fields_
    p_a_fields = p_a_ds.fields_

    r_a_keys = set(r_a_ds.names_)
    p_a_keys = set(p_a_ds.names_)
    print('r_a_keys:', r_a_keys)
    print('p_a_keys:', p_a_keys)

    r_a_fields = sorted(r_a_fields, key=lambda r: r[1][1])
    p_a_fields = sorted(p_a_fields, key=lambda p: p[1][0])
    # print('r_a_fields[0]:', r_a_fields[0])
    #print('p_a_fields[0]:', p_a_fields[0])
    diagnostic_row_keys = ['id', 'created_at', 'updated_at']
    r_fns = {'created_at': datetime_to_seconds, 'updated_at': datetime_to_seconds}

    print('checking for disparities')
    patients_with_disparities = set()
    r = 0
    p = 0
    while r < len(r_a_fields) and p < len(p_a_fields):
        print(r, p)
        print(r, r_a_fields[r])
        print(p, p_a_fields[p])
        #rkey = (r_a_fields[r][1][2], r_a_fields[r][1][4])
        #pkey = (p_a_fields[p][1][1], p_a_fields[p][1][3])
        rkey = r_a_fields[r][1][1]
        pkey = p_a_fields[p][1][0]
        if rkey < pkey:
            print(f'{r}, {p}: {rkey} not in python dataset')
            print_diagnostic_row('', r_a_ds, r_a_fields, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p_a_fields, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[r][1][1])
            patients_with_disparities.add(p_a_fields[p][1][0])
            r += 1
        elif pkey < rkey:
            print(f'{r}, {p}: {pkey} not in r dataset')
            print_diagnostic_row('', r_a_ds, r_a_fields, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p_a_fields, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[r][1][1])
            patients_with_disparities.add(p_a_fields[p][1][0])
            p += 1
        else:
#            print(r, p,
#                  r_a_fields[r][1][field_to_index(r_a_ds, 'fatigue_binary')],
#                  p_a_fields[p][1][field_to_index(p_a_ds, 'fatigue_binary')])
            r += 1
            p += 1

#    r_a_fields = sorted(r_a_fields, key=lambda r: (r[1][2], r[1][4]))
#    p_a_fields = sorted(p_a_fields, key=lambda p: (p[1][1], p[1][3]))

    for pd in patients_with_disparities:
        print(); print(pd)
#        for ir, r in enumerate(r_a_fields):
#            if r[1][2] == pd:
#                print_diagnostic_row(f'r[ir]', r_a_ds, r_a_fields, ir, diagnostic_row_keys, fns=r_fns)
#        for ip, p in enumerate(p_a_fields):
#            if p[1][1] == pd:
#                print_diagnostic_row(f'p[ip]', p_a_ds, p_a_fields, ip, diagnostic_row_keys)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--regression_test', action='store_true')
    parser.add_argument('-t', '--territory', default=None,
                        help='the territory to filter the dataset on (runs on all territories if not set)')
    parser.add_argument('-p', '--patient_data',
                        help='the location and name of the patient data csv file')
    parser.add_argument('-a', '--assessment_data',
                        help='the location and name of the assessment data csv file')
    args = parser.parse_args()
    warning = ("Warning! This a pre-release version of the joinzoe data preparation pipeline. It has not been"
              " fully tested and is used very much at your own risk, with a full commitment by you to check"
              " correctness of output before relying on it for downstream analysis.")
    print(warning)
    if args.regression_test:
        regression_test_assessments('assessments_cleaned_short.csv', args.assessment_data)
#        regression_test_patients('patients_cleaned_short.csv', args.patient_data)
    else:
        p_ds, p_fields, p_status, a_ds, a_fields, a_status, ra_fields, ra_status, res_fields, res_keys =\
            pipeline(args.patient_data, args.assessment_data, territory=args.territory)

        remaining_patients = set()
        for p in res_fields['patient_id']:
            remaining_patients.add(p)
        for ip, p in enumerate(p_fields):
            if p[1][0] not in remaining_patients:
                p_status[ip] |= FILTER_NOT_IN_FINAL_ASSESSMENTS
        print(p_status.count(0))


        print(p_fields[0])
        with open('test_patients.csv', 'w') as f:
            csvw = csv.writer(f)
            csvw.writerow(p_ds.names_)
            print('len(p_fields):', len(p_fields))
            print('p_status.count(0):', p_status.count(0))
            for ir, r in enumerate(p_fields):
                if p_status[ir] == 0:
#                    print('row', ir, '= ', r[1])
                    csvw.writerow(r[1])

        functor_fields = {'created_at': datetime_to_seconds, 'updated_at': datetime_to_seconds}

        with open('test_assessments.csv', 'w') as f:
            csvw = csv.writer(f)
            headers = list(res_fields.keys())# + ['day']
            csvw.writerow(headers)
            row_field_count = len(res_fields)
            row_values = [None] * row_field_count
            print('ra_fields:', len(ra_fields))
            print('res_fields:', len(res_fields['id']))
            print(headers)
            for ir in range(len(res_fields['id'])):
                if ra_status[ir] == 0:
                    for irh, rh in enumerate(headers):
                        # if len(row_values) <= irh:
                        #     print(f'irh {irh} is out of range')
                        # if len(res_fields[rh]) <= ir:
                        #     print(f'ir {ir} is out of range')
                        if rh in functor_fields:
                            row_values[irh] = functor_fields[rh](res_fields[rh][ir])
                        elif rh in categorical_inv_maps:
                            # if res_fields[rh][ir] >= len(categorical_inv_maps[rh]):
                            #     print("oor:", rh, res_fields[rh][ir], categorical_inv_maps[rh])
                            row_values[irh] = categorical_inv_maps[rh][res_fields[rh][ir]]
                        else:
                            row_values[irh] = res_fields[rh][ir]
                    updated = res_fields['updated_at']
                    row_values[-1] = f"{updated[0:4]}-{updated[5:7]}-{updated[8:10]}"
                    csvw.writerow(row_values)
                    for irv in range(len(row_values)):
                        row_values[irv] = None

        # remaining_patients = defaultdict(int)
        # for ir in res_fields['patient_id']:
        #     remaining_patients[ir] += 1
        # for s in remaining_patients.items():
        #     print(s[0], s[1])

