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

import copy
import csv
import datetime
import time
from collections import defaultdict

import numpy as np

from exetera.core import dataset, filtered_field, regression
from exetera.covidspecific import parsing_schemas
from exetera.processing.age_from_year_of_birth import CalculateAgeFromYearOfBirth
from exetera.processing.assessment_merge import CalculateMergedFieldCount, MergeAssessmentRows
from exetera.processing.inconsistent_symptoms import CheckInconsistentSymptoms
from exetera.processing.inconsistent_testing import CheckTestingConsistency
from exetera.core import load_schema

from exetera.core.utils import count_flag_empty, count_flag_set, build_histogram, map_between_categories, \
    to_categorical, print_diagnostic_row, valid_range_fac_inc, datetime_to_seconds, concatenate_maybe_strs
from exetera.covidspecific.utils import iterate_over_patient_assessments, iterate_over_patient_assessments2


def copy_field(field):
    if isinstance(field, list):
        return copy.deepcopy(field)
    else:
        return field.copy()


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


#patient limits
MIN_AGE = 16
MAX_AGE = 90
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
# PFILTER_NO_ASSESSMENTS = 0x2
# PFILTER_ONE_ASSESSMENT = 0x4
FILTER_MISSING_AGE = 0x8
FILTER_BAD_AGE = 0x10
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
FILTER_HEALTHY_BUT_SYMPTOMS = 0x40
FILTER_NOT_HEALTHY_BUT_NO_SYMPTOMS = 0x80
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
miscellaneous_fields = ['health_status', 'location', 'level_of_isolation', 'had_covid_test', 'tested_covid_positive']
existing_fields = ('id', 'patient_id', 'created_at', 'updated_at', 'version',
                   'country_code', 'treatment', 'other_symptoms')
custom_field_aggregators = {
    'treatment': concatenate_maybe_strs,
    'other_symptoms': concatenate_maybe_strs
}


def pipeline(patient_filename, assessment_filename, data_schema, parsing_schema, year, territories=None):

    early_filter = None
    if territories is not None:
        territories = tuple(territories.split(','))
        early_filter = ('country_code', lambda x: x in territories)

    # categorical_maps = data_schema.assessment_categorical_maps
    # TODO: use proper logging throughout
    print(); print()
    print('load patients')
    print('=============')
    p_categorical_maps = {k: v for k, v in data_schema['patients'].fields.items()
                          if v.strings_to_values is not None}
    with open(patient_filename) as f:
        geoc_ds = dataset.Dataset(f, p_categorical_maps,
                                  early_filter=early_filter,
                                  show_progress_every=1000000,
                                  stop_after=500000)
    print("sorting patients")
    geoc_ds.sort(('id',))
    geoc_ds.show()
    print("patient row count:", geoc_ds.row_count())


    print(); print()
    print('load assessments')
    print('================')
    a_categorical_maps = {k: v for k, v in data_schema['assessments'].fields.items()
                          if v.strings_to_values is not None}
    with open(assessment_filename) as f:
        asmt_ds = dataset.Dataset(f, a_categorical_maps,
                                  early_filter=early_filter,
                                  show_progress_every=1000000,
                                  stop_after=15000000)
    print('sorting assessments')
    asmt_ds.sort(('patient_id', 'updated_at'))
    asmt_ds.show()
    print("assessment row count:", asmt_ds.row_count())


    print(); print()
    print("pre-sort by patient id")
    print("======================")

    print("pre-sort patient data")
    geoc_filter_status = np.zeros(geoc_ds.row_count(), dtype=np.uint32)

    print(); print("pre-sort assessment data")
    asmt_filter_status = np.zeros(asmt_ds.row_count(), dtype=np.uint32)

    # if territories is not None:
    #     print(); print();
    #     print("filter patients from outside the territory of interest")
    #     print("------------------------------------------------------")
    #
    #     country_codes = geoc_ds.field_by_name('country_code')
    #     for ir, r in enumerate(country_codes):
    #         if r != territories:
    #             geoc_filter_status[ir] |= PFILTER_OTHER_TERRITORY
    #     print(f'other territories: filtered {count_flag_set(geoc_filter_status, PFILTER_OTHER_TERRITORY)} missing values')

    print('patients:', len(geoc_filter_status))

    print(); print()
    print("patients")
    print("--------")

    ptnt_dest_fields = dict()

    print()
    print("checking age")
    src_yobs = geoc_ds.field_by_name('year_of_birth')
    ages = np.zeros(len(src_yobs), dtype=np.uint32)
    fn = CalculateAgeFromYearOfBirth(FILTER_MISSING_AGE, FILTER_BAD_AGE,
                                     valid_range_fac_inc(MIN_AGE, MAX_AGE), year)
    fn(src_yobs, ages, geoc_filter_status)
    ptnt_dest_fields['age'] = ages
    print(f'age: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_AGE)} missing values')
    print(f'age: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_AGE)} bad values')


    print()
    print('checking weight / height / bmi')
    src_genders = geoc_ds.field_by_name('gender')
    src_weights = geoc_ds.field_by_name('weight_kg')
    src_heights = geoc_ds.field_by_name('height_cm')
    src_bmis = geoc_ds.field_by_name('bmi')

    fn_fac = parsing_schema.class_entries['validate_weight_height_bmi']
    fn = fn_fac(MIN_WEIGHT, MAX_WEIGHT, MIN_HEIGHT, MAX_HEIGHT, MIN_BMI, MAX_BMI,
                FILTER_MISSING_AGE, FILTER_BAD_AGE,
                FILTER_MISSING_WEIGHT, FILTER_BAD_WEIGHT,
                FILTER_MISSING_HEIGHT, FILTER_BAD_HEIGHT,
                FILTER_MISSING_BMI, FILTER_BAD_BMI)
    weight_clean, height_clean, bmi_clean =\
        fn(src_genders, ages, src_weights, src_heights, src_bmis, geoc_filter_status)
    ptnt_dest_fields['weight_clean'] = weight_clean
    ptnt_dest_fields['height_clean'] = height_clean
    ptnt_dest_fields['bmi_clean'] = bmi_clean
    print(f'weight: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_WEIGHT)} missing_values')
    print(f'weight: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_WEIGHT)} missing_values')
    print(f'height: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_HEIGHT)} missing_values')
    print(f'height: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_HEIGHT)} missing_values')
    print(f'bmi: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_BMI)} missing_values')
    print(f'bmi: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_BMI)} missing_values')

    print(); print('unfiltered patients:', count_flag_empty(geoc_filter_status))

    #TODO: save patients here, including filter flags



    print(); print()
    print("assessments")
    print("-----------")

    asmt_dest_fields = dict()
    asmt_dest_keys = dict()

    patient_ids = set()
    src_patient_ids = geoc_ds.field_by_name('id')
    for ir, r in enumerate(src_patient_ids):
        if geoc_filter_status[ir] == 0:
            patient_ids.add(r)
    src_asmt_patient_ids = asmt_ds.field_by_name('patient_id')
    for ir, r in enumerate(src_asmt_patient_ids):
        if r not in patient_ids:
            asmt_filter_status[ir] |= AFILTER_PATIENT_FILTERED

    print('assessments filtered due to patient filtering:',
          count_flag_set(asmt_filter_status, AFILTER_PATIENT_FILTERED))
    print('assessments filtered total:',
          count_flag_set(asmt_filter_status, FILTERA_ALL))

    print(); print("checking temperature")
    fn_fac = parsing_schema.class_entries['validate_temperature']
    fn = fn_fac(MIN_TEMP, MAX_TEMP, FILTER_MISSING_TEMP, FILTER_BAD_TEMP)
    temperature_c = fn(asmt_ds.field_by_name('temperature'), asmt_filter_status)
    asmt_dest_fields['temperature_C'] = temperature_c
    print(f'temperature: filtered {count_flag_set(asmt_filter_status, FILTER_BAD_TEMP)} bad values')

    # print(); print("checking inconsistent test / test results fields")
    # src_had_test = asmt_ds.field_by_name('had_covid_test')
    # src_tested_covid_positive = asmt_ds.field_by_name('tested_covid_positive')
    # fn = CheckTestingConsistency(FILTER_INCONSISTENT_NOT_TESTED, FILTER_INCONSISTENT_TESTED)
    # fn(src_had_test, src_tested_covid_positive, asmt_filter_status)
    # print(f'inconsistent_not_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_NOT_TESTED)} missing values')
    # print(f'inconsistent_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_TESTED)} missing values')


    print(); print('unfiltered assessments:', np.count_nonzero(asmt_filter_status == 0))


    print(); print()
    print("convert symptomatic, exposure, flattened and miscellaneous fields to bool")
    print("-------------------------------------------------------------------------")

    any_symptoms = np.zeros(asmt_ds.row_count(), dtype=np.bool)
    for s in symptomatic_fields:
        print(f"symptomatic_field '{s}' to categorical")

        asmt_dest_fields[s] = copy_field(asmt_ds.field_by_name(s))
        any_symptoms |= asmt_dest_fields[s] > 1
        print(np.count_nonzero(asmt_dest_fields[s] == True))
        print(np.count_nonzero(any_symptoms == True))

    print(build_histogram(asmt_ds.field_by_name('tested_covid_positive')))

    for f in flattened_fields:
        flattened = {'': 0, 'no': 1, 'mild': 2, 'significant': 2, 'severe': 2}
        print(f"flattened_field '{f[0]}' to categorical field '{f[1]}'")
        remap = map_between_categories(a_categorical_maps[f[0]].strings_to_values,
                                       flattened)
                                       # a_categorical_maps[f[1]].strings_to_values)
        asmt_dest_fields[f[1]] =\
            to_categorical(asmt_ds.field_by_name(f[0]), remap)
        # TODO: this shouldn't be necessary as the fields were covered in 'symptomatic_fields'
        any_symptoms |= asmt_dest_fields[f[1]] > 1

    for e in exposure_fields:
        print(f"exposure_field '{e}' to categorical")
        asmt_dest_fields[e] = copy_field(asmt_ds.field_by_name(e))
    for m in miscellaneous_fields:
        print(f"miscellaneous_field '{m}' to categorical")
        asmt_dest_fields[m] = copy_field(asmt_ds.field_by_name(m))


    print(); print()
    print("validate health status with symptoms")
    print("---------------------------------")
    fn = CheckInconsistentSymptoms(FILTER_HEALTHY_BUT_SYMPTOMS, FILTER_NOT_HEALTHY_BUT_NO_SYMPTOMS)
    # TODO: keys should be got from the dataset once it is loaded rather than referring to the categorical maps directly
    fn(asmt_dest_fields['health_status'], any_symptoms, asmt_filter_status,
       a_categorical_maps['health_status'].strings_to_values['healthy'],
       a_categorical_maps['health_status'].strings_to_values['not_healthy'])
    for f in (FILTER_HEALTHY_BUT_SYMPTOMS, FILTER_NOT_HEALTHY_BUT_NO_SYMPTOMS):
        print(f'{assessment_flag_descs[f]}: {count_flag_set(asmt_filter_status, f)}')

    print(); print('unfiltered assessments:', np.count_nonzero(asmt_filter_status == 0))


    # validate assessments per patient
    print(); print()
    print("validate covid progression")
    print("--------------------------")
    sanitised_hct_covid_results = np.ndarray(asmt_ds.row_count(), dtype=np.uint8)
    sanitised_covid_results = np.ndarray(asmt_ds.row_count(), dtype=np.uint8)
    sanitised_covid_results_key = \
        list(a_categorical_maps['tested_covid_positive'].values_to_strings.values())

    fn_fac = parsing_schema.class_entries['clean_covid_progression']
    fn = fn_fac(asmt_ds.field_by_name('had_covid_test'), asmt_ds.field_by_name('tested_covid_positive'),
                asmt_filter_status,
                sanitised_hct_covid_results, sanitised_covid_results,
                FILTER_INVALID_COVID_PROGRESSION)
    iterate_over_patient_assessments2(
        asmt_ds.field_by_name('patient_id'), asmt_filter_status, fn)

    print(f'{assessment_flag_descs[FILTER_INVALID_COVID_PROGRESSION]}:',
          count_flag_set(asmt_filter_status, FILTER_INVALID_COVID_PROGRESSION))

    asmt_dest_fields['tested_covid_positive_clean'] = sanitised_covid_results
    asmt_dest_keys['tested_covid_positive_clean'] = sanitised_covid_results_key
    asmt_dest_fields['had_covid_test_clean'] = sanitised_hct_covid_results
    asmt_dest_keys['had_covid_test_clean'] = sanitised_covid_results_key

    print(); print("checking inconsistent test / test results fields")
    fn = CheckTestingConsistency(FILTER_INCONSISTENT_NOT_TESTED, FILTER_INCONSISTENT_TESTED)
    fn(sanitised_hct_covid_results, sanitised_covid_results, asmt_filter_status)
    print(f'inconsistent_not_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_NOT_TESTED)} missing values')
    print(f'inconsistent_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_TESTED)} missing values')

    print('remaining assessments before squashing', np.count_nonzero(asmt_filter_status == 0))

    # create a new assessment space with only unfiltered rows
    print(); print()
    print("discard all filtered assessments")
    print("--------------------------------")
    remaining_asmt_fields = list()
    remaining_dest_fields = dict()

    filter_map = list()
    for ir, r in enumerate(asmt_filter_status):
        if r == 0:
            filter_map.append(ir)

    for ir, r in enumerate(asmt_ds.fields_):
        remaining_asmt_fields.append(filtered_field.FilteredField(r, filter_map))

    for k, v in asmt_dest_fields.items():
        remaining_dest_fields[k] = filtered_field.FilteredField(v, filter_map)

    print("remaining asmt fields: ", len(filter_map))
    remaining_asmt_filter_status = [0] * len(filter_map)

    print(); print()
    print("quantise assessments by day")
    print("---------------------------")

    fn = CalculateMergedFieldCount(remaining_asmt_fields[asmt_ds.field_to_index('updated_at')])
    print(len(filter_map))
    print(len(remaining_asmt_filter_status))
    remaining_patient_ids = remaining_asmt_fields[asmt_ds.field_to_index('patient_id')]
    iterate_over_patient_assessments2(remaining_patient_ids, remaining_asmt_filter_status, fn)
    remaining_asmt_row_count = len(filter_map) - fn.merged_row_count
    print(f'{len(filter_map)} - {fn.merged_row_count} = {remaining_asmt_row_count}')

    existing_field_indices = [(f, asmt_ds.field_to_index(f)) for f in existing_fields]

    resulting_fields = dict()
    for e in existing_fields:
        resulting_fields[e] = [None] * remaining_asmt_row_count
    for dk, dv in remaining_dest_fields.items():
        resulting_fields[dk] = np.zeros((remaining_asmt_row_count, ), dtype=dv.dtype)

    resulting_field_keys = dict()
    for dk, dv in asmt_dest_keys.items():
        resulting_field_keys[dk] = dv

    resulting_filter_status = np.zeros(remaining_asmt_row_count, dtype=np.uint32)

    print('remaining_dest_len:', len(remaining_dest_fields['fatigue_binary']))
    print('resulting_fields:', len(resulting_fields['patient_id']))

    print(build_histogram(remaining_dest_fields['tested_covid_positive_clean']))
    concat_field_indices =\
        [asmt_ds.field_to_index('other_symptoms'), asmt_ds.field_to_index('treatment')]
    merge = MergeAssessmentRows(concat_field_indices,
                                resulting_fields, remaining_dest_fields,
                                existing_field_indices, custom_field_aggregators,
                                remaining_asmt_filter_status, resulting_filter_status)
    iterate_over_patient_assessments(remaining_asmt_fields, remaining_asmt_filter_status, merge)
    print(merge.rfindex)

    unique_patients = defaultdict(int)
    for r in remaining_patient_ids:
        unique_patients[r] += 1
    print('unique patents in remaining assessments:', len(unique_patients))

    print(); print()
    print("filter summaries")
    print("----------------")

    print(); print('patient flags set')
    for v in patient_flag_descs.keys():
        print(f'{patient_flag_descs[v]}: {count_flag_set(geoc_filter_status, v)}')

    print(); print('assessment flags set')
    for v in assessment_flag_descs.keys():
        print(f'{assessment_flag_descs[v]}: {count_flag_set(asmt_filter_status, v)}')

    return (geoc_ds, geoc_filter_status, ptnt_dest_fields,
            asmt_ds, asmt_filter_status,
            remaining_asmt_fields, remaining_asmt_filter_status,
            resulting_fields, resulting_field_keys)


def regression_test_assessments(old_assessments, new_assessments):

    with open(old_assessments) as f:
        r_a_ds = dataset.Dataset(f)
    r_a_ds.sort(('patient_id', 'id'))
    with open(new_assessments) as f:
        p_a_ds = dataset.Dataset(f)
    p_a_ds.sort(('patient_id', 'id'))

    # r_a_fields = r_a_ds.fields_
    # p_a_fields = p_a_ds.fields_

    r_a_keys = set(r_a_ds.names_)
    p_a_keys = set(p_a_ds.names_)
    print(r_a_keys)
    print(p_a_keys)
    print('keys only in r:', r_a_keys.difference(p_a_keys))
    print('keys only in p:', p_a_keys.difference(r_a_keys))

    diagnostic_row_keys = ['id', 'patient_id', 'created_at', 'updated_at', 'health_status', 'fatigue', 'fatigue_binary', 'had_covid_test', 'tested_covid_positive']
    comparison_keys = diagnostic_row_keys
    r_fns = {'created_at': regression.datetime_compare_to_secs,
             'updated_at': regression.datetime_compare_to_secs}

    patients_with_disparities = set()
    r = 0
    p = 0
    r_ids = r_a_ds.field_by_name('id')
    r_pids = r_a_ds.field_by_name('patient_id')
    r_upda = r_a_ds.field_by_name('day')
    r_hct = r_a_ds.field_by_name('had_covid_test')
    r_tcp = r_a_ds.field_by_name('tested_covid_positive')

    p_ids = p_a_ds.field_by_name('id')
    p_pids = p_a_ds.field_by_name('patient_id')
    p_upda = p_a_ds.field_by_name('day')
    p_hct = p_a_ds.field_by_name('had_covid_test')
    p_tcp = p_a_ds.field_by_name('tested_covid_positive')
    while r < r_a_ds.row_count() and p < p_a_ds.row_count():
        rkey = (r_pids[r], r_upda[r])
        pkey = (p_pids[p], p_upda[p])
        if rkey < pkey:
            print(f'{r}, {p}: {rkey} not in python dataset')
            print_diagnostic_row('', r_a_ds, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p, diagnostic_row_keys)
            patients_with_disparities.add(r_pids[r])
            r += 1
        elif pkey < rkey:
            print(f'{r}, {p}: {pkey} not in r dataset')
            print_diagnostic_row('', r_a_ds, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p, diagnostic_row_keys)
            patients_with_disparities.add(p_pids[p])
            p += 1
        else:
            disparities = regression.check_row(r_a_ds, r, p_a_ds, p, comparison_keys, r_fns)
            if disparities != None:
                print(p_ids[p], ','.join(disparities))
            r += 1
            p += 1

        # if r < r_a_ds.row_count():
        #     treatment = r_a_fields[r_a_ds.field_to_index('treatment')][r]
        #     if treatment not in ('NA', '', 'none'):
        #         print(r, treatment)


    r_a_ds.sort(('patient_id', 'updated_at'))
    p_a_ds.sort(('patient_id', 'updated_at'))

    for pd in patients_with_disparities:
        print(); print(pd)
        for ir in range(r_a_ds.row_count()):
            if r_a_ds.value_from_fieldname(ir, 'patient_id') == pd:
                print_diagnostic_row(f'r[ir]', r_a_ds, ir, diagnostic_row_keys, fns=r_fns)
        for ip in range(p_a_ds.row_count()):
            if p_a_ds.value_from_fieldname(ir, 'patient_id') == pd:
                print_diagnostic_row(f'p[ip]', p_a_ds, ip, diagnostic_row_keys)

    print('done')


def regression_test_patients(old_patients, new_patients):
    print(); print('regression test patients')
    print('old_patients:', old_patients)
    print('new_patients:', new_patients)
    with open(old_patients) as f:
        r_a_ds = dataset.Dataset(f)
    r_a_ds.sort(('id',))
    with open(new_patients) as f:
        p_a_ds = dataset.Dataset(f)
    p_a_ds.sort(('id',))

    r_a_keys = set(r_a_ds.names_)
    p_a_keys = set(p_a_ds.names_)
    print('r_a_keys:', r_a_keys)
    print('p_a_keys:', p_a_keys)
    print('keys only in r:', r_a_keys.difference(p_a_keys))
    print('keys only in p:', p_a_keys.difference(r_a_keys))

    print('checking for disparities')
    patients_with_disparities = set()
    r = 0
    p = 0
    r_ids = r_a_ds.field_by_name('id')
    r_ages = r_a_ds.field_by_name('age')
    r_weight = r_a_ds.field_by_name('weight_kg')
    r_height = r_a_ds.field_by_name('height_cm')
    r_bmi = r_a_ds.field_by_name('bmi')
    p_ids = p_a_ds.field_by_name('id')
    p_ages = p_a_ds.field_by_name('age')
    p_weight = p_a_ds.field_by_name('weight_clean')
    p_height = p_a_ds.field_by_name('height_clean')
    p_bmi = p_a_ds.field_by_name('bmi_clean')
    while r < r_a_ds.row_count() and p < p_a_ds.row_count():
        rkey = r_ids[r]
        pkey = p_ids[p]
        if rkey < pkey:
            print(f'{r}, {p}: {rkey} not in python dataset')
            patients_with_disparities.add(rkey)
            r += 1
        elif pkey < rkey:
            print(f'{r}, {p}: {pkey} not in r dataset')
            patients_with_disparities.add(pkey)
            p += 1
        else:
            age_same = r_ages[r] == p_ages[p]
            weight_same = abs(float(r_weight[r]) - float(p_weight[p])) < 0.00001
            height_same = abs(float(r_height[r]) - float(p_height[p])) < 0.00001
            bmi_same = abs(float(r_bmi[r]) - float(p_bmi[p])) < 0.00001
            if not age_same or not weight_same or not height_same or not bmi_same:
                print(r, p,
                      r_ids[r], p_ids[p],
                      'na' if r_ages[r] is '' else r_ages[r], p_ages[p] if '' else p_ages[p],
                      'na' if r_weight[r] is '' else r_weight[r], 'na' if p_weight[p] is '' else p_weight[p],
                      'na' if r_height[r] is '' else r_height[r], 'na' if p_height[p] is '' else p_height[p],
                      'na' if r_bmi[r] is '' else r_bmi[r], 'na' if p_bmi[p] is '' else p_bmi[p]
                      )
            r += 1
            p += 1

    for pd in patients_with_disparities:
        print(); print(pd)
    print('checking_for_disparities: done')


def save_csv(pipeline_output, patient_data_out, assessment_data_out, data_schema):
    a_categorical_maps = \
        {k: v for k, v in data_schema["assessments"].fields.items()
         if v.strings_to_values is not None and v.out_of_range_label is None}

    p_ds, p_status, p_dest_fields,\
    a_ds, a_status, ra_fields, ra_status, res_fields, res_keys \
        = pipeline_output
    remaining_patients = set()
    for p in res_fields['patient_id']:
        remaining_patients.add(p)
    for ip, p in enumerate(p_ds.field_by_name('id')):
        if p not in remaining_patients:
            p_status[ip] |= FILTER_NOT_IN_FINAL_ASSESSMENTS

    print()
    print(f'writing patient data to {patient_data_out}')
    tstart = time.time()
    with open(patient_data_out, 'w') as f:
        dest_keys = list(p_dest_fields.keys())
        values = [None] * (len(p_ds.names_) + len(dest_keys))
        csvw = csv.writer(f)
        csvw.writerow(p_ds.names_ + dest_keys)
        for ir in range(p_ds.row_count()):
            if p_status[ir] == 0:
                for iv, v in enumerate(p_ds.fields_):
                    values[iv] = v[ir]
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
                    elif rh in a_categorical_maps:
                        v_to_s = a_categorical_maps[rh].values_to_strings
                        if res_fields[rh][ir] >= len(v_to_s):
                            print(f'{res_fields[rh][ir]} is out of range for {v_to_s}')
                        row_values[irh] = v_to_s[res_fields[rh][ir]]
                    else:
                        row_values[irh] = res_fields[rh][ir]
                updated = res_fields['updated_at']
                row_values[-1] = f"{updated[ir][0:4]}-{updated[ir][5:7]}-{updated[ir][8:10]}"
                csvw.writerow(row_values)
                for irv in range(len(row_values)):
                    row_values[irv] = None
    print(f'written to {assessment_data_out} in {time.time() - tstart} seconds')


def save_patient_npys(pipeline_output,
                      patient_data_out, assessment_data_out, test_data_out,
                      data_schema):

    categorical_maps = data_schema.assessment_categorical_maps

    p_ds, p_status, p_dest_fields,\
    a_ds, a_status, ra_fields, ra_status, res_fields, res_keys \
        = pipeline_output


# TODO: add json based config option
# TODO: add parsing schema option and default
# TODO: add data schema option and default
# TODO: provide specific schema element overrides (via json only)
# TODO: add flag to output values as categorical variables
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='v0.1.9')
    parser.add_argument('-r', '--regression_test', action='store_true')
    parser.add_argument('-te', '--territories', default=None,
                        help='the territories to filter the dataset on (runs on all territories if not set)')
    parser.add_argument('-p', '--patient_data',
                        help='the location and name of the patient data csv file')
    parser.add_argument('-po', '--patient_data_out',
                        help='the location and name of the output patient data csv file')
    parser.add_argument('-a', '--assessment_data',
                        help='the location and name of the assessment data csv file')
    parser.add_argument('-ao', '--assessment_data_out',
                        help='the location and name of the output assessment data csv file')
    parser.add_argument('-ps', '--parsing_schema', default=1, type=int,
                        help='the schema number to use for parsing and cleaning data')
    parser.add_argument('-y', '--year', default=datetime.datetime.now().year, type=int)
    args = parser.parse_args()

    if args.parsing_schema not in parsing_schemas.parsing_schemas:
        error_str = "the parsing schema must be one of {} for this version"
        print(error_str.format(parsing_schemas.parsing_schemas))
        exit(-1)

    if args.regression_test:
        regression_test_assessments('assessments_cleaned_short.csv', args.assessment_data)
        regression_test_patients('patients_cleaned_short.csv', args.patient_data)
    else:
        tstart = time.time()

        data_schema_version = 1
        early_filter = None
        parsing_schema_version = args.parsing_schema
        parsing_schema = parsing_schemas.ParsingSchema(parsing_schema_version)
        pipeline_output = pipeline(args.patient_data, args.assessment_data,
                                   parsing_schema, args.year,
                                   territories=args.territories)
        print(f'cleaning completed in {time.time() - tstart} seconds')

        save_csv(pipeline_output, args.patient_data_out, args.assessment_data_out)
