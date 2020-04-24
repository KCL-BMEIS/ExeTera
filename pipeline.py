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
import time
from collections import defaultdict
import numpy as np

import dataset
import data_schemas
import filtered_field
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
        result = mapdict[v]
        results[ir] = result
    return results

def copy_field(field):
    if isinstance(field, list):
        return copy.deepcopy(field)
    else:
        return field.copy()



def map_between_categories(first_map, second_map):
    result_map = dict()
    for m in first_map.keys():
        result_map[first_map[m]] = second_map[m]
    return result_map


def to_categorical2(field, transform):
    results = np.zeros_like(field, dtype=field.dtype)
    for ir, r in enumerate(field):
        results[ir] = transform[r]
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
    patient_ids = fields[1]
    cur_id = patient_ids[0]
    cur_start = 0
    cur_end = 0
    i = 1
    while i < len(filter_status):
        while patient_ids[i] == cur_id:
            cur_end = i
            i += 1
            if i >= len(filter_status):
                break

        visitor(fields, filter_status, cur_start, cur_end)

        if i < len(filter_status):
            cur_start = i
            cur_end = cur_start
            cur_id = patient_ids[i]


def iterate_over_patient_assessments2(patient_ids, filter_status, visitor):
    cur_id = patient_ids[0]
    cur_start = 0
    cur_end = 0
    i = 1
    while i < len(patient_ids):
        while patient_ids[i] == cur_id:
            cur_end = i
            i += 1
            if i >= len(patient_ids):
                break

        visitor(cur_id, filter_status, cur_start, cur_end)

        if i < len(patient_ids):
            cur_start = i
            cur_end = cur_start
            cur_id = patient_ids[i]


def datetime_to_seconds(dt):
    return f'{dt[0:4]}-{dt[5:7]}-{dt[8:10]} {dt[11:13]}:{dt[14:16]}:{dt[17:19]}'


def print_diagnostic_row(preamble, ds, ir, keys, fns=None):
    if fns is None:
        fns = dict()
    # indices = [ds.field_to_index(k) for k in keys]
    # indexed_fns = [None if k not in fns else fns[k] for k in keys]
    values = [None] * len(keys)
    # for ii, i in enumerate(indices):
    for i, k in enumerate(keys):
        if not fns or k not in fns:
            values[i] = ds.value_from_fieldname(ir, k)
        else:
            values[i] = fns[k](ds.value_from_fieldname(ir, k))
        # if indexed_fns[ii] is None:
        #     values[ii] = fields[ir][i]
        # else:
        #     values[ii] = indexed_fns[ii](fields[ir][i])
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
miscellaneous_fields = ['health_status', 'location', 'level_of_isolation', 'had_covid_test']


def pipeline(patient_filename, assessment_filename, data_schema, parsing_schema, territory=None):

    categorical_maps = data_schema.assessment_categorical_maps
    # TODO: use proper logging throughout
    print(); print();
    print('load patients')
    print('-------------')
    with open(patient_filename) as f:
        geoc_ds = dataset.Dataset(f, data_schema.patient_categorical_maps, True)
    geoc_ds.sort(('id',))
    geoc_ds.show()


    print(); print()
    print('load assessments')
    print('----------------')
    with open(assessment_filename) as f:
        asmt_ds = dataset.Dataset(f, data_schema.assessment_categorical_maps, True)
    print('sorting patient ids')
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
    geoc_fields = geoc_ds.fields_
    geoc_filter_status = [0] * geoc_ds.row_count()
    print("geoc field count:", geoc_ds.row_count())

    print(); print("pre-sort assessment data")
    asmt_filter_status = [0] * asmt_ds.row_count()
    print("asmt field count:", asmt_ds.row_count())

    if territory is not None:
        print(); print();
        print("filter patients from outside the territory of interest")
        print("------------------------------------------------------")

        country_codes = geoc_fields[geoc_ds.field_to_index('country_code')]
        for ir, r in enumerate(country_codes):
            if r != territory:
                geoc_filter_status[ir] |= PFILTER_OTHER_TERRITORY
        print(f'other territories: filtered {count_flag_set(geoc_filter_status, PFILTER_OTHER_TERRITORY)} missing values')

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

    src_yobs = geoc_fields[geoc_ds.field_to_index('year_of_birth')]
    filter_list(src_yobs, geoc_filter_status, FILTER_MISSING_YOB, FILTER_BAD_YOB,
                is_int, to_int, valid_range_fac_inc(MIN_YOB, MAX_YOB))
    print(f'yob: filtered {count_flag_set(geoc_filter_status, FILTER_MISSING_YOB)} missing values')
    print(f'yob: filtered {count_flag_set(geoc_filter_status, FILTER_BAD_YOB)} bad values')
    print(geoc_filter_status.count(0))
    age = np.zeros_like(src_yobs, dtype=np.int16)
    for ir, r in enumerate(src_yobs):
        if geoc_filter_status[ir] & (FILTER_MISSING_YOB | FILTER_BAD_YOB):
            age[ir] = 0
        else:
            age[ir] = 2020 - to_int(r)
    ptnt_dest_fields['age'] = age

    fn_fac = parsing_schema.class_entries['validate_weight_height_bmi']
    src_weights = geoc_fields[geoc_ds.field_to_index('weight_kg')]
    src_heights = geoc_fields[geoc_ds.field_to_index('height_cm')]
    src_bmis = geoc_fields[geoc_ds.field_to_index('bmi')]
    fn = fn_fac(MIN_WEIGHT, MAX_WEIGHT, MIN_HEIGHT, MAX_HEIGHT, MIN_BMI, MAX_BMI,
                FILTER_MISSING_WEIGHT, FILTER_BAD_WEIGHT,
                FILTER_MISSING_HEIGHT, FILTER_BAD_HEIGHT,
                FILTER_MISSING_BMI, FILTER_BAD_BMI)
    height_clean, weight_clean, bmi_clean =\
        fn(src_weights, src_heights, src_bmis, geoc_filter_status)
    ptnt_dest_fields['weight_clean'] = weight_clean
    ptnt_dest_fields['height_clean'] = height_clean
    ptnt_dest_fields['bmi_clean'] = bmi_clean


    print(); print('unfiltered patients:', geoc_filter_status.count(0))


    print(); print()
    print("assessments")
    print("-----------")

    asmt_dest_fields = dict()
    asmt_dest_keys = dict()

    patient_ids = set()
    src_patient_ids = geoc_fields[0]
    for ir, r in enumerate(src_patient_ids):
        if geoc_filter_status[ir] == 0:
            patient_ids.add(r)
    src_asmt_patient_ids = asmt_ds.fields_[1]
    for ir, r in enumerate(src_asmt_patient_ids):
        if r not in patient_ids:
            asmt_filter_status[ir] |= AFILTER_PATIENT_FILTERED

    print('assessments filtered due to patient filtering:',
          count_flag_set(asmt_filter_status, FILTERA_ALL))

    print(); print("checking temperature")
    fn_fac = parsing_schema.class_entries['validate_temperature']
    fn = fn_fac(MIN_TEMP, MAX_TEMP, FILTER_MISSING_TEMP, FILTER_BAD_TEMP)
    temperature_c = fn(asmt_ds.field_by_name('temperature'), asmt_filter_status)
    asmt_dest_fields['temperature_C'] = temperature_c
    print(f'temperature: filtered {count_flag_set(asmt_filter_status, FILTER_BAD_TEMP)} bad values')

    print(); print("checking inconsistent test / test results fields")
    src_had_test = asmt_ds.field_by_name('had_covid_test')
    src_tested_covid_positive = asmt_ds.field_by_name('tested_covid_positive')

    for ir in range(asmt_ds.row_count()):
        had_test = src_had_test[ir]
        test_result = src_tested_covid_positive[ir]
        if had_test != 2 and test_result != 0:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_NOT_TESTED
        if had_test == 2 and test_result == 0:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_TESTED

    print(f'inconsistent_not_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_NOT_TESTED)} missing values')
    print(f'inconsistent_tested: filtered {count_flag_set(asmt_filter_status, FILTER_INCONSISTENT_TESTED)} missing values')
    print(); print('unfiltered assessments:', asmt_filter_status.count(0))


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

    print(build_histogram_from_list(asmt_ds.field_by_name('tested_covid_positive')))

    for f in flattened_fields:
        print(f"flattened_field '{f[0]}' to categorical field '{f[1]}'")
        remap = map_between_categories(categorical_maps[f[0]].strings_to_values,
                                       categorical_maps[f[1]].strings_to_values)
        asmt_dest_fields[f[1]] =\
            to_categorical2(asmt_ds.field_by_name(f[0]), remap)
        # TODO: this shouldn't be necessary as the fields were covered in 'symptomatic_fields'
        any_symptoms |= asmt_dest_fields[f[1]] > 1

    for e in exposure_fields:
        print(f"exposure_field '{e}' to categorical")
        asmt_dest_fields[e] = copy_field(asmt_ds.field_by_name(e))
    for m in miscellaneous_fields:
        print(f"miscellaneous_field '{m}' to categorical")
        asmt_dest_fields[m] = copy_field(asmt_ds.field_by_name(m))

    print(); print()
    print("filter inconsistent health status")
    print("---------------------------------")
    # TODO: can use the processed field
    src_health_status = asmt_ds.field_by_name('health_status')
    i_healthy = categorical_maps['health_status'].strings_to_values['healthy']
    i_not_healthy = categorical_maps['health_status'].strings_to_values['not_healthy']
    for ir in range(asmt_ds.row_count()):
        if src_health_status[ir] == i_healthy and any_symptoms[ir]:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_SYMPTOMS
        elif src_health_status[ir] == i_not_healthy and not any_symptoms[ir]:
            asmt_filter_status[ir] |= FILTER_INCONSISTENT_NO_SYMPTOMS

    for f in (FILTER_INCONSISTENT_SYMPTOMS, FILTER_INCONSISTENT_NO_SYMPTOMS):
        print(f'{assessment_flag_descs[f]}: {count_flag_set(asmt_filter_status, f)}')

    print(); print('unfiltered assessments:', asmt_filter_status.count(0))

    # validate assessments per patient
    print(); print()
    print("validate covid progression")
    print("--------------------------")
    sanitised_covid_results = np.ndarray(asmt_ds.row_count(), dtype=np.uint8)
    sanitised_covid_results_key = categorical_maps['tested_covid_positive'].values_to_strings[:]

    fn_fac = parsing_schema.class_entries['clean_covid_progression']
    fn = fn_fac(asmt_ds.field_by_name('tested_covid_positive'), asmt_filter_status,
                sanitised_covid_results_key, sanitised_covid_results,
                FILTER_INVALID_COVID_PROGRESSION)
    iterate_over_patient_assessments2(
        asmt_ds.field_by_name('patient_id'), asmt_filter_status, fn)

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

    class CalculateMergedFieldCount:
        def __init__(self, updated_ats):
            self.updated_ats = updated_ats
            self.merged_row_count = 0

        def __call__(self, patient_id, filter_status, start, end):
            for i in range(start + 1, end + 1):
                last_date_str = self.updated_ats[i-1]
                last_date = (last_date_str[0:4], last_date_str[5:7], last_date_str[8:10])
                cur_date_str = self.updated_ats[i]
                cur_date = (cur_date_str[0:4], cur_date_str[5:7], cur_date_str[8:10])
                if last_date == cur_date:
                    self.merged_row_count += 1


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
            for e in self.existing_field_indices:
                self.resulting_fields[e[0]][self.rfindex] = source_fields[e[1]][source_index]
            for ck, cv in self.created_fields.items():
                self.resulting_fields[ck][self.rfindex] =\
                    max(self.resulting_fields[ck][self.rfindex], cv[source_index])

        def __call__(self, fields, dummy, start, end):
            # write the first row to the current resulting field index
            prev_date_str = fields[3][start]
            prev_date = (prev_date_str[0:4], prev_date_str[5:7], prev_date_str[8:10])
            self.populate_row(fields, start)

            for i in range(start + 1, end + 1):
                cur_date_str = fields[3][i]
                cur_date = (cur_date_str[0:4], cur_date_str[5:7], cur_date_str[8:10])
                if cur_date != prev_date:
                    self.rfindex += 1
                if i % 1000000 == 0 and i > 0:
                    print('.')
                self.populate_row(fields, i)
                prev_date = cur_date

            # finally, update the resulting field index one more time
            self.rfindex += 1

    fn = CalculateMergedFieldCount(remaining_asmt_fields[asmt_ds.field_to_index('updated_at')])
    print(len(filter_map))
    print(len(remaining_asmt_filter_status))
    remaining_patient_ids = remaining_asmt_fields[asmt_ds.field_to_index('patient_id')]
    iterate_over_patient_assessments2(remaining_patient_ids, remaining_asmt_filter_status, fn)
    remaining_asmt_row_count = len(filter_map) - fn.merged_row_count
    print(f'{len(filter_map)} - {fn.merged_row_count} = {remaining_asmt_row_count}')

    existing_fields = ('id', 'patient_id', 'created_at', 'updated_at', 'version',
                       'country_code')
    existing_field_indices = [(f, asmt_ds.field_to_index(f)) for f in existing_fields]

    resulting_fields = dict()
    for e in existing_fields:
        resulting_fields[e] = [None] * remaining_asmt_row_count
    for dk, dv in remaining_dest_fields.items():
        resulting_fields[dk] = np.zeros((remaining_asmt_row_count, ), dtype=dv.dtype)

    resulting_field_keys = dict()
    for dk, dv in asmt_dest_keys.items():
        resulting_field_keys[dk] = dv

    print('remaining_dest_len:', len(remaining_dest_fields['fatigue_binary']))
    print('resulting_fields:', len(resulting_fields['patient_id']))

    print(build_histogram_from_list(remaining_dest_fields['tested_covid_positive']))
    merge = MergeAssessmentRows(resulting_fields, remaining_dest_fields, existing_field_indices)
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

    return (geoc_ds, geoc_fields, geoc_filter_status, ptnt_dest_fields,
            asmt_ds, asmt_ds.fields_, asmt_filter_status,
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

    diagnostic_row_keys = ['id', 'patient_id', 'created_at', 'updated_at', 'health_status', 'fatigue', 'fatigue_binary', 'had_covid_test', 'tested_covid_positive']
    r_fns = {'created_at': datetime_to_seconds, 'updated_at': datetime_to_seconds}

    patients_with_disparities = set()
    r = 0
    p = 0
    while r < len(r_a_fields) and p < len(p_a_fields):
        rkey = (r_a_fields[2][r], r_a_fields[1][r])
        pkey = (p_a_fields[1][p], p_a_fields[0][p])
        if rkey < pkey:
            print(f'{r}, {p}: {rkey} not in python dataset')
            print_diagnostic_row('', r_a_ds, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[2][r])
            patients_with_disparities.add(p_a_fields[1][p])
            r += 1
        elif pkey < rkey:
            print(f'{r}, {p}: {pkey} not in r dataset')
            print_diagnostic_row('', r_a_ds, r, diagnostic_row_keys, fns=r_fns)
            print_diagnostic_row('', p_a_ds, p, diagnostic_row_keys)
            patients_with_disparities.add(r_a_fields[2][r])
            patients_with_disparities.add(p_a_fields[1][p])
            p += 1
        else:
            r += 1
            p += 1

        if r < r_a_ds.row_count():
            treatment = r_a_fields[r_a_ds.field_to_index('treatment')][r]
            if treatment not in ('NA', '', 'none'):
                print(r, treatment)


    # r_a_fields = sorted(r_a_fields, key=lambda r: (r[2], r[4]))
    # p_a_fields = sorted(p_a_fields, key=lambda p: (p[1], p[3]))
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

    categorical_maps = data_schema.assessment_categorical_maps

    p_ds, p_fields, p_status, p_dest_fields,\
    a_ds, a_fields, a_status, ra_fields, ra_status, res_fields, res_keys \
        = pipeline_output
    remaining_patients = set()
    for p in res_fields['patient_id']:
        remaining_patients.add(p)
    for ip, p in enumerate(p_ds.field_by_name('id')):
        if p not in remaining_patients:
            p_status[ip] |= FILTER_NOT_IN_FINAL_ASSESSMENTS

    print();
    print(f'writing patient data to {patient_data_out}')
    tstart = time.time()
    with open(patient_data_out, 'w') as f:
        dest_keys = list(p_dest_fields.keys())
        values = [None] * (len(p_ds.names_) + len(dest_keys))
        csvw = csv.writer(f)
        csvw.writerow(p_ds.names_ + dest_keys)
        for ir in range(p_ds.row_count()):
            if p_status[ir] == 0:
                for iv, v in enumerate(p_fields):
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
                    elif rh in categorical_maps:
                        v_to_s = categorical_maps[rh].values_to_strings
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


# TODO: add json based config option
# TODO: add parsing schema option and default
# TODO: add data schema option and default
# TODO: provide specific schema element overrides (via json only)
# TODO: add flag to output values as categorical variables
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
    parser.add_argument('-ps', '--parsing_schema', default=1, type=int,
                        help='the schema number to use for parsing and cleaning data')
    args = parser.parse_args()

    if args.parsing_schema not in parsing_schemas.parsing_schemas:
        error_str = "the parsing schema must be one of {} for this version"
        print(error_str.format(parsing_schemas.parsing_schemas))
        exit(-1)


    if args.regression_test:
        regression_test_assessments('assessments_cleaned_short.csv', args.assessment_data)
        regression_test_patients('patients_cleaned_short.csv', args.patient_data)
    else:
        print(); print(f'cleaning')
        tstart = time.time()

        data_schema_version = 1
        data_schema = data_schemas.DataSchema(data_schema_version)
        parsing_schema_version = args.parsing_schema
        parsing_schema = parsing_schemas.ParsingSchema(parsing_schema_version)
        pipeline_output = pipeline(args.patient_data, args.assessment_data,
                                   data_schema, parsing_schema,
                                   territory=args.territory)
        print(f'cleaning completed in {time.time() - tstart} seconds')

        save_csv(pipeline_output, args.patient_data_out, args.assessment_data_out, data_schema)
