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

from datetime import datetime
import time
from collections import defaultdict

import numpy as np
import numba

from exetera.processing.age_from_year_of_birth import calculate_age_from_year_of_birth_fast
from exetera.processing.weight_height_bmi import weight_height_bmi_fast_1
from exetera.processing.inconsistent_symptoms import check_inconsistent_symptoms_1
from exetera.processing.temperature import validate_temperature_1
from exetera.processing.combined_healthcare_worker import combined_hcw_with_contact
from exetera.core import persistence
from exetera.core.persistence import DataStore
from exetera.core.session import Session
from exetera.core import readerwriter as rw
from exetera.core import utils

# TODO: replace datastore with session and readers/writers with fields

# TODO: postprocessing activities
# * assessment sort by (patient_id, created_at)
# * aggregate from assessments to patients
#   * was first unwell
#   * first assessment
#   * last assessment
#   * assessment count
#   * assessment index start
#   * assessment index end

def log(*a, **kwa):
    print(*a, **kwa)


def postprocess(dataset, destination, timestamp=None, flags=None):

    if flags is None:
        flags = set()

    do_daily_asmts = 'daily' in flags
    has_patients = 'patients' in dataset.keys()
    has_assessments = 'assessments' in dataset.keys()
    has_tests = 'tests' in dataset.keys()
    has_diet = 'diet' in dataset.keys()

    sort_enabled = lambda x: True
    process_enabled = lambda x: True

    sort_patients = sort_enabled(flags) and True
    sort_assessments = sort_enabled(flags) and True
    sort_tests = sort_enabled(flags) and True
    sort_diet = sort_enabled(flags) and True

    make_assessment_patient_id_fkey = process_enabled(flags) and True
    year_from_age = process_enabled(flags) and True
    clean_weight_height_bmi = process_enabled(flags) and True
    health_worker_with_contact = process_enabled(flags) and True
    clean_temperatures = process_enabled(flags) and True
    check_symptoms = process_enabled(flags) and True
    create_daily = process_enabled(flags) and do_daily_asmts
    make_patient_level_assessment_metrics = process_enabled(flags) and True
    make_patient_level_daily_assessment_metrics = process_enabled(flags) and do_daily_asmts
    make_new_test_level_metrics = process_enabled(flags) and True
    make_diet_level_metrics = True
    make_healthy_diet_index = True

    ds = DataStore(timestamp=timestamp)
    s = Session()

    # patients ================================================================

    sorted_patients_src = None

    if has_patients:
        patients_src = dataset['patients']

        write_mode = 'write'

        if 'patients' not in destination.keys():
            patients_dest = ds.get_or_create_group(destination, 'patients')
            sorted_patients_src = patients_dest

            # Patient sort
            # ============
            if sort_patients:
                duplicate_filter = \
                    persistence.filter_duplicate_fields(ds.get_reader(patients_src['id'])[:])

                for k in patients_src.keys():
                    t0 = time.time()
                    r = ds.get_reader(patients_src[k])
                    w = r.get_writer(patients_dest, k)
                    ds.apply_filter(duplicate_filter, r, w)
                    print(f"'{k}' filtered in {time.time() - t0}s")

                print(np.count_nonzero(duplicate_filter == True),
                      np.count_nonzero(duplicate_filter == False))
                sort_keys = ('id',)
                ds.sort_on(
                    patients_dest, patients_dest, sort_keys, write_mode='overwrite')

            # Patient processing
            # ==================
            if year_from_age:
                log("year of birth -> age; 18 to 90 filter")
                t0 = time.time()
                age = ds.get_numeric_writer(patients_dest, 'age', 'uint32',
                                            write_mode)
                age_filter = ds.get_numeric_writer(patients_dest, 'age_filter',
                                                   'bool', write_mode)
                age_16_to_90 = ds.get_numeric_writer(patients_dest, '16_to_90_years',
                                                     'bool', write_mode)
                print('year_of_birth:', patients_dest['year_of_birth'])
                for k in patients_dest['year_of_birth'].attrs.keys():
                    print(k, patients_dest['year_of_birth'].attrs[k])
                calculate_age_from_year_of_birth_fast(
                    ds, 16, 90,
                    patients_dest['year_of_birth'], patients_dest['year_of_birth_valid'],
                    age, age_filter, age_16_to_90,
                    2020)
                log(f"completed in {time.time() - t0}")

                print('age_filter count:', np.sum(patients_dest['age_filter']['values'][:]))
                print('16_to_90_years count:', np.sum(patients_dest['16_to_90_years']['values'][:]))

            if clean_weight_height_bmi:
                log("height / weight / bmi; standard range filters")
                t0 = time.time()

                weights_clean = ds.get_numeric_writer(patients_dest, 'weight_kg_clean',
                                                      'float32', write_mode)
                weights_filter = ds.get_numeric_writer(patients_dest, '40_to_200_kg',
                                                       'bool', write_mode)
                heights_clean = ds.get_numeric_writer(patients_dest, 'height_cm_clean',
                                                      'float32', write_mode)
                heights_filter = ds.get_numeric_writer(patients_dest, '110_to_220_cm',
                                                       'bool', write_mode)
                bmis_clean = ds.get_numeric_writer(patients_dest, 'bmi_clean',
                                                   'float32', write_mode)
                bmis_filter = ds.get_numeric_writer(patients_dest, '15_to_55_bmi',
                                                    'bool', write_mode)

                weight_height_bmi_fast_1(ds, 40, 200, 110, 220, 15, 55,
                                         None, None, None, None,
                                         patients_dest['weight_kg'], patients_dest['weight_kg_valid'],
                                         patients_dest['height_cm'], patients_dest['height_cm_valid'],
                                         patients_dest['bmi'], patients_dest['bmi_valid'],
                                         weights_clean, weights_filter, None,
                                         heights_clean, heights_filter, None,
                                         bmis_clean, bmis_filter, None)
                log(f"completed in {time.time() - t0}")

            if health_worker_with_contact:
                with utils.Timer("health_worker_with_contact field"):
                    #writer = ds.get_categorical_writer(patients_dest, 'health_worker_with_contact', 'int8')
                    combined_hcw_with_contact(ds,
                                              ds.get_reader(patients_dest['healthcare_professional']),
                                              ds.get_reader(patients_dest['contact_health_worker']),
                                              ds.get_reader(patients_dest['is_carer_for_community']),
                                              patients_dest, 'health_worker_with_contact')

    # assessments =============================================================

    sorted_assessments_src = None
    if has_assessments:
        assessments_src = dataset['assessments']
        if 'assessments' not in destination.keys():
            assessments_dest = ds.get_or_create_group(destination, 'assessments')
            sorted_assessments_src = assessments_dest

            if sort_assessments:
                sort_keys = ('patient_id', 'created_at')
                with utils.Timer("sorting assessments"):
                    ds.sort_on(
                        assessments_src, assessments_dest, sort_keys)

            if has_patients:
                if make_assessment_patient_id_fkey:
                    print("creating 'assessment_patient_id_fkey' foreign key index for 'patient_id'")
                    t0 = time.time()
                    patient_ids = ds.get_reader(sorted_patients_src['id'])
                    assessment_patient_ids =\
                        ds.get_reader(sorted_assessments_src['patient_id'])
                    assessment_patient_id_fkey =\
                        ds.get_numeric_writer(assessments_dest, 'assessment_patient_id_fkey', 'int64')
                    ds.get_index(patient_ids, assessment_patient_ids, assessment_patient_id_fkey)
                    print(f"completed in {time.time() - t0}s")

            if clean_temperatures:
                print("clean temperatures")
                t0 = time.time()
                temps = ds.get_reader(sorted_assessments_src['temperature'])
                temp_units = ds.get_reader(sorted_assessments_src['temperature_unit'])
                temps_valid = ds.get_reader(sorted_assessments_src['temperature_valid'])
                dest_temps = temps.get_writer(assessments_dest, 'temperature_c_clean', write_mode)
                dest_temps_valid =\
                    temps_valid.get_writer(assessments_dest, 'temperature_35_to_42_inclusive', write_mode)
                dest_temps_modified =\
                    temps_valid.get_writer(assessments_dest, 'temperature_modified', write_mode)
                validate_temperature_1(35.0, 42.0,
                                       temps, temp_units, temps_valid,
                                       dest_temps, dest_temps_valid, dest_temps_modified)
                print(f"temperature cleaning done in {time.time() - t0}")

            if check_symptoms:
                print('check inconsistent health_status')
                t0 = time.time()
                check_inconsistent_symptoms_1(ds, sorted_assessments_src, assessments_dest)
                print(time.time() - t0)

    # tests ===================================================================

    if has_tests:
        if sort_tests:
            tests_src = dataset['tests']
            tests_dest = ds.get_or_create_group(destination, 'tests')
            sort_keys = ('patient_id', 'created_at')
            ds.sort_on(tests_src, tests_dest, sort_keys)

    # diet ====================================================================

    if has_diet:
        diet_src = dataset['diet']
        if 'diet' not in destination.keys():
            diet_dest = ds.get_or_create_group(destination, 'diet')
            sorted_diet_src = diet_dest
            if sort_diet:
                sort_keys = ('patient_id', 'display_name', 'id')
                ds.sort_on(diet_src, diet_dest, sort_keys)


    if has_assessments:
        if do_daily_asmts:
            daily_assessments_dest = ds.get_or_create_group(destination, 'daily_assessments')




    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())
    print(dataset['assessments'].keys())
    print(dataset['tests'].keys())

    # write_mode = 'overwrite'
    write_mode = 'write'


    # Daily assessments
    # =================

    if has_assessments:
        if create_daily:
            print("generate daily assessments")
            patient_ids = ds.get_reader(sorted_assessments_src['patient_id'])
            created_at_days = ds.get_reader(sorted_assessments_src['created_at_day'])
            raw_created_at_days = created_at_days[:]

            if 'assessment_patient_id_fkey' in assessments_src.keys():
                patient_id_index = assessments_src['assessment_patient_id_fkey']
            else:
                patient_id_index = assessments_dest['assessment_patient_id_fkey']
            patient_id_indices = ds.get_reader(patient_id_index)
            raw_patient_id_indices = patient_id_indices[:]


            print("Calculating patient id index spans")
            t0 = time.time()
            patient_id_index_spans = ds.get_spans(fields=(raw_patient_id_indices,
                                                         raw_created_at_days))
            print(f"Calculated {len(patient_id_index_spans)-1} spans in {time.time() - t0}s")


            print("Applying spans to 'health_status'")
            t0 = time.time()
            default_behavour_overrides = {
                'id': ds.apply_spans_last,
                'patient_id': ds.apply_spans_last,
                'patient_index': ds.apply_spans_last,
                'created_at': ds.apply_spans_last,
                'created_at_day': ds.apply_spans_last,
                'updated_at': ds.apply_spans_last,
                'updated_at_day': ds.apply_spans_last,
                'version': ds.apply_spans_max,
                'country_code': ds.apply_spans_first,
                'date_test_occurred': None,
                'date_test_occurred_guess': None,
                'date_test_occurred_day': None,
                'date_test_occurred_set': None,
            }
            for k in sorted_assessments_src.keys():
                t1 = time.time()
                reader = ds.get_reader(sorted_assessments_src[k])
                if k in default_behavour_overrides:
                    apply_span_fn = default_behavour_overrides[k]
                    if apply_span_fn is not None:
                        apply_span_fn(patient_id_index_spans, reader,
                                      reader.get_writer(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    else:
                        print(f"  Skipping field {k}")
                else:
                    if isinstance(reader, rw.CategoricalReader):
                        ds.apply_spans_max(
                            patient_id_index_spans, reader,
                            reader.get_writer(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    elif isinstance(reader, rw.IndexedStringReader):
                        ds.apply_spans_concat(
                            patient_id_index_spans, reader,
                            reader.get_writer(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    elif isinstance(reader, rw.NumericReader):
                        ds.apply_spans_max(
                            patient_id_index_spans, reader,
                            reader.get_writer(daily_assessments_dest, k))
                        print(f"  Field {k} aggregated in {time.time() - t1}s")
                    else:
                        print(f"  No function for {k}")

            print(f"apply_spans completed in {time.time() - t0}s")


    # TODO - patient measure: assessments per patient

    if has_patients and has_assessments:
            if make_patient_level_assessment_metrics:
                if 'assessment_patient_id_fkey' in assessments_dest:
                    src = assessments_dest['assessment_patient_id_fkey']
                else:
                    src = assessments_src['assessment_patient_id_fkey']
                assessment_patient_id_fkey = ds.get_reader(src)
                # generate spans from the assessment-space patient_id foreign key
                spans = ds.get_spans(field=assessment_patient_id_fkey)

                ids = ds.get_reader(patients_dest['id'])

                # print('predicate_and_join')
                # acpp2 = ds.get_numeric_writer(patients_dest, 'assessment_count_2', 'uint32')
                # ds.predicate_and_join(ds.apply_spans_count, ids,
                #                              assessment_patient_id_fkey, None, acpp2, spans)

                print('calculate assessment counts per patient')
                t0 = time.time()
                writer = ds.get_numeric_writer(patients_dest, 'assessment_count', 'uint32')
                aggregated_counts = ds.aggregate_count(fkey_index_spans=spans)
                ds.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
                print(f"calculated assessment counts per patient in {time.time() - t0}")

                print('calculate first assessment days per patient')
                t0 = time.time()
                reader = ds.get_reader(sorted_assessments_src['created_at_day'])
                writer = ds.get_fixed_string_writer(patients_dest, 'first_assessment_day', 10)
                aggregated_counts = ds.aggregate_first(fkey_index_spans=spans, reader=reader)
                ds.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
                print(f"calculated first assessment days per patient in {time.time() - t0}")

                print('calculate last assessment days per patient')
                t0 = time.time()
                reader = ds.get_reader(sorted_assessments_src['created_at_day'])
                writer = ds.get_fixed_string_writer(patients_dest, 'last_assessment_day', 10)
                aggregated_counts = ds.aggregate_last(fkey_index_spans=spans, reader=reader)
                ds.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
                print(f"calculated last assessment days per patient in {time.time() - t0}")

                print('calculate maximum assessment test result per patient')
                t0 = time.time()
                reader = ds.get_reader(sorted_assessments_src['tested_covid_positive'])
                writer = reader.get_writer(patients_dest, 'max_assessment_test_result')
                max_result_value = ds.aggregate_max(fkey_index_spans=spans, reader=reader)
                ds.join(ids, assessment_patient_id_fkey, max_result_value, writer, spans)
                print(f"calculated maximum assessment test result in {time.time() - t0}")

    # TODO - patient measure: daily assessments per patient

    if has_assessments and do_daily_asmts and make_patient_level_daily_assessment_metrics:
        print("creating 'daily_assessment_patient_id_fkey' foreign key index for 'patient_id'")
        t0 = time.time()
        patient_ids = ds.get_reader(sorted_patients_src['id'])
        daily_assessment_patient_ids =\
            ds.get_reader(daily_assessments_dest['patient_id'])
        daily_assessment_patient_id_fkey =\
            ds.get_numeric_writer(daily_assessments_dest, 'daily_assessment_patient_id_fkey',
                                  'int64')
        ds.get_index(patient_ids, daily_assessment_patient_ids,
                     daily_assessment_patient_id_fkey)
        print(f"completed in {time.time() - t0}s")

        spans = ds.get_spans(
            field=ds.get_reader(daily_assessments_dest['daily_assessment_patient_id_fkey']))

        print('calculate daily assessment counts per patient')
        t0 = time.time()
        writer = ds.get_numeric_writer(patients_dest, 'daily_assessment_count', 'uint32')
        aggregated_counts = ds.aggregate_count(fkey_index_spans=spans)
        daily_assessment_patient_id_fkey =\
            ds.get_reader(daily_assessments_dest['daily_assessment_patient_id_fkey'])
        ds.join(ids, daily_assessment_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated daily assessment counts per patient in {time.time() - t0}")


    # TODO - new test count per patient:
    if has_tests and make_new_test_level_metrics:
        print("creating 'test_patient_id_fkey' foreign key index for 'patient_id'")
        t0 = time.time()
        patient_ids = ds.get_reader(sorted_patients_src['id'])
        test_patient_ids = ds.get_reader(tests_dest['patient_id'])
        test_patient_id_fkey = ds.get_numeric_writer(tests_dest, 'test_patient_id_fkey',
                                                     'int64')
        ds.get_index(patient_ids, test_patient_ids, test_patient_id_fkey)
        test_patient_id_fkey = ds.get_reader(tests_dest['test_patient_id_fkey'])
        spans = ds.get_spans(field=test_patient_id_fkey)
        print(f"completed in {time.time() - t0}s")

        print('calculate test_counts per patient')
        t0 = time.time()
        writer = ds.get_numeric_writer(patients_dest, 'test_count', 'uint32')
        aggregated_counts = ds.aggregate_count(fkey_index_spans=spans)
        ds.join(ids, test_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated test counts per patient in {time.time() - t0}")

        print('calculate test_result per patient')
        t0 = time.time()
        test_results = ds.get_reader(tests_dest['result'])
        writer = test_results.get_writer(patients_dest, 'max_test_result')
        aggregated_results = ds.aggregate_max(fkey_index_spans=spans, reader=test_results)
        ds.join(ids, test_patient_id_fkey, aggregated_results, writer, spans)
        print(f"calculated max_test_result per patient in {time.time() - t0}")

    if has_diet and make_diet_level_metrics:
        with utils.Timer("Making patient-level diet questions count", new_line=True):
            d_pids_ = s.get(diet_dest['patient_id']).data[:]
            d_pid_spans = s.get_spans(d_pids_)
            d_distinct_pids = s.apply_spans_first(d_pid_spans, d_pids_)
            d_pid_counts = s.apply_spans_count(d_pid_spans)
            p_diet_counts = s.create_numeric(patients_dest, 'diet_counts', 'int32')
            s.merge_left(left_on=s.get(patients_dest['id']).data[:], right_on=d_distinct_pids,
                         right_fields=(d_pid_counts,), right_writers=(p_diet_counts,))
