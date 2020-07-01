from datetime import datetime, timezone
import time
import math

import numpy as np
import h5py
from numba import jit, njit

import utils
from processing.age_from_year_of_birth import calculate_age_from_year_of_birth_fast
from processing.weight_height_bmi import weight_height_bmi_fast_1
from processing.inconsistent_symptoms import check_inconsistent_symptoms_1
from processing.temperature import validate_temperature_1
import data_schemas
import parsing_schemas
import persistence


# TODO: hard filter
# TODO: journalling for hdf5 robustness

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

def postprocess(dataset, destination, data_schema, process_schema, timestamp=None, flags='all'):

    chunksize = 1 << 20

    patients_src = dataset['patients']
    # patients_dest = destination.create_group('patients')
    patients_dest = persistence.get_or_create_group(destination, 'patients')
    assessments_src = dataset['assessments']
    # assessments_dest = destination.create_group('assessments')
    assessments_dest = persistence.get_or_create_group(destination, 'assessments')
    daily_assessments_dest = persistence.get_or_create_group(destination, 'daily_assessments')
    tests_src = dataset['tests']
    tests_dest = persistence.get_or_create_group(destination, 'tests')

    sort_enabled = lambda x: x in ('sort', 'all')
    process_enabled = lambda x: x in ('process', 'all')
    sort_patients = sort_enabled(flags) and True
    sort_assessments = sort_enabled(flags) and True
    sort_tests = sort_enabled(flags) and True

    make_assessment_patient_id_index = process_enabled(flags) and True
    year_from_age = process_enabled(flags) and True
    clean_weight_height_bmi = process_enabled(flags) and True
    clean_temperatures = process_enabled(flags) and True
    check_symptoms = process_enabled(flags) and True
    create_daily = process_enabled(flags) and False
    make_patient_level_assessment_metrics = process_enabled(flags) and True

    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())
    print(dataset['assessments'].keys())

    # write_mode = 'overwrite'
    write_mode = 'write'

    # Sorting
    # #######

    if sort_patients:
        sort_keys = ('id',)
        persistence.sort_on(
            patients_src, patients_dest, sort_keys, timestamp=timestamp, write_mode=write_mode)

    if sort_assessments:
        sort_keys = ('patient_id', 'created_at')
        persistence.sort_on(
            assessments_src, assessments_dest, sort_keys, timestamp=timestamp)

        # print("creating 'patient_index' foreign key index for 'patient_id'")
        # t0 = time.time()
        # patient_ids = persistence.get_reader_from_field(patients_dest['id'])
        # assessment_patient_ids =\
        #     persistence.get_reader_from_field(assessments_dest['patient_id'])
        # assessment_patient_id_index =\
        #     assessment_patient_ids.getwriter(assessments_dest, 'patient_index', timestamp)
        # persistence.get_index(patient_ids, assessment_patient_ids, assessment_patient_id_index)
        # print(f"completed in {time.time() - t0}s")

        print("checking sort order")
        t0 = time.time()
        raw_patient_ids = persistence.NewFixedStringReader(assessments_dest['patient_id'])[:]
        raw_created_ats = persistence.NewTimestampReader(assessments_dest['created_at'])[:]
        last_pid = raw_patient_ids[0]
        last_cat = raw_created_ats[0]
        duplicates = 0
        for i_r in range(1, len(raw_patient_ids)):
            pid = raw_patient_ids[i_r]
            cat = raw_created_ats[i_r]
            if (last_pid, last_cat) == (pid, cat):
                duplicates += 1
            if (last_pid, last_cat) > (pid, cat):
                print(i_r,
                      last_pid, datetime.fromtimestamp(last_cat),
                      pid, datetime.fromtimestamp(cat))
            last_pid = pid
            last_cat = cat
            # if i_r < 1000:
            #     print(i_r, pid, datetime.fromtimestamp(cat))
        print(f"sort order checked({duplicates} duplicate row keys found) in {time.time() - t0}")

    if sort_tests:
        sort_keys = ('patient_id', 'created_at')
        persistence.sort_on(
            tests_src, tests_dest, sort_keys, timestamp=timestamp)


    # Processing
    # ##########

    sorted_patients_src = patients_dest if sort_patients else patients_src
    sorted_assessments_src = assessments_dest if sort_assessments else assessments_src

    # Patient processing
    # ==================

    if year_from_age:
        log("year of birth -> age; 18 to 90 filter")
        t0 = time.time()
        age = persistence.NewNumericWriter(patients_dest, chunksize, 'age', timestamp, 'uint32',
                                           write_mode)
        age_filter = persistence.NewNumericWriter(patients_dest, chunksize, 'age_filter',
                                                  timestamp, 'bool', write_mode)
        age_16_to_90 = persistence.NewNumericWriter(patients_dest, chunksize, '16_to_90_years',
                                                timestamp, 'bool', write_mode)
        calculate_age_from_year_of_birth_fast(
            16, 90,
            patients_src['year_of_birth'], patients_src['year_of_birth_valid'],
            age, age_filter, age_16_to_90,
            2020)
        log(f"completed in {time.time() - t0}")

        print('age_filter count:', np.sum(patients_dest['age_filter']['values'][:]))
        print('16_to_90_years count:', np.sum(patients_dest['16_to_90_years']['values'][:]))

    if clean_weight_height_bmi:
        log("height / weight / bmi; standard range filters")
        t0 = time.time()

        weights_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'weight_kg_clean',
                                                     timestamp, 'float32', write_mode)
        weights_filter = persistence.NewNumericWriter(patients_dest, chunksize, '40_to_200_kg',
                                                      timestamp, 'bool', write_mode)
        heights_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'height_cm_clean',
                                                     timestamp, 'float32', write_mode)
        heights_filter = persistence.NewNumericWriter(patients_dest, chunksize, '110_to_220_cm',
                                                      timestamp, 'bool', write_mode)
        bmis_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'bmi_clean',
                                                  timestamp, 'float32', write_mode)
        bmis_filter = persistence.NewNumericWriter(patients_dest, chunksize, '15_to_55_bmi',
                                                   timestamp, 'bool', write_mode)

        weight_height_bmi_fast_1(40, 200, 110, 220, 15, 55,
                                 None, None, None, None,
                                 patients_src['weight_kg'], patients_src['weight_kg_valid'],
                                 patients_src['height_cm'], patients_src['height_cm_valid'],
                                 patients_src['bmi'], patients_src['bmi_valid'],
                                 weights_clean, weights_filter, None,
                                 heights_clean, heights_filter, None,
                                 bmis_clean, bmis_filter, None)
        log(f"completed in {time.time() - t0}")

    # Assessment processing
    # =====================

    if make_assessment_patient_id_index:
        print("creating 'patient_index' foreign key index for 'patient_id'")
        t0 = time.time()
        patient_ids = persistence.get_reader_from_field(sorted_patients_src['id'])
        assessment_patient_ids =\
            persistence.get_reader_from_field(sorted_assessments_src['patient_id'])
        assessment_patient_id_index =\
            persistence.NewNumericWriter(assessments_dest, chunksize, 'patient_id_index',
                                         timestamp, 'int64')
        persistence.get_index(patient_ids, assessment_patient_ids, assessment_patient_id_index)
        print(f"completed in {time.time() - t0}s")


    if clean_temperatures:
        print("clean temperatures")
        t0 = time.time()
        temps = persistence.NewNumericReader(sorted_assessments_src['temperature'])
        temp_units = persistence.NewFixedStringReader(sorted_assessments_src['temperature_unit'])
        temps_valid = persistence.NewNumericReader(sorted_assessments_src['temperature_valid'])
        dest_temps = temps.getwriter(assessments_dest, 'temperature_c_clean', timestamp,
                                     write_mode)
        dest_temps_valid =\
            temps_valid.getwriter(assessments_dest, 'temperature_35_to_42_inclusive', timestamp,
                                  write_mode)
        dest_temps_modified =\
            temps_valid.getwriter(assessments_dest, 'temperature_modified', timestamp, write_mode)
        validate_temperature_1(35.0, 42.0,
                               temps, temp_units, temps_valid,
                               dest_temps, dest_temps_valid, dest_temps_modified)
        print(f"temperature cleaning done in {time.time() - t0}")


    if check_symptoms:
        print('check inconsistent health_status')
        t0 = time.time()
        check_inconsistent_symptoms_1(sorted_assessments_src, assessments_dest, timestamp)
        print(time.time() - t0)


    # Test processing
    # ===============


    # Daily assessments
    # =================

    if create_daily:
        print("generate daily assessments")
        patient_ids = persistence.get_reader_from_field(sorted_assessments_src['patient_id'])
        raw_patient_ids = patient_ids[:]
        created_at_days =\
            persistence.get_reader_from_field(sorted_assessments_src['created_at_day'])
        raw_created_at_days = created_at_days[:]

        if 'patient_id_index' in assessments_src.keys():
            patient_id_index = assessments_src['patient_id_index']
        else:
            patient_id_index = assessments_dest['patient_id_index']
        patient_id_indices =\
            persistence.get_reader_from_field(patient_id_index)
        raw_patient_id_indices = patient_id_indices[:]


        print("Calculating patient id index spans")
        t0 = time.time()
        patient_id_index_spans =\
            persistence.get_spans(fields=(raw_patient_id_indices, raw_created_at_days))
        print(f"Calculated {len(patient_id_index_spans)-1} spans in {time.time() - t0}s")


        print("Applying spans to 'health_status'")
        t0 = time.time()
        default_behavour_overrides = {
            'id': persistence.apply_spans_last,
            'patient_id': persistence.apply_spans_first,
            'patient_index': persistence.apply_spans_first,
            'created_at': persistence.apply_spans_last,
            'created_at_day': persistence.apply_spans_first,
            'updated_at': persistence.apply_spans_last,
            'updated_at_day': persistence.apply_spans_first,
            'version': persistence.apply_spans_max,
            'country_code': persistence.apply_spans_first,
            'date_test_occurred': None,
            'date_test_occurred_guess': None,
            'date_test_occurred_day': None,
            'date_test_occurred_set': None,
        }
        for k in sorted_assessments_src.keys():
            t1 = time.time()
            reader = persistence.get_reader_from_field(sorted_assessments_src[k])
            if k in default_behavour_overrides:
                apply_span_fn = default_behavour_overrides[k]
                if apply_span_fn is not None:
                    apply_span_fn(patient_id_index_spans, reader,
                                  reader.getwriter(daily_assessments_dest, k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                else:
                    print(f"  Skipping field {k}")
            else:
                if isinstance(reader, persistence.NewCategoricalReader):
                    persistence.apply_spans_max(patient_id_index_spans, reader,
                                                reader.getwriter(daily_assessments_dest,
                                                                 k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                elif isinstance(reader, persistence.NewIndexedStringReader):
                    persistence.apply_spans_concat(patient_id_index_spans, reader,
                                                   reader.getwriter(daily_assessments_dest,
                                                                    k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                elif isinstance(reader, persistence.NewNumericReader):
                    persistence.apply_spans_max(patient_id_index_spans, reader,
                                                reader.getwriter(daily_assessments_dest,
                                                        k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                else:
                    print(f"  No function for {k}")

        print(f"apply_spans completed in {time.time() - t0}s")


    # TODO - patient measure: assessments per patient

    if make_patient_level_assessment_metrics:
        if 'patient_id_index' in assessments_dest:
            src = assessments_dest['patient_id_index']
        else:
            src = assessments_src['patient_id_index']
        assessment_patient_id_index = persistence.NewNumericReader(src)
        spans = persistence.get_spans(field=assessment_patient_id_index)

        assessment_counts = np.zeros(len(spans)-1, dtype=np.uint32)
        persistence.apply_spans_count(spans, assessment_counts)
        patient_ids_for_counts = assessment_patient_id_index[:][spans[:-1]]

        #TODO: needs a persistence function to perform mapping of counts to another space

        assessment_count_per_patient =\
            persistence.NewNumericWriter(patients_dest, chunksize, 'assessment_count',
                                         timestamp, 'uint32')
        ids = persistence.get_reader_from_field(patients_src['id'])
        patient_space_counts = assessment_count_per_patient.chunk_factory(len(ids))
        patient_space_counts[patient_ids_for_counts] = assessment_counts
        assessment_count_per_patient.write(patient_space_counts)



    # TODO - patient measure: daily assessments per patient


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='the dataset to load')
    parser.add_argument('-d', '--destination', help='the dataset to write results to')
    parser.add_argument('--sort', default=False, action='store_true')
    parser.add_argument('--process', default=False, action='store_true')
    parser.add_argument('--all', default=False, action='store_true')
    args = parser.parse_args()

    data_schema = data_schemas.DataSchema(1)
    parsing_schema = parsing_schemas.ParsingSchema(1)
    timestamp = str(datetime.now(timezone.utc))

    if args.sort + args.process + args.all > 1:
        raise ValueError("At most one of '--sort', '--daily', and '--all' may be set")
    elif args.sort + args.process + args.all == 0:
        flags = 'all'
    else:
        if args.sort is True:
            flags = 'sort'
        elif args.process is True:
            flags = 'process'
        elif args.all is True:
            flags = 'all'

    with h5py.File(args.source, 'r') as ds:
        with h5py.File(args.destination, 'w') as ts:
            postprocess(ds, ts, data_schema, parsing_schema, timestamp, flags=flags)
