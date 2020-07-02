from datetime import datetime
import time

import numpy as np

from processing.age_from_year_of_birth import calculate_age_from_year_of_birth_fast
from processing.weight_height_bmi import weight_height_bmi_fast_1
from processing.inconsistent_symptoms import check_inconsistent_symptoms_1
from processing.temperature import validate_temperature_1
from core import persistence as per


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
    patients_dest = per.get_or_create_group(destination, 'patients')
    assessments_src = dataset['assessments']
    assessments_dest = per.get_or_create_group(destination, 'assessments')
    daily_assessments_dest = per.get_or_create_group(destination, 'daily_assessments')
    tests_src = dataset['tests']
    tests_dest = per.get_or_create_group(destination, 'tests')

    sort_enabled = lambda x: x in ('sort', 'all')
    process_enabled = lambda x: x in ('process', 'all')

    sort_patients = sort_enabled(flags) and True
    sort_assessments = sort_enabled(flags) and True
    sort_tests = sort_enabled(flags) and True

    make_assessment_patient_id_fkey = process_enabled(flags) and True
    year_from_age = process_enabled(flags) and True
    clean_weight_height_bmi = process_enabled(flags) and True
    clean_temperatures = process_enabled(flags) and True
    check_symptoms = process_enabled(flags) and True
    create_daily = process_enabled(flags) and True
    make_patient_level_assessment_metrics = process_enabled(flags) and True
    make_patient_level_daily_assessment_metrics = process_enabled(flags) and create_daily and True

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
        per.sort_on(
            patients_src, patients_dest, sort_keys, timestamp=timestamp, write_mode=write_mode)

    if sort_assessments:
        sort_keys = ('patient_id', 'created_at')
        per.sort_on(
            assessments_src, assessments_dest, sort_keys, timestamp=timestamp)

        # print("creating 'patient_index' foreign key index for 'patient_id'")
        # t0 = time.time()
        # patient_ids = per.get_reader_from_field(patients_dest['id'])
        # assessment_patient_ids =\
        #     per.get_reader_from_field(assessments_dest['patient_id'])
        # assessment_patient_id_fkey =\
        #     assessment_patient_ids.getwriter(assessments_dest, 'patient_index', timestamp)
        # per.get_index(patient_ids, assessment_patient_ids, assessment_patient_id_fkey)
        # print(f"completed in {time.time() - t0}s")

        print("checking sort order")
        t0 = time.time()
        raw_patient_ids = per.FixedStringReader(assessments_dest['patient_id'])[:]
        raw_created_ats = per.TimestampReader(assessments_dest['created_at'])[:]
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
        print(f"sort order checked({duplicates} duplicate row keys found) in {time.time() - t0}")

    if sort_tests:
        sort_keys = ('patient_id', 'created_at')
        per.sort_on(
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
        age = per.NumericWriter(patients_dest, chunksize, 'age', timestamp, 'uint32',
                                        write_mode)
        age_filter = per.NumericWriter(patients_dest, chunksize, 'age_filter',
                                               timestamp, 'bool', write_mode)
        age_16_to_90 = per.NumericWriter(patients_dest, chunksize, '16_to_90_years',
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

        weights_clean = per.NumericWriter(patients_dest, chunksize, 'weight_kg_clean',
                                                  timestamp, 'float32', write_mode)
        weights_filter = per.NumericWriter(patients_dest, chunksize, '40_to_200_kg',
                                                   timestamp, 'bool', write_mode)
        heights_clean = per.NumericWriter(patients_dest, chunksize, 'height_cm_clean',
                                                  timestamp, 'float32', write_mode)
        heights_filter = per.NumericWriter(patients_dest, chunksize, '110_to_220_cm',
                                                   timestamp, 'bool', write_mode)
        bmis_clean = per.NumericWriter(patients_dest, chunksize, 'bmi_clean',
                                               timestamp, 'float32', write_mode)
        bmis_filter = per.NumericWriter(patients_dest, chunksize, '15_to_55_bmi',
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

    if make_assessment_patient_id_fkey:
        print("creating 'assessment_patient_id_fkey' foreign key index for 'patient_id'")
        t0 = time.time()
        patient_ids = per.get_reader(sorted_patients_src['id'])
        assessment_patient_ids =\
            per.get_reader(sorted_assessments_src['patient_id'])
        assessment_patient_id_fkey =\
            per.NumericWriter(assessments_dest, chunksize, 'assessment_patient_id_fkey',
                                      timestamp, 'int64')
        per.get_index(patient_ids, assessment_patient_ids, assessment_patient_id_fkey)
        print(f"completed in {time.time() - t0}s")


    if clean_temperatures:
        print("clean temperatures")
        t0 = time.time()
        temps = per.NumericReader(sorted_assessments_src['temperature'])
        temp_units = per.FixedStringReader(sorted_assessments_src['temperature_unit'])
        temps_valid = per.NumericReader(sorted_assessments_src['temperature_valid'])
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
        patient_ids = per.get_reader(sorted_assessments_src['patient_id'])
        raw_patient_ids = patient_ids[:]
        created_at_days =\
            per.get_reader(sorted_assessments_src['created_at_day'])
        raw_created_at_days = created_at_days[:]

        if 'assessment_patient_id_fkey' in assessments_src.keys():
            patient_id_index = assessments_src['assessment_patient_id_fkey']
        else:
            patient_id_index = assessments_dest['assessment_patient_id_fkey']
        patient_id_indices =\
            per.get_reader(patient_id_index)
        raw_patient_id_indices = patient_id_indices[:]


        print("Calculating patient id index spans")
        t0 = time.time()
        patient_id_index_spans =\
            per.get_spans(fields=(raw_patient_id_indices, raw_created_at_days))
        print(f"Calculated {len(patient_id_index_spans)-1} spans in {time.time() - t0}s")


        print("Applying spans to 'health_status'")
        t0 = time.time()
        default_behavour_overrides = {
            'id': per.apply_spans_last,
            'patient_id': per.apply_spans_first,
            'patient_index': per.apply_spans_first,
            'created_at': per.apply_spans_last,
            'created_at_day': per.apply_spans_first,
            'updated_at': per.apply_spans_last,
            'updated_at_day': per.apply_spans_first,
            'version': per.apply_spans_max,
            'country_code': per.apply_spans_first,
            'date_test_occurred': None,
            'date_test_occurred_guess': None,
            'date_test_occurred_day': None,
            'date_test_occurred_set': None,
        }
        for k in sorted_assessments_src.keys():
            t1 = time.time()
            reader = per.get_reader(sorted_assessments_src[k])
            if k in default_behavour_overrides:
                apply_span_fn = default_behavour_overrides[k]
                if apply_span_fn is not None:
                    apply_span_fn(patient_id_index_spans, reader,
                                  reader.getwriter(daily_assessments_dest, k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                else:
                    print(f"  Skipping field {k}")
            else:
                if isinstance(reader, per.CategoricalReader):
                    per.apply_spans_max(patient_id_index_spans, reader,
                                        reader.getwriter(daily_assessments_dest, k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                elif isinstance(reader, per.IndexedStringReader):
                    per.apply_spans_concat(patient_id_index_spans, reader,
                                           reader.getwriter(daily_assessments_dest, k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                elif isinstance(reader, per.NumericReader):
                    per.apply_spans_max(patient_id_index_spans, reader,
                                        reader.getwriter(daily_assessments_dest, k, timestamp))
                    print(f"  Field {k} aggregated in {time.time() - t1}s")
                else:
                    print(f"  No function for {k}")

        print(f"apply_spans completed in {time.time() - t0}s")


    # TODO - patient measure: assessments per patient

    if make_patient_level_assessment_metrics:
        if 'assessment_patient_id_fkey' in assessments_dest:
            src = assessments_dest['assessment_patient_id_fkey']
        else:
            src = assessments_src['assessment_patient_id_fkey']
        assessment_patient_id_fkey = per.get_reader(src)
        # generate spans from the assessment-space patient_id foreign key
        spans = per.get_spans(field=assessment_patient_id_fkey)

        ids = per.get_reader(patients_src['id'])

        #TODO: needs a persistence function to perform mapping of values to another space

        # print('predicate_and_join')
        # acpp2 = per.NumericWriter(patients_dest, chunksize, 'assessment_count_2', timestamp, 'uint32')
        # per.predicate_and_join(per.apply_spans_count, ids, assessment_patient_id_fkey, None, acpp2, spans)

        print('calculate assessment counts per patient')
        t0 = time.time()
        writer = per.NumericWriter(patients_dest, chunksize, 'assessment_count', timestamp, 'uint32')
        aggregated_counts = per.aggregate_count(fkey_index_spans=spans)
        per.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated assessment counts per patient in {time.time() - t0}")

        print('calculate first assessment days per patient')
        t0 = time.time()
        reader = per.get_reader(sorted_assessments_src['created_at_day'])
        writer = per.FixedStringWriter(patients_dest, chunksize, 'first_assessment_day', timestamp, 10)
        aggregated_counts = per.aggregate_first(fkey_index_spans=spans, reader=reader)
        per.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated first assessment days per patient in {time.time() - t0}")

        print('calculate last assessment days per patient')
        t0 = time.time()
        pids = per.get_reader(sorted_assessments_src['patient_id'])
        reader = per.get_reader(sorted_assessments_src['created_at_day'])
        writer = per.FixedStringWriter(patients_dest, chunksize, 'last_assessment_day', timestamp, 10)
        aggregated_counts = per.aggregate_last(fkey_index_spans=spans, reader=reader)
        per.join(ids, assessment_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated last assessment days per patient in {time.time() - t0}")

    # TODO - patient measure: daily assessments per patient

    if make_patient_level_daily_assessment_metrics:
        print("creating 'daily_assessment_patient_id_fkey' foreign key index for 'patient_id'")
        t0 = time.time()
        patient_ids = per.get_reader(sorted_patients_src['id'])
        daily_assessment_patient_ids =\
            per.get_reader(daily_assessments_dest['patient_id'])
        daily_assessment_patient_id_fkey =\
            per.NumericWriter(daily_assessments_dest, chunksize, 'daily_assessment_patient_id_fkey',
                                      timestamp, 'int64')
        per.get_index(patient_ids, daily_assessment_patient_ids, daily_assessment_patient_id_fkey)
        print(f"completed in {time.time() - t0}s")

        spans = per.get_spans(
            field=per.get_reader(daily_assessments_dest['daily_assessment_patient_id_fkey']))

        print('calculate daily assessment counts per patient')
        t0 = time.time()
        writer = per.NumericWriter(patients_dest, chunksize, 'daily_assessment_count', timestamp, 'uint32')
        aggregated_counts = per.aggregate_count(fkey_index_spans=spans)
        daily_assessment_patient_id_fkey =\
            per.get_reader(daily_assessments_dest['daily_assessment_patient_id_fkey'])
        per.join(ids, daily_assessment_patient_id_fkey, aggregated_counts, writer, spans)
        print(f"calculated daily assessment counts per patient in {time.time() - t0}")


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-s', '--source', help='the dataset to load')
#     parser.add_argument('-d', '--destination', help='the dataset to write results to')
#     parser.add_argument('--sort', default=False, action='store_true')
#     parser.add_argument('--process', default=False, action='store_true')
#     parser.add_argument('--all', default=False, action='store_true')
#     args = parser.parse_args()
#
#     data_schema = data_schemas.DataSchema(1)
#     parsing_schema = parsing_schemas.ParsingSchema(1)
#     timestamp = str(datetime.now(timezone.utc))
#
#     if args.sort + args.process + args.all > 1:
#         raise ValueError("At most one of '--sort', '--daily', and '--all' may be set")
#     elif args.sort + args.process + args.all == 0:
#         flags = 'all'
#     else:
#         if args.sort is True:
#             flags = 'sort'
#         elif args.process is True:
#             flags = 'process'
#         elif args.all is True:
#             flags = 'all'
#
#     with h5py.File(args.source, 'r') as ds:
#         with h5py.File(args.destination, 'w') as ts:
#             postprocess(ds, ts, data_schema, parsing_schema, timestamp, flags=flags)
