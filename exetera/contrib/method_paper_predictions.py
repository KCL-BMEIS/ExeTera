import math
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core.persistence import DataStore
from exetera.processing.nat_medicine_model import nature_medicine_model_1


def method_paper_prediction_pipeline(ds, src_data, dest_data, first_timestamp, last_timestamp):
    s_ptnts = src_data['patients']
    s_asmts = src_data['assessments']
    s_tests = src_data['tests']

    first_dt = datetime.fromtimestamp(first_timestamp)
    last_dt = datetime.fromtimestamp(last_timestamp)
    print(s_tests.keys())

    # Filter patients to be only from England
    # =======================================

    eng_pats = set()
    p_ids_ = ds.get_reader(s_ptnts['id'])[:]
    p_lsoas_ = ds.get_reader(s_ptnts['lsoa11cd'])[:]
    for i in range(len(p_ids_)):
        lsoa = p_lsoas_[i]
        if len(lsoa) > 0 and lsoa[0] == 69: # E
            eng_pats.add(p_ids_[i])
    print("eng pats:", len(eng_pats))

    if "flat_asmts" not in dest_data.keys():
        flat_tests = dest_data.create_group('flat_tests')

        # Filter tests
        # ============

        t_cats = ds.get_reader(s_tests['created_at'])
        raw_t_cats = t_cats[:]
        t_dts = ds.get_reader(s_tests['date_taken_specific'])
        raw_t_dts = t_dts[:]
        t_dsbs = ds.get_reader(s_tests['date_taken_between_start'])
        raw_t_dsbs = t_dsbs[:]
        t_dsbe = ds.get_reader(s_tests['date_taken_between_end'])
        raw_t_dsbe = t_dsbe[:]

        # remove non GB tests
        cur_filter = (ds.get_reader(s_tests['country_code'])[:] == b'GB')
        test_filter = cur_filter[:]
        print("standard test filter GB:", np.count_nonzero(test_filter), len(test_filter))

        # remove non england tests
        t_pids_ = ds.get_reader(s_tests['patient_id'])[:]
        cur_filter = np.zeros(len(t_pids_), dtype=np.bool)
        for i in range(len(t_pids_)):
            cur_filter[i] = t_pids_[i] in eng_pats
        test_filter = test_filter & cur_filter
        print("standard test filter Eng:", np.count_nonzero(test_filter), len(test_filter))

        # remove tests where no dates are set
        cur_filter = np.logical_not((raw_t_dts == 0) & (raw_t_dsbs == 0) & (raw_t_dsbe == 0))
        test_filter = test_filter & cur_filter
        print("standard test filter 1:", np.count_nonzero(test_filter), len(test_filter))

        # remove tests where all three dates are set
        cur_filter = np.logical_not((raw_t_dts != 0) & (raw_t_dsbs != 0) & (raw_t_dsbe != 0))
        test_filter = test_filter & cur_filter
        print("standard test filter 2:", np.count_nonzero(test_filter), len(test_filter))

        # remove tests where only one of the date range tests is set
        cur_filter = np.logical_not((raw_t_dsbs != 0) & (raw_t_dsbe == 0) |
                                    (raw_t_dsbs == 0) & (raw_t_dsbe != 0))
        test_filter = test_filter & cur_filter
        print("standard test filter 3:", np.count_nonzero(test_filter), len(test_filter))

        # remove tests where specific date is set but out of range
        cur_filter =\
            (raw_t_dts == 0) | ((raw_t_dts >= first_timestamp) & (raw_t_dts <= last_timestamp))
        test_filter = test_filter & cur_filter
        print("standard test filter 4:", np.count_nonzero(test_filter), len(test_filter))

        # remove tests where beginning date is set but out of range
        cur_filter =\
            (raw_t_dsbs == 0) | ((raw_t_dsbs >= first_timestamp) & (raw_t_dsbs <= last_timestamp))
        test_filter = test_filter & cur_filter
        print("standard test filter 5:", np.count_nonzero(test_filter), len(test_filter))

        # remove tests where ending date is set but out of range
        cur_filter = \
            (raw_t_dsbe == 0) | ((raw_t_dsbe >= first_timestamp) & (raw_t_dsbe <= last_timestamp))
        test_filter = test_filter & cur_filter
        print("standard test filter 6:", np.count_nonzero(test_filter), len(test_filter))

        test_timestamps = np.where(raw_t_dts != 0,
                                   raw_t_dts,
                                   raw_t_dsbs + (raw_t_dsbe - raw_t_dsbs) / 2)

        # remove tests where the test date is after the created at date
        cur_filter = test_timestamps <= raw_t_cats
        test_filter = test_filter & cur_filter
        print("standard test filter 7:", np.count_nonzero(test_filter), len(test_filter))

        t_rsts = ds.get_reader(s_tests['result'])
        t_rsts.get_writer(flat_tests, 'result').write(ds.apply_filter(test_filter, t_rsts))
        t_pids = ds.get_reader(s_tests['patient_id'])
        t_pids.get_writer(flat_tests, 'patient_id').write(ds.apply_filter(test_filter, t_pids))
        ds.get_timestamp_writer(flat_tests, 'eff_test_date').write(
            ds.apply_filter(test_filter, test_timestamps))

        # test_min_ts = datetime.fromtimestamp(test_timestamps[test_filter].min())
        # test_max_ts = datetime.fromtimestamp(test_timestamps[test_filter].max())
        # print(test_min_ts, test_max_ts)
    else:
        flat_tests = dest_data["flat_tests"]


    symptoms = ('persistent_cough', 'fatigue', 'delirium', 'shortness_of_breath', 'fever',
                'diarrhoea', 'abdominal_pain', 'chest_pain', 'hoarse_voice', 'skipped_meals',
                'loss_of_smell')

    if "flat_asmts" not in dest_data.keys():
        flat_asmts = dest_data.create_group('flat_asmts')

        # Filter assessments
        # ------------------

        symptom_thresholds = {s: 2 for s in symptoms}
        symptom_thresholds['fatigue'] = 3
        symptom_thresholds['shortness_of_breath'] = 3

        with utils.Timer("filter all out of date range assessments and non-uk assessments", new_line=True):
            a_cats = ds.get_reader(s_asmts['created_at'])[:]
            # in_date_range = (a_cats >= first_timestamp) & (a_cats < last_timestamp)
            in_date_range = a_cats >= first_timestamp
            in_date_range = in_date_range & (ds.get_reader(s_asmts['country_code'])[:] == b'GB')

            a_pids = ds.get_reader(s_asmts['patient_id'])[:]
            in_eng = np.zeros(len(a_pids), dtype=np.bool)
            for i in range(len(a_pids)):
                if a_pids[i] in eng_pats:
                    in_eng[i] = True
            print("in_eng:", in_eng.sum(), len(in_eng))
            in_date_range = in_date_range & in_eng

        with utils.Timer("get indices of final assessments of each day for each person"):
            f_a_pids = ds.apply_filter(in_date_range, a_pids)
            f_a_catds = ds.apply_filter(in_date_range, ds.get_reader(s_asmts['created_at_day'])[:])
            spans = ds.get_spans(f_a_pids)

            last_daily_asmt_filter = np.zeros(len(f_a_pids), dtype=np.bool)
            for s in range(len(spans)-1):
                sb = spans[s]
                se = spans[s+1]
                subspans = ds.get_spans(f_a_catds[sb:se])
                if s < 3:
                    print(subspans)
                for s2 in range(1, len(subspans)):
                    last_daily_asmt_filter[sb + subspans[s2]-1] = True
            print("last_daily_asmt_filter:", last_daily_asmt_filter.sum())
            print(last_daily_asmt_filter[:50])

            # otherspans = ds.get_spans(f_a_catds)
            # last_daily_asmts = np.zeros(len(otherspans)-1, dtype='int64')
            # ds.apply_spans_index_of_last(otherspans, last_daily_asmts)
            # print("last_daily_asmts:", len(last_daily_asmts))

        # pc = ds.get_reader(s_asmts['persistent_cough'])[:]
        # pc1 = ds.apply_indices(last_daily_asmts, ds.apply_filter(in_date_range, pc))
        # pc2 = ds.apply_indices(last_daily_asmts, pc)
        # print(len(pc1), len(pc2))
        # print(np.array_equal(pc1, pc2))


        with utils.Timer("flattening and filtering symptoms"):
            for s in symptoms:
                reader = ds.get_reader(s_asmts[s])
                writer = ds.get_numeric_writer(flat_asmts, s, 'bool')
                filtered = ds.apply_filter(last_daily_asmt_filter, ds.apply_filter(in_date_range, reader[:]))
                writer.write(filtered >= symptom_thresholds[s])

        with utils.Timer("flattening and filtering other fields", new_line=True):
            for f in ('id', 'patient_id', 'created_at', 'created_at_day', 'tested_covid_positive'):
                reader = ds.get_reader(s_asmts[f])
                writer = reader.get_writer(flat_asmts, f)
                ds.apply_filter(in_date_range, reader, writer)
                reader = ds.get_reader(flat_asmts[f])
                writer = reader.get_writer(flat_asmts, f, write_mode='overwrite')
                ds.apply_filter(last_daily_asmt_filter, reader, writer)
                print("  {}".format(f), len(ds.get_reader(flat_asmts[f])))

        # telemetry only
        for s in symptoms:
            print(s, len(ds.get_reader(flat_asmts[s])),
                  np.count_nonzero(ds.get_reader(flat_asmts[s])[:]))
    else:
        flat_asmts = dest_data["flat_asmts"]


    # Filter tests
    # ------------

    # # filter tests within day range first
    # t_cats = ds.get_reader(s_tests['created_at'])
    # raw_t_cats = t_cats[:]
    # t_rsts = ds.get_reader(s_tests['result'])
    # t_pids = ds.get_reader(s_tests['patient_id'])
    # # test_date_filter = (raw_t_cats >= first_timestamp) & (raw_t_cats < last_timestamp)
    # test_date_filter = raw_t_cats >= first_timestamp
    # test_date_filter = test_date_filter & (ds.get_reader(s_tests['country_code'])[:] == b'GB')
    # t_cats.get_writer(flat_tests, 'created_at').write(ds.apply_filter(test_date_filter, raw_t_cats))
    # t_rsts.get_writer(flat_tests, 'result').write(ds.apply_filter(test_date_filter, t_rsts))
    # t_pids.get_writer(flat_tests, 'patient_id').write(ds.apply_filter(test_date_filter, t_pids))
    #
    # raw_t_cats = ds.get_reader(flat_tests['created_at'])[:]
    # min_test_day = datetime.fromtimestamp(np.min(raw_t_cats))
    # max_test_day = datetime.fromtimestamp(np.max(raw_t_cats))
    # print(min_test_day, max_test_day)

    # Calculate prevalence
    # --------------------

    if 'prediction' not in flat_asmts:
        intercept = -1.19015973
        weights = {'persistent_cough': 0.23186655,
                   'fatigue': 0.56532346,
                   'delirium': -0.12935112,
                   'shortness_of_breath': 0.58273967,
                   'fever': 0.16580974,
                   'diarrhoea': 0.10236126,
                   'abdominal_pain': -0.11204163,
                   'chest_pain': -0.12318634,
                   'hoarse_voice': -0.17818597,
                   'skipped_meals': 0.25902482,
                   'loss_of_smell': 1.82895239}

        with utils.Timer("predicting covid by assessment", new_line=True):
            cumulative = np.zeros(len(ds.get_reader(flat_asmts['persistent_cough'])), dtype='float64')
            for s in symptoms:
                reader = ds.get_reader(flat_asmts[s])
                cumulative += reader[:] * weights[s]
            cumulative += intercept
            print("  {}".format(len(cumulative)))
            ds.get_numeric_writer(flat_asmts, 'prediction', 'float32', writemode='overwrite').write(cumulative)
            pos_filter = cumulative > 0.0
            print("pos_filter: ", np.count_nonzero(pos_filter), len(pos_filter))
    else:
        cumulative = ds.get_reader(flat_asmts['prediction'])[:]

    # apply
    # positive test -> imputed positive -> negative test
    spans = ds.get_spans(ds.get_reader(flat_asmts['patient_id'])[:])
    print('spans:', len(spans))

    # generate a numpy array for each day, where each entry in the array is a patient with
    # assessments still in the dataset after the initial filter

    daydict = defaultdict(int)
    with utils.Timer("checking date deltas", new_line=True):
        a_cats = ds.get_reader(flat_asmts['created_at'])[:]
        first_day = datetime.fromtimestamp(first_timestamp)
        for i_r in range(len(a_cats)):
            daydict[(datetime.fromtimestamp(a_cats[i_r]) - first_day).days] += 1
        sdaydict = sorted(daydict.items())
        print(sdaydict)

    # build a combined id index for assessments and tests
    # ---------------------------------------------------
    remaining_a_pids = ds.get_reader(flat_asmts['patient_id'])[:]
    remaining_t_pids = ds.get_reader(flat_tests['patient_id'])[:]
    print("pids from assessments and tests:", len(remaining_a_pids), len(remaining_t_pids),
          len(set(remaining_a_pids).union(set(remaining_t_pids))))
    a_pid_index, t_pid_index = ds.get_shared_index((remaining_a_pids, remaining_t_pids))
    print("merging indices:", len(a_pid_index), len(t_pid_index), max(np.max(a_pid_index), np.max(t_pid_index)))

    max_index = max(a_pid_index[-1], t_pid_index[-1])
    print('max indices:', a_pid_index[-1], t_pid_index[-1])


    # calculate offset days for assessments
    # -------------------------------------

    first_day = datetime.fromtimestamp(first_timestamp)
    a_cats = ds.get_reader(flat_asmts['created_at'])[:]
    a_tcps = ds.get_reader(flat_asmts['tested_covid_positive'])[:]
    a_offset_days = np.zeros(len(a_cats), dtype='int16')

    with utils.Timer("calculate offset days for assessments", new_line=True):
        for i_r, r in enumerate(a_cats):
            a_offset_days[i_r] = (datetime.fromtimestamp(a_cats[i_r]) - first_day).days
        print("assessment_dates:", sorted(utils.build_histogram(a_offset_days)))


    # calculate offset days for tests
    # -------------------------------

    t_etds = ds.get_reader(flat_tests['eff_test_date'])
    raw_t_etds = t_etds[:]
    t_rsts = ds.get_reader(flat_tests['result'])
    t_pids = ds.get_reader(flat_tests['patient_id'])

    t_offset_days = np.zeros(len(raw_t_etds), dtype='int16')
    t_offset_dates = [None] * len(raw_t_etds)
    for i_r, r in enumerate(raw_t_etds):
        t_offset_days[i_r] = (datetime.fromtimestamp(raw_t_etds[i_r]) - first_day).days
        t_offset_dates[i_r] = datetime.fromtimestamp(raw_t_etds[i_r]).date()
    print("test_dates:", sorted(utils.build_histogram(t_offset_days)))
    print("test_dates2:", sorted(utils.build_histogram(t_offset_dates)))


    # create the destination arrays to hold daily data per patient
    # ------------------------------------------------------------
    daycount = max(a_offset_days.max(), t_offset_days.max()) + 1
    i_days = list([None] * daycount)
    t_days = list([None] * daycount)
    print("daycount:", daycount)
    for i in range(daycount):
        i_days[i] = np.zeros(max_index+1, dtype='int16')
        t_days[i] = np.zeros(max_index+1, dtype='int16')


    # incorporate assessment predictions and positive test results
    # note: a_offset_days is in assessment space
    print("len(a_offset_days):", len(a_offset_days))
    print("len(a_pid_index):", len(a_pid_index))
    with utils.Timer("incorporating assessments and assessment-based tests"):
        for i_r, r in enumerate(a_offset_days):
            # i_days[a_offset_days[i_r]][a_pid_index[i_r]] =\
            #     from_tcp if from_tcp != 0 else from_prediction
            from_prediction = 7 if cumulative[i_r] > 0.0 else -7
            i_days[a_offset_days[i_r]][a_pid_index[i_r]] = from_prediction
            from_tcp = 7 if a_tcps[i_r] == 3 else -7 if a_tcps[i_r] == 2 else 0
            t_days[a_offset_days[i_r]][a_pid_index[i_r]] = from_tcp

    # incorporate test results by to appropriate day's entry
    with utils.Timer("incorporating test_results"):
        for i_r, r in enumerate(t_offset_days):
            day = t_days[t_offset_days[i_r]]
            if t_rsts[i_r] == 4:
                day[t_pid_index[i_r]] = 7
            elif t_rsts[i_r] == 3:
                day[t_pid_index[i_r]] = -7
            # day[t_pid_index[i_r]] = 7 if t_rsts[i_r] == 4 else -7 if t_rsts[i_r] == 3 else 0
            # if day[t_pid_index[i_r]] == 0:
            #     day[t_pid_index[i_r]] = 7 if t_rsts[i_r] == 4 else -7 if t_rsts[i_r] == 3 else 0
            # else:
            #     day[t_pid_index[i_r]] = 7 if t_rsts[i_r] == 4 else max(day[t_pid_index[i_r]], -7)

    for i_d, d in enumerate(i_days):
        print(i_d, np.count_nonzero(d))

    with utils.Timer("calculating progression"):
        for da in (i_days, t_days):
            for i_d in range(len(da)-1):
                prior_d = da[i_d]
                next_d = da[i_d + 1]
                next_d[:] = np.where(next_d != 0,
                                     next_d,
                                     np.where(prior_d > 0, prior_d-1, np.minimum(prior_d+1, 0)))
    for d in range(len(i_days)):
        i_d = i_days[d]
        t_d = t_days[d]
        i_present = np.count_nonzero(i_d != 0)
        i_positive = np.count_nonzero(i_d > 0)
        t_present = np.count_nonzero(t_d != 0)
        t_positive = np.count_nonzero(t_d > 0)
        c_d = np.where(t_d == 0, i_d, t_d)
        c_present = np.count_nonzero(c_d != 0)
        c_positive = np.count_nonzero(c_d > 0)

        day = first_day + timedelta(days=d)
        if c_present != 0:
            print(day, i_present, i_positive, t_present, t_positive, c_present, c_positive,
                  c_positive / c_present)
        else:
            print(day, i_present, i_positive, t_present, t_positive, c_present, c_positive,
                  "NA")


if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20201008_full.hdf5'
    dest_file = '/home/ben/covid/ds_20201008_supplements.hdf5'
    with h5py.File(src_file, 'r') as src_data:
        with h5py.File(dest_file, 'r+') as dest_data:
            method_paper_prediction_pipeline(datastore, src_data, dest_data,
                                             datetime.strptime("2020-03-01", '%Y-%m-%d').timestamp(),
                                             datetime.strptime("2020-10-08", '%Y-%m-%d').timestamp())
