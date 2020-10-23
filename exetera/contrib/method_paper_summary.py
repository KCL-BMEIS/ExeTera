from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core.persistence import DataStore

def method_paper_summary_pipeline(ds, src_data, dest_data, first_timestamp, last_timestamp):
    s_ptnts = src_data['patients']
    s_asmts = src_data['assessments']
    filters = ds.get_or_create_group(dest_data, 'filters')
    print(s_ptnts.keys())
    print(src_data['tests'].keys())

    conditions = ('has_kidney_disease', 'has_lung_disease', 'has_heart_disease', 'has_diabetes',
                  'has_hayfever', 'has_cancer')

    symptoms = ('persistent_cough', 'fatigue', 'delirium', 'shortness_of_breath', 'fever',
                'diarrhoea', 'abdominal_pain', 'chest_pain', 'hoarse_voice', 'skipped_meals',
                'loss_of_smell')
    symptom_thresholds = {s: 2 for s in symptoms}
    symptom_thresholds['fatigue'] = 3
    symptom_thresholds['shortness_of_breath'] = 3

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


    # generating patient filter
    # -------------------------
    if 'patient_filter' not in filters.keys():
        with utils.Timer("generating patient filter", new_line=True):
            p_filter = ds.get_reader(s_ptnts['year_of_birth_valid'])[:]

            # valid age ranges
            r_ = ds.get_reader(s_ptnts['age'])[:]
            f_ = (r_ >= 18) & (r_ <= 100)
            p_filter = p_filter & f_

            # gender filter
            r_ = ds.get_reader(s_ptnts['gender'])[:]
            f_ = (r_ == 1) | (r_ == 2)
            p_filter = p_filter & f_

            # country code
            r_ = ds.get_reader(s_ptnts['country_code'])[:]
            f_ = r_ == b'GB'
            p_filter = p_filter & f_
            print("UK:", p_filter.sum(), len(p_filter))

            # # England only
            # r_ = ds.get_reader(s_ptnts['lsoa11cd'])[:]
            # f_ = np.zeros(len(r_), dtype=np.bool)
            # for i in range(len(r_)):
            #     lsoa = r_[i]
            #     if len(lsoa) > 0 and lsoa[0] == 69: # E
            #         f_[i] = True
            # p_filter = p_filter & f_
            # print("Eng:", p_filter.sum(), len(p_filter))

            # no assessments
            r_ = ds.get_reader(s_ptnts['assessment_count'])[:]
            f_ = r_ > 0
            p_filter = p_filter & f_
            print("No asmts:", p_filter.sum(), len(p_filter))

            print("  {}, {}".format(np.count_nonzero(p_filter),
                                    np.count_nonzero(p_filter == False)))
            ds.get_numeric_writer(filters, 'patient_filter', 'bool').write(p_filter)


    # generating assessment filter
    # ----------------------------
    if 'assessment_filter' not in filters.keys():
        with utils.Timer("generating assessment filter", new_line=True):
            a_filter = np.ones(len(ds.get_reader(s_asmts['id'])), dtype=np.bool)

            # created_at in range
            r_ = ds.get_reader(s_asmts['created_at'])[:]
            f_ = (r_ >= first_timestamp) & (r_ < last_timestamp)
            a_filter = a_filter & f_

            # country code
            r_ = ds.get_reader(s_asmts['country_code'])[:]
            f_ = r_ == b'GB'
            a_filter = a_filter & f_

            with utils.Timer(f"filtering out orphaned assessments"):
                p_ids_ = ds.get_reader(s_ptnts['id'])[:]
                p_ids_ = ds.apply_filter(ds.get_reader(filters['patient_filter'])[:], p_ids_)
                a_pids_ = ds.get_reader(s_asmts['patient_id'])[:]
                f_ = persistence.foreign_key_is_in_primary_key(p_ids_, a_pids_)
            a_filter = a_filter & f_

            print("  {}, {}".format(np.count_nonzero(a_filter),
                                    np.count_nonzero(a_filter == False)))
            ds.get_numeric_writer(filters, 'assessment_filter', 'bool').write(a_filter)

    # filtering patients
    # ------------------
    if 'filtered_patients' not in dest_data.keys():
        flt_ptnts = dest_data.create_group('filtered_patients')
        with utils.Timer("filtering/flattening patient fields", new_line=True):
            p_filter = ds.get_reader(filters['patient_filter'])[:]

            r = ds.get_reader(s_ptnts['age'])
            r.get_writer(flt_ptnts, 'age').write(ds.apply_filter(p_filter, r[:]))

            for k in conditions:
                r = ds.get_reader(s_ptnts[k])
                ds.get_numeric_writer(flt_ptnts, k, 'bool').write(
                    ds.apply_filter(p_filter, r[:]) == 2)

            smoker1 = ds.get_reader(s_ptnts['is_smoker'])
            smoker2 = ds.get_reader(s_ptnts['smoker_status'])
            smoker = (smoker1[:] == 2) | (smoker2[:] == 3)
            ds.get_numeric_writer(flt_ptnts, 'smoker', 'bool').write(smoker)

            gender_ = ds.get_reader(s_ptnts['gender'])
            ds.get_numeric_writer(flt_ptnts, 'gender', 'uint8').write(
                ds.apply_filter(p_filter, gender_) - 1)
    else:
        flt_ptnts = dest_data['filtered_patients']

    # filtering assessments
    # ---------------------
    if 'filtered_assessments' not in dest_data.keys():
        flt_asmts = dest_data.create_group('filtered_assessments')
        with utils.Timer("filtering/flattening symptoms",
                         new_line=True):
            a_filter = ds.get_reader(filters['assessment_filter'])[:]
            for s in symptoms:
                r_ = ds.get_reader(s_asmts[s])[:]
                ds.get_numeric_writer(flt_asmts, s, 'bool').write(
                    ds.apply_filter(a_filter, r_) >= symptom_thresholds[s])
            a_pids = ds.get_reader(s_asmts['patient_id'])
            a_pids.get_writer(flt_asmts, 'patient_id').write(ds.apply_filter(a_filter, a_pids[:]))
    else:
        flt_asmts = dest_data['filtered_assessments']

    # predicting covid
    # ----------------
    if 'prediction' not in dest_data['filtered_assessments']:
        with utils.Timer("generating covid prediction", new_line=True):
            cumulative = np.zeros(len(ds.get_reader(flt_asmts['persistent_cough'])), dtype='float64')
            for s in symptoms:
                reader = ds.get_reader(flt_asmts[s])
                cumulative += reader[:] * weights[s]
            cumulative += intercept
            print("positive predictions", np.count_nonzero(cumulative > 0.0), len(cumulative))

            a_pids_ = ds.get_reader(flt_asmts['patient_id'])[:]
            spans = ds.get_spans(a_pids_)
            max_prediction_inds = ds.apply_spans_index_of_max(spans, cumulative)
            max_predictions = cumulative[max_prediction_inds]

            ds.get_numeric_writer(flt_asmts, 'prediction', 'float32').write(max_predictions)
            pos_filter = max_predictions > 0.0
            print("pos_filter: ", np.count_nonzero(pos_filter), len(pos_filter))

    # generating table results
    print('total_assessments:', np.count_nonzero(ds.get_reader(filters['assessment_filter'])[:]))
    subjects = np.count_nonzero(ds.get_reader(filters['patient_filter'])[:])
    genders = ds.get_reader(flt_ptnts['gender'])[:]
    predicted_c19 = np.count_nonzero(ds.get_reader(flt_asmts['prediction'])[:] > 0.0)
    age_mean = np.mean(ds.get_reader(flt_ptnts['age'])[:])
    age_std = np.std(ds.get_reader(flt_ptnts['age'])[:])
    print('subjects:', subjects)
    male = np.count_nonzero(genders)
    female = np.count_nonzero(genders == False)
    print('gender: {}:{}, {:.2%}:{:.2%}'.format(male, female,
                                              male / len(genders), female / len(genders)))
    # print('predicted covid-19:', predicted_c19)
    print('{}:'.format('predicted covid-19'),
          '{} {:.2%}'.format(predicted_c19,
                             predicted_c19 / len(ds.get_reader(flt_asmts['prediction']))))
    print('age {:.2f} ({:.2f})'.format(age_mean, age_std))
    for k in conditions + ('smoker',):
        kr_ = ds.get_reader(flt_ptnts[k])[:]
        pos = np.count_nonzero(kr_)
        print('{}:'.format(k), '{} {:.2%}'.format(pos, pos / len(kr_)))
        # print(np.count_nonzero(kr_), len(kr_))


if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20201008_full.hdf5'
    dest_file = '/home/ben/covid/ds_20201008_summary.hdf5'
    with h5py.File(src_file, 'r+') as src_data:
        with h5py.File(dest_file, 'w') as dest_data:
            method_paper_summary_pipeline(datastore, src_data, dest_data,
                                         datetime.strptime("2020-03-01", '%Y-%m-%d').timestamp(),
                                         datetime.strptime("2020-10-08", '%Y-%m-%d').timestamp())
# if __name__ == '__main__':
#     datastore = DataStore()
#     src_file = '/home/ben/covid/ds_20200929_full.hdf5'
#     dest_file = '/home/ben/covid/ds_20200929_supplement_table.hdf5'
#     with h5py.File(src_file, 'r+') as src_data:
#         with h5py.File(dest_file, 'w') as dest_data:
#             method_paper_summary_pipeline(datastore, src_data, dest_data,
#                                          datetime.strptime("2020-04-01", '%Y-%m-%d').timestamp(),
#                                          datetime.strptime("2020-09-29", '%Y-%m-%d').timestamp())
