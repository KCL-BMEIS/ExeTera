import numpy as np
import pandas as pd

import h5py

from exetera.core.session import Session
from exetera.core.utils import Timer
from exetera.core.persistence import foreign_key_is_in_primary_key

def selectCleanPatients_bmi_age_year(filename_in, filename_out, list_fields):
    s = Session()
    with h5py.File(filename_in, 'r') as source:
        with h5py.File(filename_out, 'w') as output:
            src_pat = source['patients']

            # filtering patients
            filter_bmi_ = s.get(src_pat['bmi_valid']).data[:]
            filter_years_ = s.get(src_pat['16_to_90_years']).data[:]
            blood_type_ = s.get(src_pat['blood_group']).data[:]
            filter_blood_type_ = (blood_type_ > 0) & (blood_type_ < 5)
            filter_has_new_test_ = s.get(src_pat['max_test_result']).data[:] > 2
            filter_has_old_test_ = s.get(src_pat['max_assessment_test_result']).data[:] > 1

            totalFilter_ = filter_bmi_ & filter_years_ & filter_blood_type_ & (filter_has_new_test_ | filter_has_old_test_)
            print(totalFilter_.sum(), len(totalFilter_))

            filtered_pat = output.create_group('patients')
            for k in list_fields:
                src = s.get(src_pat[k])
                dest = src.create_like(filtered_pat, k)
                s.apply_filter(totalFilter_, src, dest)

            positive_test = s.create_numeric(filtered_pat, 'positive_test', 'bool')
            old_tests = s.apply_filter(totalFilter_, s.get(src_pat['max_assessment_test_result']).data[:])
            print(np.unique(old_tests, return_counts=True))
            new_tests = s.apply_filter(totalFilter_, s.get(src_pat['max_test_result']).data[:])
            print(np.unique(new_tests, return_counts=True))
            positive_test.data.write((old_tests == 3) | (new_tests == 4))
            print(positive_test.data[:].sum(), len(positive_test.data))



def selectCleanPatients_bmi_age_year_old(filename_in, filename_out, list_fields):
    s = Session()
    test_interesting = ['id', 'result', 'patient_id']
    with h5py.File(filename_in, 'r') as source:
        with h5py.File(filename_out, 'w') as output:
            src_pat = source['patients']
            src_tests = source['tests']
            src_asmt = source['assessments']

            # filtering patients
            filter_bmi_ = s.get(src_pat['bmi_valid']).data[:]
            filter_years_ = s.get(src_pat['16_to_90_years']).data[:]
            blood_type_ = s.get(src_pat['blood_group']).data[:]
            filter_blood_type_ = (blood_type_ > 0) & (blood_type_ < 5)
            filter_has_test_ = s.get(src_pat['max_test_result']).data[:] > 2
            totalFilter_ = filter_bmi_ & filter_years_ & filter_blood_type_ & filter_has_test_
            print(totalFilter_.sum(), len(totalFilter_))

            filtered_pat = output.create_group('patients')
            for k in list_fields:
                src = s.get(src_pat[k])
                dest = src.create_like(filtered_pat, k)
                s.apply_filter(totalFilter_, src, dest)

            # NEW TESTS
            # Filtering definite results and the maxmium value
            results_raw_ = s.get(src_tests['result']).data[:]
            patient_id_ = s.get(src_tests['patient_id']).data[:]
            results_pos_neg_ = np.where(np.logical_or(results_raw_ == 4, results_raw_ == 3), True, False)
            spans_tests = s.get_spans(patient_id_)
            filter_test_ = np.zeros(len(results_raw_), dtype=np.bool)
            for span in range(spans_tests.size - 1):
                ind_maxResults = np.argmax(results_raw_[spans_tests[span]:spans_tests[span + 1]])
                filter_test_[ind_maxResults + spans_tests[span]] = True

            in_remaining_patients = foreign_key_is_in_primary_key(s.get(filtered_pat['id']).data[:], patient_id_)
            print(in_remaining_patients.sum(), len(in_remaining_patients))

            filtered_tests = output.create_group('tests_results')
            totalFilter_tests = np.logical_and(results_pos_neg_, filter_test_)
            for k in test_interesting:
                src = s.get(src_tests[k])
                dest = src.create_like(filtered_tests, k)
                s.apply_filter(totalFilter_tests, src, dest)

            #
            #
            with Timer('merge_cvs', new_line=True):
                id = s.get(filtered_pat['id'])
                patient_id = s.get(filtered_tests['patient_id'])
                test_df = pd.DataFrame(
                    {'patient_id': patient_id.data[:], 'result': s.get(filtered_tests['result']).data[:]})
                patient_df = pd.DataFrame({'id': id.data[:], 'blood_group': s.get(filtered_pat['blood_group']).data[:],
                                           'age': s.get(filtered_pat['age']).data[:],
                                           'gender': s.get(filtered_pat['gender']).data[:],
                                           'bmi_clean': s.get(filtered_pat['bmi_clean']).data[:]})
                mdf = pd.merge(left=test_df, right=patient_df, how='inner', left_on='patient_id', right_on='id')
                print("len(mdf['id']) =", len(mdf['id']))
                name = filename_in
                name = name[:name.rfind('.')]
                test_df.to_csv(name + 'resultsTests.csv')
                patient_df.to_csv(name + 'resultsPat.csv')
            with Timer('left_merge ', new_line=True):
                #
                mdf.to_csv(name + 'resultsTestsMerged.csv')

                pat_tested_covid = s.get(filtered_tests['result']).create_like(filtered_pat, 'result')
                r_fields = (s.get(filtered_tests['result']),)
                r_results_fields = (pat_tested_covid,)
                print(len(patient_id.data))
                print(len(id.data))
                s.merge_left(patient_id, id, r_fields, r_results_fields)
                print(s.get(filtered_pat['id']).data[:].shape)
                print(s.get(filtered_pat['result']).data[:].shape)


list_fields = ['age', 'bmi_clean', 'already_had_covid', 'ethnicity', 'gender', 'id','blood_group']
file_in = '/home/ben/covid/ds_20200929_full.hdf5'
file_out = '/home/ben/covid/ds_michela_temp.hdf5'

#
#merginResults(file_in,file_out)
selectCleanPatients_bmi_age_year(file_in,file_out,list_fields)