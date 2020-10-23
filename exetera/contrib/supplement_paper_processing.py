#!/usr/bin/env python

from collections import defaultdict
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import time

import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils
from exetera.core.persistence import DataStore
from exetera.core.session import Session
from exetera.processing.nat_medicine_model import nature_medicine_model_1
from exetera.processing.effective_test_date import effective_test_date


# patient fields to export
patient_fields_to_export =\
    ['id', 'created_at', 'created_at_day', 'updated_at', 'updated_at_day', 'country_code',
     'version', 'a1c_measurement_mmol', 'a1c_measurement_mmol_valid', 'a1c_measurement_percent',
     'a1c_measurement_percent_valid', 'activity_change', 'age', 'alcohol_change',
     'already_had_covid', 'always_used_shortage', 'assessment_count', 'blood_group', 'bmi',
     'bmi_clean', 'cancer_type', 'contact_health_worker', # 'daily_assessment_count',
     # 'diabetes_diagnosis_year', 'diabetes_diagnosis_year_valid', 'diabetes_oral_biguanide',
     # 'diabetes_oral_dpp4', 'diabetes_oral_meglitinides', 'diabetes_oral_other_medication',
     # 'diabetes_oral_sglt2', 'diabetes_oral_sulfonylurea', 'diabetes_oral_thiazolidinediones',
     # 'diabetes_treatment_basal_insulin', 'diabetes_treatment_insulin_pump',
     # 'diabetes_treatment_lifestyle', 'diabetes_treatment_none',
     # 'diabetes_treatment_other_injection', 'diabetes_treatment_other_oral',
     # 'diabetes_treatment_pfnts', 'diabetes_treatment_rapid_insulin',
     'diabetes_type',
     'diabetes_type_other',
     # 'diabetes_uses_cgm',
     'diet_change', 'does_chemotherapy', 'ethnicity',
     'ever_had_covid_test', 'first_assessment_day', 'gender', 'has_asthma', 'has_cancer',
     'has_diabetes', 'has_eczema', 'has_hayfever', 'has_heart_disease', 'has_kidney_disease',
     'has_lung_disease', 'has_lung_disease_only', 'have_used_PPE',
     'have_worked_in_hospital_care_facility', 'have_worked_in_hospital_clinic',
     'have_worked_in_hospital_home_health', 'have_worked_in_hospital_inpatient',
     'have_worked_in_hospital_other', 'have_worked_in_hospital_outpatient',
     'have_worked_in_hospital_school_clinic', 'healthcare_professional', 'height_cm',
     'height_cm_clean', 'height_cm_valid', 'help_available', 'housebound_problems',
     'ht_combined_oral_contraceptive_pill', 'ht_depot_injection_or_implant',
     'ht_hormone_treatment_therapy', 'ht_mirena_or_other_coil', 'ht_none',
     'ht_oestrogen_hormone_therapy', 'ht_pfnts', 'ht_progestone_only_pill',
     'health_worker_with_contact', 'ht_testosterone_hormone_therapy', 'imd_decile', 'imd_rank',
     'interacted_patients_with_covid', 'interacted_with_covid', 'is_carer_for_community',
     'is_pregnant', 'is_smoker', 'last_asked_level_of_isolation',
     'last_asked_level_of_isolation_day', 'last_assessment_day', 'lifestyle_version',
     'limited_activity', 'mobility_aid', 'need_inside_help', 'need_outside_help', 'needs_help',
     'never_used_shortage',
     # 'past_symptom_abdominal_pain', 'past_symptom_anosmia',
     # 'past_symptom_chest_pain', 'past_symptom_delirium', 'past_symptom_diarrhoea',
     # 'past_symptom_fatigue', 'past_symptom_fever', 'past_symptom_hoarse_voice',
     # 'past_symptom_persistent_cough', 'past_symptom_shortness_of_breath',
     # 'past_symptom_skipped_meals', 'past_symptoms_changed', 'past_symptoms_days_ago',
     # 'past_symptoms_days_ago_valid',
     'period_frequency', 'period_status', 'period_stopped_age',
     'period_stopped_age_valid', 'pregnant_weeks', 'pregnant_weeks_valid', 'race_is_other',
     'race_is_prefer_not_to_say', 'race_is_uk_asian', 'race_is_uk_black', 'race_is_uk_chinese',
     'race_is_uk_middle_eastern', 'race_is_uk_mixed_other', 'race_is_uk_mixed_white_black',
     'race_is_uk_white', 'race_is_us_asian', 'race_is_us_black', 'race_is_us_hawaiian_pacific',
     'race_is_us_indian_native', 'race_is_us_white', 'race_other', 'reported_by_another',
     'same_household_as_reporter', 'se_postcode', 'smoked_years_ago', 'smoked_years_ago_valid',
     'smoker_status', 'snacking_change', 'sometimes_used_shortage', 'still_have_past_symptoms',
     'takes_any_blood_pressure_medications', 'takes_aspirin',
     'takes_blood_pressure_medications_pril', 'takes_blood_pressure_medications_sartan',
     'takes_corticosteroids', 'takes_immunosuppressants', 'test_count', 'unwell_month_before',
     # 'vs_asked_at', 'vs_asked_at_day', 'vs_garlic', 'vs_multivitamins', 'vs_none', 'vs_omega_3',
     # 'vs_pftns', 'vs_probiotics', 'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc',
     'weight_change',
     'weight_change_kg', 'weight_change_kg_valid', 'weight_change_pounds',
     'weight_change_pounds_valid', 'weight_kg', 'weight_kg_clean', 'weight_kg_valid', 'zipcode']


def supplement_paper_processing_2(ds, src_data, dest_data):
    src_ptnts = src_data['patients']
    src_asmts = src_data['assessments']
    src_tests = src_data['tests']

    a_cats_ = ds.get_reader(src_asmts['created_at'])[:]
    start_timestamp = a_cats_.min()
    end_timestamp = a_cats_.max()

    supplement_fields = ('vs_garlic', 'vs_multivitamins', 'vs_none', 'vs_omega_3',
                         'vs_pftns', 'vs_probiotics', 'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc')

    verbose = True
    def filter_status(verbose, name, field, indent=0):
        if verbose:
            preamble = ''.join([" "] * indent)
            print('{}filter after {}: {}/{}'.format(
                preamble, name, np.count_nonzero(field), np.count_nonzero(field == False)))

    # initial patient filtering / flattening
    # --------------------------------------
    if 'filtered_patients' not in dest_data:
        with utils.Timer("initial patient filtering / flattening", new_line=True):
            flt_ptnts = dest_data.create_group('filtered_patients')

            patient_filter = np.ones(len(ds.get_reader(src_ptnts['id'])), dtype=np.bool)
            print("  Initial patient count; {} patients included".format(len(patient_filter)))

            patient_filter = np.ones(len(ds.get_reader(src_ptnts['id'])), dtype=np.bool)
            for k in ('16_to_90_years', '40_to_200_kg', '110_to_220_cm', '15_to_55_bmi'):
                patient_filter = patient_filter & ds.get_reader(src_ptnts[k])[:]
                filter_status(verbose, "{}".format(k), patient_filter, 2)


            # get gender filter
            ptnt_gender = ds.get_reader(src_ptnts['gender'])[:]
            gender_filter = (ptnt_gender == 1) | (ptnt_gender == 2)
            patient_filter = patient_filter & gender_filter
            del gender_filter
            filter_status(verbose, 'gender', patient_filter, 2)

            # get filter for people who were asked questions
            vs_asked_filter = ds.get_reader(src_ptnts['vs_asked_at'])[:] != 0
            patient_filter = patient_filter & vs_asked_filter
            del vs_asked_filter
            filter_status(verbose, 'vs_asked_at', patient_filter, 2)


            # get filter for people with assessments
            has_assessments_filter = ds.get_reader(src_ptnts['assessment_count'])[:] > 0
            patient_filter = patient_filter & has_assessments_filter
            del has_assessments_filter
            filter_status(verbose, 'assessment_count', patient_filter, 2)

            # filter and flatten patient fields
            # ---------------------------------

            for k in patient_fields_to_export:
                with utils.Timer("  filtering {}".format(k)):
                    if k == 'gender':
                        r = ds.get_reader(src_ptnts[k])
                        ds.apply_filter(patient_filter, r[:] - 1,
                                        r.get_writer(flt_ptnts, k))
                    else:
                        r = ds.get_reader(src_ptnts[k])
                        ds.apply_filter(patient_filter, r, r.get_writer(flt_ptnts, k))

            # filter and flatten supplement fields
            for s in supplement_fields:
                with utils.Timer("  filtering {}".format(s)):
                    reader = ds.get_reader(src_ptnts[s])
                    writer = ds.get_numeric_writer(flt_ptnts, s, 'int8')
                    filtered = ds.apply_filter(patient_filter, reader)
                    filtered = filtered.astype('int8')
                    filtered = np.where((filtered - 1) < 0, 0, filtered - 1)
                    writer.write(filtered)

            for s in supplement_fields:
                print("  {}".format(s), utils.build_histogram(ds.get_reader(flt_ptnts[s])[:]))

            print("applying asked about supplements filter; {} patients included".format(
                np.count_nonzero(patient_filter == True)
            ))
            patient_indices = np.arange(len(patient_filter))[patient_filter != 0]
            ds.get_numeric_writer(flt_ptnts, 'patient_indices', 'int64').write(patient_indices)

    else:
        print('  filtered patients already calculated')
        flt_ptnts = dest_data['filtered_patients']

    # initial assessment filtering / flattening
    # -----------------------------------------
    if 'filtered_assessments' not in dest_data.keys():
        with utils.Timer("filtering / flattening assessments", new_line=True):
            flt_asmts = dest_data.create_group('filtered_assessments')

            with utils.Timer(f"filtering out orphaned assessments"):
                asmt_not_orphaned_filter =\
                    persistence.foreign_key_is_in_primary_key(
                        ds.get_reader(flt_ptnts['id']), ds.get_reader(src_asmts['patient_id']))

            asmt_filter = asmt_not_orphaned_filter

            symptom_operators = ('persistent_cough', 'skipped_meals', 'loss_of_smell',
                                 'fatigue')
            symptom_thresholds = {s: 2 for s in symptom_operators}
            symptom_thresholds['fatigue'] = 3
            for s in symptom_operators:
                with utils.Timer(f"flattening and filtering {s}"):
                    reader = ds.get_reader(src_asmts[s])
                    writer = ds.get_numeric_writer(flt_asmts, s, 'uint8')
                    writer.write(
                        ds.apply_filter(asmt_filter, reader[:] >= symptom_thresholds[s]))

            # filter assessment_ids
            asmt_core_fields = ('id', 'patient_id', 'created_at', 'updated_at',
                                'tested_covid_positive')

            for k in asmt_core_fields:
                with utils.Timer(f"filtering '{k}'"):
                    reader = ds.get_reader(src_asmts[k])
                    ds.apply_filter(asmt_filter, reader,
                                    reader.get_writer(flt_asmts, k))

            with utils.Timer("converting and filtering 'location' -> 'hospitalization' field"):
                asmt_locations = ds.get_reader(src_asmts['location'])
                ds.apply_filter(asmt_filter, asmt_locations[:] > 1,
                                ds.get_numeric_writer(flt_asmts, 'hospitalization', 'uint8'))

            with utils.Timer('flattening treatment field'):
                asmt_treatment = ds.get_reader(src_asmts['treatment'])
                ds.apply_filter(asmt_filter, asmt_treatment[:] > 1,
                                ds.get_numeric_writer(flt_asmts, 'treatment', 'uint8'))

            with utils.Timer('flattening health_status field'):
                ds.apply_filter(asmt_filter, ds.get_reader(src_asmts['health_status'])[:] < 2,
                                ds.get_numeric_writer(flt_asmts, 'healthy', 'bool'))
    else:
        print('filtered assessments already calculated')
        flt_asmts = dest_data['filtered_assessments']

    # calculate initially healthy flag and join to patients
    # -----------------------------------------------------
    if 'initially_healthy' not in flt_ptnts:
        with utils.Timer('determining initially unhealthy patients', new_line=True):
            a_pids_ = ds.get_reader(flt_asmts['patient_id'])[:]
            a_hlts_ = ds.get_reader(flt_asmts['healthy'])[:]
            print("  ", sorted(utils.build_histogram(a_hlts_)))
            asmt_spans = ds.get_spans(a_pids_)
            first_asmt_indices = ds.apply_spans_index_of_first(asmt_spans)
            f_a_pids_ = ds.apply_indices(first_asmt_indices, a_pids_)
            f_a_hlts_ = ds.apply_indices(first_asmt_indices, a_hlts_)
            print("  len(f_a_hlts_):", len(f_a_hlts_))
            d = dict()
            for i_r in range(len(f_a_hlts_)):
                d[f_a_pids_[i_r]] = f_a_hlts_[i_r]
            # print("  ", np.array_equal(ds.get_reader(flt_ptnts['id'])[:], f_a_pids_))
            p_ids_ = ds.get_reader(flt_ptnts['id'])[:]
            # print("  initially_health eq:", np.array_equal(f_a_pids_, p_ids_))
            p_init_hlt = np.zeros(len(p_ids_), dtype=np.bool)
            for i_p, p in enumerate(p_ids_):
                if p in d:
                    v = d[p]
                    p_init_hlt[i_p] = v
            # print("  check result eq:", np.array_equal(f_a_hlts_, p_init_hlt))
            ds.get_numeric_writer(flt_ptnts, 'initially_healthy', 'bool').write(p_init_hlt)

    # initial test filtering /flattening and date calculation
    # -------------------------------------------------------
    if 'filtered_tests' not in dest_data.keys():
        with utils.Timer("filtering tests", new_line=True):
            flt_tests = dest_data.create_group('filtered_tests')

            # Filter tests
            # ============

            t_cats = ds.get_reader(src_tests['created_at'])
            t_dts = ds.get_reader(src_tests['date_taken_specific'])
            t_dsbs = ds.get_reader(src_tests['date_taken_between_start'])
            t_dsbe = ds.get_reader(src_tests['date_taken_between_end'])
            t_effd_, t_filt_ =\
                effective_test_date(ds, start_timestamp, end_timestamp,
                                    t_cats, t_dts, t_dsbs, t_dsbe,
                                    flt_tests, 'eff_test_date', flt_tests, 'eff_test_filter')

            for k in ('patient_id', 'result'):
                reader = ds.get_reader(src_tests[k])
                writer = reader.get_writer(flt_tests, k)
                writer.write(ds.apply_filter(t_filt_, reader))
            etd_r = ds.get_reader(flt_tests['eff_test_date'])
            etd_w = etd_r.get_writer(flt_tests, 'eff_test_date', write_mode='overwrite')
            filt_effd_ = ds.apply_filter(t_filt_, t_effd_)
            etd_w.write(filt_effd_)

            # get test months
            t_month = ds.get_fixed_string_writer(flt_tests, 'test_month', 7)
            months_ = np.zeros(len(filt_effd_), dtype='S7')
            for i_r in range(len(months_)):
                dt = datetime.fromtimestamp(filt_effd_[i_r])
                months_[i_r] = '{}-{:02}'.format(dt.year, dt.month).encode()
            t_month.write(months_)
    else:
        print('filtered test fields already calculated')
        flt_tests = dest_data['filtered_tests']

    # join age, gender to assessments
    # -------------------------------
    if 'age' not in flt_asmts:
        with utils.Timer("mapping age / gender from filtered patients to filtered assessments",
                         new_line=True):
            p_ids_ = ds.get_reader(flt_ptnts['id'])[:]
            a_pids_ = ds.get_reader(flt_asmts['patient_id'])[:]
            with utils.Timer('  map assessment patient_ids to patient_ids'):
                asmt_to_ptnt = ds.get_index(p_ids_, a_pids_)

            with utils.Timer('  map patient age to assessment space'):
                ptnt_ages = ds.get_reader(flt_ptnts['age'])
                ptnt_ages.get_writer(flt_asmts, 'age').write(ptnt_ages[:][asmt_to_ptnt])

            with utils.Timer('  map patient gender to assessment gender'):
                ptnt_genders = ds.get_reader(flt_ptnts['gender'])
                ptnt_genders.get_writer(flt_asmts, 'gender').write(
                    ptnt_genders[:][asmt_to_ptnt])

            a_to_p = ds.get_index(p_ids_, a_pids_)

            a_ages_ = ds.apply_indices(a_to_p, ptnt_ages[:])
            ptnt_ages.get_writer(flt_asmts, 'age', write_mode='overwrite').write(a_ages_)
            a_genders_ = ds.apply_indices(a_to_p, ptnt_genders[:])
            ptnt_genders.get_writer(flt_asmts, 'gender',
                                    write_mode='overwrite').write(a_genders_)

    # calculate and select maximum predicted covid
    # --------------------------------------------
    if 'final_assessments' not in dest_data.keys():
        with utils.Timer('running covid prediction', new_line=True):
            t0 = time.time()
            scores = nature_medicine_model_1(ds.get_reader(flt_asmts['persistent_cough']),
                                             ds.get_reader(flt_asmts['skipped_meals']),
                                             ds.get_reader(flt_asmts['loss_of_smell']),
                                             ds.get_reader(flt_asmts['fatigue']),
                                             ds.get_reader(flt_asmts['age']),
                                             ds.get_reader(flt_asmts['gender']))
            ds.get_numeric_writer(flt_asmts, 'prediction_score', 'float32',
                                  writemode='overwrite').write(scores)

            predictions = scores > 0.5
            print("  covid prediction run in {}s".format(time.time() - t0))
            print("  covid prediction covid / not covid",
                  np.count_nonzero(predictions == 1),
                  np.count_nonzero(predictions == 0))
            ds.get_numeric_writer(flt_asmts, 'prediction', 'uint8',
                                  writemode='overwrite').write(predictions)

            # get the maximum prediction
            with utils.Timer("  getting spans for assessment patient_ids"):
                spans = ds.get_spans(ds.get_reader(flt_asmts['patient_id']))

            with utils.Timer("  getting max predictions for each patient"):
                max_score_indices = ds.apply_spans_index_of_max(spans, scores)

            print('  len(flt_ptnts["id"]:', len(ds.get_reader(flt_ptnts['id'])))
            print('  len(max_score_indices):', len(max_score_indices), max_score_indices[:10])

            fnl_asmts = dest_data.create_group('final_assessments')
            # apply the indices to the filtered assessments
            for k in flt_asmts.keys():
                with utils.Timer(f"applying selected_assessment indices to {k} "):
                    reader = ds.get_reader(flt_asmts[k])
                    ds.apply_indices(max_score_indices, reader,
                                   reader.get_writer(fnl_asmts, k))
                    # reader.get_writer(fnl_asmts, k).write(reader[:][max_score_indices])

            print(flt_asmts.keys())
            print(fnl_asmts.keys())

            flt_ptnts_ = ds.get_reader(flt_ptnts['id'])[:]
            fnl_asmts_ = ds.get_reader(fnl_asmts['patient_id'])[:]
            print('  len(flt_ptnts_):', len(flt_ptnts_))
            print('  len(fnl_asmts_):', len(fnl_asmts_))
            final_p_indices = ds.get_index(flt_ptnts_, fnl_asmts_)
            print("  len(final_p_indices):", len(final_p_indices))

            if not np.array_equal(flt_ptnts_[final_p_indices], fnl_asmts_):
                raise ValueError("filtered patients should map exactly with final assessments "
                                 "but don't!")

            gfpi = ds.get_or_create_group(dest_data, 'final_patient_index')
            ds.get_numeric_writer(gfpi, 'final_p_indices', 'int32',
                                  writemode='overwrite').write(final_p_indices)
            print(final_p_indices[:50])
    else:
        print('final_assessments already calculated')
        fnl_asmts = dest_data['final_assessments']

    # select maximum assessment test result and month
    # -----------------------------------------------
    if 'old_test_result' not in flt_ptnts.keys():
        with utils.Timer('collating max assessment test results / months', new_line=True):
            p_ids_ = ds.get_reader(flt_ptnts['id'])[:]
            print("  p_ids_", len(p_ids_))
            a_pids = ds.get_reader(flt_asmts['patient_id'])
            a_pids_ = a_pids[:]
            print("  a_pids_", len(a_pids_))
            a_cats = ds.get_reader(flt_asmts['created_at'])
            a_cats_ = a_cats[:]
            print("  a_cats_", len(a_cats_))
            a_tcps = ds.get_reader(flt_asmts['tested_covid_positive'])
            a_tcps_ = a_tcps[:]
            print("  a_tcps_", len(a_tcps_))
            asmt_spans = ds.get_spans(a_pids)

            asmt_test_indices = ds.apply_spans_index_of_max(asmt_spans, a_tcps)
            print('  len(asmt_test_indices):', len(asmt_test_indices))
            d = dict()
            for ia, a in enumerate(asmt_test_indices):
                d[a_pids_[a]] = (a_tcps_[a], a_cats_[a], ia)

            max_results_ = np.zeros(len(p_ids_), dtype=np.int8)
            result_months_ = np.zeros(len(p_ids_), dtype='S7')
            with utils.Timer("  mapping maximum assessment test result / month to patients"):
                for ip, p in enumerate(p_ids_):
                    if p in d:
                        v = d[p]
                        max_results_[ip] = v[0]
                        dt = datetime.fromtimestamp(v[1])
                        result_months_[ip] = '{}-{:02}'.format(dt.year, dt.month).encode()

            print(np.count_nonzero(max_results_ > 0))

            ds.get_numeric_writer(flt_ptnts, 'old_test_result', 'int8',
                                  writemode='overwrite').write(max_results_)
            ds.get_fixed_string_writer(flt_ptnts, 'old_test_month', 7,
                                       writemode='overwrite').write(result_months_)

    # select maximum test result and month
    # ------------------------------------
    if 'new_test_result' not in flt_ptnts.keys():
        with utils.Timer("calculating max test results and month", new_line=True):
            print("  ", flt_ptnts.keys())
            t_pids_ = ds.get_reader(flt_tests['patient_id'])[:]
            t_rslts_ = ds.get_reader(flt_tests['result'])[:]
            t_months_ = ds.get_reader(flt_tests['test_month'])[:]
            test_spans = ds.get_spans(t_pids_)
            max_result_indices = ds.apply_spans_index_of_max(test_spans, t_rslts_)

            d = dict()
            for i_t, t in enumerate(max_result_indices):
                d[t_pids_[t]] = (t_rslts_[t], t_months_[t], i_t)

            # build a dictionary for max results by pid
            p_ids_ = ds.get_reader(flt_ptnts['id'])[:]
            max_results_ = np.zeros(len(p_ids_), dtype=np.int8)
            result_months_ = np.zeros(len(p_ids_), dtype='S7')
            for i_p, p in enumerate(p_ids_):
                if p in d:
                    v = d[p]
                    max_results_[i_p] = v[0]
                    result_months_[i_p] = v[1]

            ds.get_numeric_writer(flt_ptnts, 'new_test_result', 'int8',
                                  writemode='overwrite').write(max_results_)
            ds.get_fixed_string_writer(flt_ptnts, 'new_test_month', 7,
                                       writemode='overwrite').write(result_months_)

    # join assessment fields to patients
    # ----------------------------------
    print('patients')
    print(flt_ptnts.keys())
    p_ids_ = ds.get_reader(flt_ptnts['id'])[:]
    print(len(p_ids_))

    print('assessments')
    print(fnl_asmts.keys())
    a_pids_ = ds.get_reader(fnl_asmts['patient_id'])[:]
    print(len(a_pids_))

    print(np.array_equal(p_ids_, a_pids_))

    initial_fields = ['id', '']
    asmt_fields = ['fatigue', 'loss_of_smell', 'persistent_cough', 'skipped_meals',
                   'hospitalization', 'treatment', 'prediction_score', 'sore_throat', 'prediction']

    first_patient_fields = ['id', 'created_at', 'updated_at', 'version', 'country_code']
    last_patient_fields = ['old_test_result', 'old_test_month', 'new_test_result', 'new_test_month']
    patient_fields = [p for p in flt_ptnts.keys() if p not in first_patient_fields + last_patient_fields]
    patient_fields = first_patient_fields + patient_fields + last_patient_fields
    fields_to_export = [(p, flt_ptnts[p]) for p in patient_fields] + \
                       [('asmt_{}'.format(a), fnl_asmts[a]) for a in asmt_fields]
    exporter.export_to_csv('/home/ben/covid/supplements_patients.csv', ds, fields_to_export)


# =============================================================================


def supplement_paper_processing_diet(session, src_data, dest_data, include_diet=False):
    src_ptnts = src_data['patients']
    src_asmts = src_data['assessments']
    src_tests = src_data['tests']
    src_diet = src_data['diet']

    a_cats_ = session.get_reader(src_asmts['created_at'])[:]
    start_timestamp = a_cats_.min()
    end_timestamp = a_cats_.max()

    supplement_fields = ('vs_garlic', 'vs_multivitamins', 'vs_none', 'vs_omega_3',
                         'vs_pftns', 'vs_probiotics', 'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc')
    ptnt_sup_fields = ('vs_none', 'vs_omega_3', 'vs_multivitamins',
                       'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc', 'vs_garlic', 'vs_probiotics')
    diet_sup_fields = ('takes_supplements', 'supplements_omega3', 'supplements_multivitamin',
                       'supplements_vitamin_c', 'supplements_vitamin_d', 'supplements_zinc',
                       'supplements_garlic', 'supplements_probiotic')
    consistent_sup_fields = ['{}_consistent'.format(f) for f in ptnt_sup_fields]

    verbose = True
    def filter_status(verbose, name, field, indent=0):
        if verbose:
            preamble = ''.join([" "] * indent)
            print('{}filter after {}: {}/{}'.format(
                preamble, name, np.count_nonzero(field), np.count_nonzero(field == False)))

    # initial patient filtering / flattening
    # --------------------------------------
    if 'filtered_patients' not in dest_data:
        with utils.Timer("initial patient filtering / flattening", new_line=True):
            flt_ptnts = dest_data.create_group('filtered_patients')

            patient_filter = np.ones(len(session.get_reader(src_ptnts['id'])), dtype=np.bool)
            print("  Initial patient count; {} patients included".format(len(patient_filter)))

            patient_filter = np.ones(len(session.get_reader(src_ptnts['id'])), dtype=np.bool)
            for k in ('16_to_90_years', '40_to_200_kg', '110_to_220_cm', '15_to_55_bmi'):
                patient_filter = patient_filter & session.get_reader(src_ptnts[k])[:]
                filter_status(verbose, "{}".format(k), patient_filter, 2)

            # get gender filter
            ptnt_gender = session.get_reader(src_ptnts['gender'])[:]
            gender_filter = (ptnt_gender == 1) | (ptnt_gender == 2)
            patient_filter = patient_filter & gender_filter
            del gender_filter
            filter_status(verbose, 'gender', patient_filter, 2)

            # get filter for people who were asked questions
            vs_asked_filter = session.get_reader(src_ptnts['vs_asked_at'])[:] != 0
            patient_filter = patient_filter & vs_asked_filter
            del vs_asked_filter
            filter_status(verbose, 'vs_asked_at', patient_filter, 2)

            # get filter for people with assessments
            has_assessments_filter = session.get_reader(src_ptnts['assessment_count'])[:] > 0
            patient_filter = patient_filter & has_assessments_filter
            del has_assessments_filter
            filter_status(verbose, 'assessment_count', patient_filter, 2)

            # get filter for diet count
            diet_count_filter = session.get_reader(src_ptnts['diet_counts'])[:] > 0
            patient_filter = patient_filter & diet_count_filter
            del diet_count_filter
            filter_status(verbose, 'diet_counts', patient_filter, 2)

            # filter and flatten patient fields
            # ---------------------------------

            for k in patient_fields_to_export:
                with utils.Timer("  filtering {}".format(k)):
                    if k == 'gender':
                        r = session.get_reader(src_ptnts[k])
                        session.apply_filter(patient_filter, r[:] - 1,
                                             r.get_writer(flt_ptnts, k))
                    else:
                        r = session.get_reader(src_ptnts[k])
                        session.apply_filter(patient_filter, r, r.get_writer(flt_ptnts, k))

            # filter and flatten supplement fields
            for s in supplement_fields:
                with utils.Timer("  filtering {}".format(s)):
                    reader = session.get_reader(src_ptnts[s])
                    writer = session.get_numeric_writer(flt_ptnts, s, 'int8')
                    filtered = session.apply_filter(patient_filter, reader)
                    filtered = filtered.astype('int8')
                    filtered = np.where((filtered - 1) < 0, 0, filtered - 1)
                    writer.write(filtered)

            for s in supplement_fields:
                print("  {}".format(s), utils.build_histogram(session.get_reader(flt_ptnts[s])[:]))

            print("applying asked about supplements filter; {} patients included".format(
                np.count_nonzero(patient_filter == True)
            ))
            patient_indices = np.arange(len(patient_filter))[patient_filter != 0]
            session.get_numeric_writer(flt_ptnts, 'patient_indices', 'int64').write(patient_indices)

    else:
        print('  filtered patients already calculated')
        flt_ptnts = dest_data['filtered_patients']

    # creating supplement consistency fields
    # --------------------------------------
    if include_diet:
        if 'vs_none_consistent' not in flt_ptnts:

            with utils.Timer("adding supplement consistency fields using diet", new_line=True):
                # mapping diet fields to patient
                d_pids_ = session.get_reader(src_diet['patient_id'])[:]
                diet_filter = persistence.foreign_key_is_in_primary_key(
                    session.get_reader(flt_ptnts['id']), d_pids_)

                print("diet_entries before filter", len(d_pids_))
                print("unique diet patients before filter:", len(np.unique(d_pids_)))
                d_pids_ = session.apply_filter(diet_filter, d_pids_)
                print("diet_entries after filter", len(d_pids_))
                print("unique diet patients after filter:", len(np.unique(d_pids_)))

                print("flattening diet supplement fields")
                flattened_fields = list()
                for s in diet_sup_fields:
                    f = session.get_reader(src_diet[s])[:]
                    f = session.apply_filter(diet_filter, f)
                    if s == 'takes_supplements':
                        print("ts:", np.unique(f, return_counts=True))
                        f = np.where(f == 2, 0, 1)
                    else:
                        f = np.where(f == 2, 1, 0)
                    flattened_fields.append(f)
                    print(s, np.unique(f, return_counts=True))

                f_p_ids = session.get_reader(flt_ptnts['id'])[:]

                print("checking consistency within multiple diet entries per patient")
                consistent_fields = list()
                for i_s, s in enumerate(diet_sup_fields):
                    diet_per_patient = defaultdict(int)
                    f = flattened_fields[i_s]
                    for i in range(len(d_pids_)):
                        diet_per_patient[d_pids_[i]] |= 2 if f[i] else 1
                    l_diet_per_patient = sorted(list(diet_per_patient.items()))
                    print(np.unique(list(diet_per_patient.values()), return_counts=True))
                    f_d_pids = [v[0] for v in l_diet_per_patient]
                    # print(len(f_p_ids), len(f_d_pids))
                    # print(np.array_equal(f_p_ids, f_d_pids))
                    consistent_fields.append(
                        np.asarray([v[1] for v in l_diet_per_patient], dtype=np.int8))

                print("checking consistency between diet answers and patient supplement answers")
                for i_s, s in enumerate(diet_sup_fields):
                    vs_field = session.get_reader(flt_ptnts[ptnt_sup_fields[i_s]])[:]
                    consistent = consistent_fields[i_s] | np.where(vs_field, 2, 1)
                    print(np.unique(consistent, return_counts=True))

                    writer = session.get_numeric_writer(
                        flt_ptnts, consistent_sup_fields[i_s], 'int8')
                    writer.write(consistent < 3)


    # initial assessment filtering / flattening
    # -----------------------------------------
    if 'filtered_assessments' not in dest_data.keys():
        with utils.Timer("filtering / flattening assessments", new_line=True):
            flt_asmts = dest_data.create_group('filtered_assessments')

            with utils.Timer(f"filtering out orphaned assessments"):
                asmt_filter =\
                    persistence.foreign_key_is_in_primary_key(
                        session.get_reader(flt_ptnts['id']), session.get_reader(src_asmts['patient_id']))

            symptom_operators = ('persistent_cough', 'skipped_meals', 'loss_of_smell',
                                 'fatigue', 'sore_throat')
            symptom_thresholds = {s: 2 for s in symptom_operators}
            symptom_thresholds['fatigue'] = 3
            for s in symptom_operators:
                with utils.Timer(f"flattening and filtering {s}"):
                    reader = session.get_reader(src_asmts[s])
                    writer = session.get_numeric_writer(flt_asmts, s, 'uint8')
                    writer.write(
                        session.apply_filter(asmt_filter, reader[:] >= symptom_thresholds[s]))

            # filter assessment_ids
            asmt_core_fields = ('id', 'patient_id', 'created_at', 'updated_at',
                                'tested_covid_positive')

            for k in asmt_core_fields:
                with utils.Timer(f"filtering '{k}'"):
                    reader = session.get_reader(src_asmts[k])
                    session.apply_filter(asmt_filter, reader,
                                         reader.get_writer(flt_asmts, k))

            with utils.Timer("converting and filtering 'location' -> 'hospitalization' field"):
                asmt_locations = session.get_reader(src_asmts['location'])
                session.apply_filter(asmt_filter, asmt_locations[:] > 1,
                                     session.get_numeric_writer(flt_asmts, 'hospitalization', 'uint8'))

            with utils.Timer('flattening treatment field'):
                asmt_treatment = session.get_reader(src_asmts['treatment'])
                session.apply_filter(asmt_filter, asmt_treatment[:] > 1,
                                     session.get_numeric_writer(flt_asmts, 'treatment', 'uint8'))

            with utils.Timer('flattening health_status field'):
                session.apply_filter(asmt_filter, session.get_reader(src_asmts['health_status'])[:] < 2,
                                     session.get_numeric_writer(flt_asmts, 'healthy', 'bool'))
    else:
        print('filtered assessments already calculated')
        flt_asmts = dest_data['filtered_assessments']

    # calculate initially healthy flag and join to patients
    # -----------------------------------------------------
    if 'initially_healthy' not in flt_ptnts:
        with utils.Timer('determining initially unhealthy patients', new_line=True):
            a_pids_ = session.get_reader(flt_asmts['patient_id'])[:]
            a_hlts_ = session.get_reader(flt_asmts['healthy'])[:]
            print("  ", sorted(utils.build_histogram(a_hlts_)))
            asmt_spans = session.get_spans(a_pids_)
            first_asmt_indices = session.apply_spans_index_of_first(asmt_spans)
            f_a_pids_ = session.apply_indices(first_asmt_indices, a_pids_)
            f_a_hlts_ = session.apply_indices(first_asmt_indices, a_hlts_)
            print("  len(f_a_hlts_):", len(f_a_hlts_))
            d = dict()
            for i_r in range(len(f_a_hlts_)):
                d[f_a_pids_[i_r]] = f_a_hlts_[i_r]
            # print("  ", np.array_equal(ds.get_reader(flt_ptnts['id'])[:], f_a_pids_))
            p_ids_ = session.get_reader(flt_ptnts['id'])[:]
            # print("  initially_health eq:", np.array_equal(f_a_pids_, p_ids_))
            p_init_hlt = np.zeros(len(p_ids_), dtype=np.bool)
            for i_p, p in enumerate(p_ids_):
                if p in d:
                    v = d[p]
                    p_init_hlt[i_p] = v
            # print("  check result eq:", np.array_equal(f_a_hlts_, p_init_hlt))
            session.get_numeric_writer(flt_ptnts, 'initially_healthy', 'bool').write(p_init_hlt)

    # initial test filtering /flattening and date calculation
    # -------------------------------------------------------
    if 'filtered_tests' not in dest_data.keys():
        with utils.Timer("filtering tests", new_line=True):
            flt_tests = dest_data.create_group('filtered_tests')

            # Filter tests
            # ============

            t_cats = session.get_reader(src_tests['created_at'])
            t_dts = session.get_reader(src_tests['date_taken_specific'])
            t_dsbs = session.get_reader(src_tests['date_taken_between_start'])
            t_dsbe = session.get_reader(src_tests['date_taken_between_end'])
            t_effd_, t_filt_ =\
                effective_test_date(session, start_timestamp, end_timestamp,
                                    t_cats, t_dts, t_dsbs, t_dsbe,
                                    flt_tests, 'eff_test_date', flt_tests, 'eff_test_filter')

            for k in ('patient_id', 'result'):
                reader = session.get_reader(src_tests[k])
                writer = reader.get_writer(flt_tests, k)
                writer.write(session.apply_filter(t_filt_, reader))
            etd_r = session.get_reader(flt_tests['eff_test_date'])
            etd_w = etd_r.get_writer(flt_tests, 'eff_test_date', write_mode='overwrite')
            filt_effd_ = session.apply_filter(t_filt_, t_effd_)
            etd_w.write(filt_effd_)

            # get test months
            t_month = session.get_fixed_string_writer(flt_tests, 'test_month', 7)
            months_ = np.zeros(len(filt_effd_), dtype='S7')
            for i_r in range(len(months_)):
                dt = datetime.fromtimestamp(filt_effd_[i_r])
                months_[i_r] = '{}-{:02}'.format(dt.year, dt.month).encode()
            t_month.write(months_)
    else:
        print('filtered test fields already calculated')
        flt_tests = dest_data['filtered_tests']

    # join age, gender to assessments
    # -------------------------------
    if 'age' not in flt_asmts:
        with utils.Timer("mapping age / gender from filtered patients to filtered assessments",
                         new_line=True):
            p_ids_ = session.get_reader(flt_ptnts['id'])[:]
            a_pids_ = session.get_reader(flt_asmts['patient_id'])[:]
            with utils.Timer('  map assessment patient_ids to patient_ids'):
                asmt_to_ptnt = session.get_index(p_ids_, a_pids_)

            with utils.Timer('  map patient age to assessment space'):
                ptnt_ages = session.get_reader(flt_ptnts['age'])
                ptnt_ages.get_writer(flt_asmts, 'age').write(ptnt_ages[:][asmt_to_ptnt])

            with utils.Timer('  map patient gender to assessment gender'):
                ptnt_genders = session.get_reader(flt_ptnts['gender'])
                ptnt_genders.get_writer(flt_asmts, 'gender').write(
                    ptnt_genders[:][asmt_to_ptnt])

            a_to_p = session.get_index(p_ids_, a_pids_)

            a_ages_ = session.apply_indices(a_to_p, ptnt_ages[:])
            ptnt_ages.get_writer(flt_asmts, 'age', write_mode='overwrite').write(a_ages_)
            a_genders_ = session.apply_indices(a_to_p, ptnt_genders[:])
            ptnt_genders.get_writer(flt_asmts, 'gender',
                                    write_mode='overwrite').write(a_genders_)

    # calculate and select maximum predicted covid
    # --------------------------------------------
    if 'final_assessments' not in dest_data.keys():
        with utils.Timer('running covid prediction', new_line=True):
            t0 = time.time()
            scores = nature_medicine_model_1(session.get_reader(flt_asmts['persistent_cough']),
                                             session.get_reader(flt_asmts['skipped_meals']),
                                             session.get_reader(flt_asmts['loss_of_smell']),
                                             session.get_reader(flt_asmts['fatigue']),
                                             session.get_reader(flt_asmts['age']),
                                             session.get_reader(flt_asmts['gender']))
            session.get_numeric_writer(flt_asmts, 'prediction_score', 'float32',
                                       writemode='overwrite').write(scores)

            predictions = scores > 0.5
            print("  covid prediction run in {}s".format(time.time() - t0))
            print("  covid prediction covid / not covid",
                  np.count_nonzero(predictions == 1),
                  np.count_nonzero(predictions == 0))
            session.get_numeric_writer(flt_asmts, 'prediction', 'uint8',
                                       writemode='overwrite').write(predictions)

            # get the maximum prediction
            with utils.Timer("  getting spans for assessment patient_ids"):
                spans = session.get_spans(session.get_reader(flt_asmts['patient_id']))

            with utils.Timer("  getting max predictions for each patient"):
                max_score_indices = session.apply_spans_index_of_max(spans, scores)

            print('  len(flt_ptnts["id"]:', len(session.get_reader(flt_ptnts['id'])))
            print('  len(max_score_indices):', len(max_score_indices), max_score_indices[:10])

            fnl_asmts = dest_data.create_group('final_assessments')
            # apply the indices to the filtered assessments
            for k in flt_asmts.keys():
                with utils.Timer(f"applying selected_assessment indices to {k} "):
                    reader = session.get_reader(flt_asmts[k])
                    session.apply_indices(max_score_indices, reader,
                                          reader.get_writer(fnl_asmts, k))
                    # reader.get_writer(fnl_asmts, k).write(reader[:][max_score_indices])

            print(flt_asmts.keys())
            print(fnl_asmts.keys())

            flt_ptnts_ = session.get_reader(flt_ptnts['id'])[:]
            fnl_asmts_ = session.get_reader(fnl_asmts['patient_id'])[:]
            print('  len(flt_ptnts_):', len(flt_ptnts_))
            print('  len(fnl_asmts_):', len(fnl_asmts_))
            final_p_indices = session.get_index(flt_ptnts_, fnl_asmts_)
            print("  len(final_p_indices):", len(final_p_indices))

            if not np.array_equal(flt_ptnts_[final_p_indices], fnl_asmts_):
                raise ValueError("filtered patients should map exactly with final assessments "
                                 "but don't!")

            gfpi = session.get_or_create_group(dest_data, 'final_patient_index')
            session.get_numeric_writer(gfpi, 'final_p_indices', 'int32',
                                       writemode='overwrite').write(final_p_indices)
            print(final_p_indices[:50])
    else:
        print('final_assessments already calculated')
        fnl_asmts = dest_data['final_assessments']

    # select maximum assessment test result and month
    # -----------------------------------------------
    if 'old_test_result' not in flt_ptnts.keys():
        with utils.Timer('collating max assessment test results / months', new_line=True):
            p_ids_ = session.get_reader(flt_ptnts['id'])[:]
            print("  p_ids_", len(p_ids_))
            a_pids = session.get_reader(flt_asmts['patient_id'])
            a_pids_ = a_pids[:]
            print("  a_pids_", len(a_pids_))
            a_cats = session.get_reader(flt_asmts['created_at'])
            a_cats_ = a_cats[:]
            print("  a_cats_", len(a_cats_))
            a_tcps = session.get_reader(flt_asmts['tested_covid_positive'])
            a_tcps_ = a_tcps[:]
            print("  a_tcps_", len(a_tcps_))
            asmt_spans = session.get_spans(a_pids)

            asmt_test_indices = session.apply_spans_index_of_max(asmt_spans, a_tcps)
            print('  len(asmt_test_indices):', len(asmt_test_indices))
            d = dict()
            for ia, a in enumerate(asmt_test_indices):
                d[a_pids_[a]] = (a_tcps_[a], a_cats_[a], ia)

            max_results_ = np.zeros(len(p_ids_), dtype=np.int8)
            result_months_ = np.zeros(len(p_ids_), dtype='S7')
            with utils.Timer("  mapping maximum assessment test result / month to patients"):
                for ip, p in enumerate(p_ids_):
                    if p in d:
                        v = d[p]
                        max_results_[ip] = v[0]
                        dt = datetime.fromtimestamp(v[1])
                        result_months_[ip] = '{}-{:02}'.format(dt.year, dt.month).encode()

            print(np.count_nonzero(max_results_ > 0))

            session.get_numeric_writer(flt_ptnts, 'old_test_result', 'int8',
                                       writemode='overwrite').write(max_results_)
            session.get_fixed_string_writer(flt_ptnts, 'old_test_month', 7,
                                            writemode='overwrite').write(result_months_)

    # select maximum test result and month
    # ------------------------------------
    if 'new_test_result' not in flt_ptnts.keys():
        with utils.Timer("calculating max test results and month", new_line=True):
            print("  ", flt_ptnts.keys())
            t_pids_ = session.get_reader(flt_tests['patient_id'])[:]
            t_rslts_ = session.get_reader(flt_tests['result'])[:]
            t_months_ = session.get_reader(flt_tests['test_month'])[:]
            test_spans = session.get_spans(t_pids_)
            max_result_indices = session.apply_spans_index_of_max(test_spans, t_rslts_)

            d = dict()
            for i_t, t in enumerate(max_result_indices):
                d[t_pids_[t]] = (t_rslts_[t], t_months_[t], i_t)

            # build a dictionary for max results by pid
            p_ids_ = session.get_reader(flt_ptnts['id'])[:]
            max_results_ = np.zeros(len(p_ids_), dtype=np.int8)
            result_months_ = np.zeros(len(p_ids_), dtype='S7')
            for i_p, p in enumerate(p_ids_):
                if p in d:
                    v = d[p]
                    max_results_[i_p] = v[0]
                    result_months_[i_p] = v[1]

            session.get_numeric_writer(flt_ptnts, 'new_test_result', 'int8',
                                       writemode='overwrite').write(max_results_)
            session.get_fixed_string_writer(flt_ptnts, 'new_test_month', 7,
                                            writemode='overwrite').write(result_months_)

    # join assessment fields to patients
    # ----------------------------------
    print('patients')
    print(flt_ptnts.keys())
    p_ids_ = session.get_reader(flt_ptnts['id'])[:]
    print(len(p_ids_))

    print('assessments')
    print(fnl_asmts.keys())
    a_pids_ = session.get_reader(fnl_asmts['patient_id'])[:]
    print(len(a_pids_))

    print(np.array_equal(p_ids_, a_pids_))

    initial_fields = ['id', '']
    asmt_fields = ['fatigue', 'loss_of_smell', 'persistent_cough', 'skipped_meals', 'sore_throat',
                   'hospitalization', 'treatment', 'prediction_score', 'prediction']

    first_patient_fields = ['id', 'created_at', 'updated_at', 'version', 'country_code']
    last_patient_fields = ['old_test_result', 'old_test_month', 'new_test_result', 'new_test_month']
    patient_fields = \
        [p for p in flt_ptnts.keys() if p not in first_patient_fields + last_patient_fields]
    patient_fields = \
        first_patient_fields + patient_fields + last_patient_fields
    fields_to_export = [(p, flt_ptnts[p]) for p in patient_fields] + \
                       [('asmt_{}'.format(a), fnl_asmts[a]) for a in asmt_fields]
    exporter.export_to_csv('/home/ben/covid/supplements_patients.csv', session, fields_to_export)


if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20200914_full.hdf5'
    dest_file = '/home/ben/covid/ds_20200914_supplements.hdf5'
    with h5py.File(src_file, 'r') as src_data:
        with h5py.File(dest_file, 'w') as dest_data:
            supplement_paper_processing_diet(datastore, src_data, dest_data)
            # supplement_paper_processing_diet(Session(), src_data, dest_data)
