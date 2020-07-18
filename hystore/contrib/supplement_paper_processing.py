#!/usr/bin/env python

from collections import defaultdict
from datetime import datetime, timezone
import time

import numpy as np
import h5py
import pandas as pd

from hystore.core import exporter, persistence, utils
from hystore.core.persistence import DataStore
from hystore.processing.nat_medicine_model import nature_medicine_model_1
"""
I then cleaned and recoded the assessments based on my scripts for HRT/menopause and added them to the patient file
after the last column (after bmi_clean).

In the "suppl_assessments..." file you have the symptoms recoded as 1 for TRUE and 0 for FALSE/NA (fatigue and
shortness of breath were coded as in the Nat. Medicine paper); you also have hospitalization with 1 for hospital/back
from hospital and 0 for home/NA, and treatment with 1 for oxygen/ventilation and 0 for any other treatment/NA. You also
have predicted Covid based on the formula in Nat. Medicine.
In the "suppl_w_tested_covid_old..." file you have the answers for covid PCR tests with 1 for positive and  0 for
negative. We excluded NAs and waiting from our analyses so this data frame is smaller. If you want to include NAs in
your analysis you can left_join the "suppl_assessments..." and "suppl_w_tested_covid_old..." data; the patients left
without values for Covid testing will be the ones with NAs or waiting.
Since mid-May, ZOE started storing the tests for Covid in a data frame separate from the assessments - where each row
is the "final" response for a test instead of a daily entry in the assessment table. This is in the
"suppl_w_tested_covid_new..." file. Although this scheme is better for storing test data, patients have multiple tests
and results are often conflicting. For now we decided to code 1 for having any test being positive and 0 for having
only negative results. We also excluded failed tests and waiting from our analyses. Even though the number of patients
testing for Covid are higher here than in the assessments table, the older tests still give me more agreeable results
with predicted Covid.
FYI: I asked Ben about merging older and newer results and he advised against (first because most patients with older
tests are not updating anymore and the ones that do update, again, have conflicting answers).
"""

"""
Process
 * build patient filter from patent-level details
   * apply 16_to_90 filter
   * apply valid hwbmi filter
   * apply answered vitamin questions filter
   * apply has single test result


Cristina vitamin

test_positive = positive either old or new tests
test_positive = negative



logistic regression 
"""


# patient fields to export
patient_fields_to_export =\
    ['id', 'created_at', 'created_at_day', 'updated_at', 'updated_at_day', 'country_code', 'version',
     'a1c_measurement_mmol', 'a1c_measurement_mmol_valid', 'a1c_measurement_percent', 'a1c_measurement_percent_valid',
     'activity_change', 'age', 'alcohol_change', 'already_had_covid', 'always_used_shortage', 'assessment_count',
     'blood_group', 'bmi', 'bmi_clean', 'cancer_type', 'contact_health_worker', 'daily_assessment_count',
     'diabetes_diagnosis_year', 'diabetes_diagnosis_year_valid', 'diabetes_oral_biguanide', 'diabetes_oral_dpp4',
     'diabetes_oral_meglitinides', 'diabetes_oral_other_medication', 'diabetes_oral_sglt2',
     'diabetes_oral_sulfonylurea', 'diabetes_oral_thiazolidinediones', 'diabetes_treatment_basal_insulin',
     'diabetes_treatment_insulin_pump', 'diabetes_treatment_lifestyle', 'diabetes_treatment_none',
     'diabetes_treatment_other_injection', 'diabetes_treatment_other_oral', 'diabetes_treatment_pfnts',
     'diabetes_treatment_rapid_insulin', 'diabetes_type', 'diabetes_type_other', 'diabetes_uses_cgm', 'diet_change',
     'does_chemotherapy', 'ethnicity', 'ever_had_covid_test', 'first_assessment_day', 'gender', 'has_asthma',
     'has_cancer', 'has_diabetes', 'has_eczema', 'has_hayfever', 'has_heart_disease', 'has_kidney_disease',
     'has_lung_disease', 'has_lung_disease_only', 'have_used_PPE', 'have_worked_in_hospital_care_facility',
     'have_worked_in_hospital_clinic', 'have_worked_in_hospital_home_health', 'have_worked_in_hospital_inpatient',
     'have_worked_in_hospital_other', 'have_worked_in_hospital_outpatient', 'have_worked_in_hospital_school_clinic',
     'healthcare_professional', 'height_cm', 'height_cm_clean', 'height_cm_valid', 'help_available',
     'housebound_problems', 'ht_combined_oral_contraceptive_pill', 'ht_depot_injection_or_implant',
     'ht_hormone_treatment_therapy', 'ht_mirena_or_other_coil', 'ht_none', 'ht_oestrogen_hormone_therapy', 'ht_pfnts',
     'ht_progestone_only_pill', 'ht_testosterone_hormone_therapy', 'interacted_patients_with_covid',
     'interacted_with_covid', 'is_carer_for_community', 'is_pregnant', 'is_smoker', 'ladcd',
     'last_asked_level_of_isolation', 'last_asked_level_of_isolation_day', 'last_assessment_day', 'lifestyle_version',
     'limited_activity', 'lsoa11cd', 'lsoa11nm', 'mobility_aid',
     'msoa11cd', 'msoa11nm', 'need_inside_help', 'need_outside_help', 'needs_help', 'never_used_shortage',
     'outward_postcode', 'outward_postcode_latitude', 'outward_postcode_latitude_valid',
     'outward_postcode_longitude', 'outward_postcode_longitude_valid', 'outward_postcode_region',
     'outward_postcode_town_area', 'past_symptom_abdominal_pain', 'past_symptom_anosmia', 'past_symptom_chest_pain',
     'past_symptom_delirium', 'past_symptom_diarrhoea', 'past_symptom_fatigue', 'past_symptom_fever',
     'past_symptom_hoarse_voice', 'past_symptom_persistent_cough', 'past_symptom_shortness_of_breath',
     'past_symptom_skipped_meals', 'past_symptoms_changed', 'past_symptoms_days_ago', 'past_symptoms_days_ago_valid',
     'period_frequency', 'period_status', 'period_stopped_age', 'period_stopped_age_valid', 'pregnant_weeks',
     'pregnant_weeks_valid', 'race_is_other', 'race_is_prefer_not_to_say', 'race_is_uk_asian',
     'race_is_uk_black', 'race_is_uk_chinese', 'race_is_uk_middle_eastern', 'race_is_uk_mixed_other',
     'race_is_uk_mixed_white_black', 'race_is_uk_white', 'race_is_us_asian', 'race_is_us_black',
     'race_is_us_hawaiian_pacific', 'race_is_us_indian_native', 'race_is_us_white', 'race_other',
     'reported_by_another', 'same_household_as_reporter', 'se_postcode', 'smoked_years_ago', 'smoked_years_ago_valid',
     'smoker_status', 'snacking_change', 'sometimes_used_shortage', 'still_have_past_symptoms',
     'takes_any_blood_pressure_medications', 'takes_aspirin', 'takes_blood_pressure_medications_pril',
     'takes_blood_pressure_medications_sartan', 'takes_corticosteroids', 'takes_immunosuppressants', 'test_count',
     'unwell_month_before', 'vs_asked_at', 'vs_asked_at_day', 'vs_garlic', 'vs_multivitamins', 'vs_none', 'vs_omega_3',
     'vs_pftns', 'vs_probiotics', 'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc', 'weight_change',
     'weight_change_kg', 'weight_change_kg_valid', 'weight_change_pounds', 'weight_change_pounds_valid', 'weight_kg',
     'weight_kg_clean', 'weight_kg_valid', 'zipcode']


def supplement_paper_processing(ds, src_data, dest_data, timestamp=str(datetime.now(timezone.utc))):
    src_tests = src_data['tests']
    src_ptnts = src_data['patients']
    src_asmts = src_data['daily_assessments']
    flt_tests = dest_data.create_group('tests')
    flt_ptnts = dest_data.create_group('patients')
    flt_asmts = dest_data.create_group('daily_assessments')
    fnl_asmts = dest_data.create_group('final_assessments')


    # build patient filter from patient-level details
    # -----------------------------------------------

    print(src_ptnts.keys())

    print('vs_none: ', utils.build_histogram(ds.get_reader(src_ptnts['vs_none'])[:]))

    patient_filter = np.ones(len(ds.get_reader(src_ptnts['id'])), dtype=np.bool)
    print(f"Initial patient count; {len(patient_filter)} patients included")

    # get 16_to_90 filter
    filter_16_to_90 = ds.get_reader(src_ptnts['16_to_90_years'])[:]
    patient_filter = patient_filter & filter_16_to_90
    del filter_16_to_90
    print(f"applying age filter; {np.count_nonzero(patient_filter == True)} patients included")

    # get valid weight/height/bmi only
    filter_weight = ds.get_reader(src_ptnts['40_to_200_kg'])[:]
    patient_filter = patient_filter & filter_weight
    del filter_weight
    filter_height = ds.get_reader(src_ptnts['110_to_220_cm'])[:]
    patient_filter = patient_filter & filter_height
    del filter_height
    filter_bmi = ds.get_reader(src_ptnts['15_to_55_bmi'])[:]
    patient_filter = patient_filter & filter_bmi
    del filter_bmi
    print(f"applying height / weight / bmi filters; "
          f"{np.count_nonzero(patient_filter == True)} patients included")

    # get gender filter
    ptnt_gender = ds.get_reader(src_ptnts['gender'])[:]
    gender_filter = (ptnt_gender == 1) | (ptnt_gender == 2)
    patient_filter = patient_filter & gender_filter
    del gender_filter

    # get filter for people who were asked questions
    vs_asked_filter = ds.get_reader(src_ptnts['vs_asked_at'])[:] != 0
    patient_filter = patient_filter & vs_asked_filter
    del vs_asked_filter
    print(f"applying asked about supplements filter; "
          f"{np.count_nonzero(patient_filter == True)} patients included")

    # get filter for people with assessments
    has_assessments_filter = ds.get_reader(src_ptnts['daily_assessment_count'])[:] > 0
    patient_filter = patient_filter & has_assessments_filter
    del has_assessments_filter


    # filter and flatten patient fields
    # ---------------------------------

    # filter id, age, gender
    src_pids = ds.get_reader(src_ptnts['id'])
    ds.apply_filter(patient_filter, src_pids, src_pids.get_writer(flt_ptnts, 'id', timestamp))

    src_ages = ds.get_reader(src_ptnts['age'])
    ds.apply_filter(patient_filter, src_ages, src_ages.get_writer(flt_ptnts, 'age', timestamp))

    src_genders = ds.get_reader(src_ptnts['gender'])
    ds.apply_filter(patient_filter, src_genders[:] - 1,
                    src_genders.get_writer(flt_ptnts, 'gender', timestamp))

    # filter and flatten supplement fields
    supplement_fields = ('vs_garlic', 'vs_multivitamins', 'vs_none', 'vs_omega_3',
                         'vs_pftns', 'vs_probiotics', 'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc')
    for s in supplement_fields:
        t0 = time.time()
        reader = ds.get_reader(src_ptnts[s])
        writer = ds.get_numeric_writer(flt_ptnts, s, timestamp, 'int8')
        filtered = ds.apply_filter(patient_filter, reader)
        filtered = filtered.astype('int8')
        filtered = np.where((filtered - 1) < 0, 0, filtered - 1)
        writer.write(filtered)
        print(f"  {s} processed in {time.time() - t0}s")


    for s in supplement_fields:
        print(s, utils.build_histogram(ds.get_reader(flt_ptnts[s])[:]))

    # build assessment-level filter
    # -----------------------------

    # get a filter for the remaining patients' assessments
    with utils.Timer(f"filtering out orphaned assessments"):
        asmt_not_orphaned_filter =\
            persistence.foreign_key_is_in_primary_key(ds.get_reader(flt_ptnts['id']),
                                                      ds.get_reader(src_asmts['patient_id']))

    symptom_operators = ('persistent_cough', 'skipped_meals', 'loss_of_smell', 'fatigue')
    for s in symptom_operators:
        with utils.Timer(f"flattening and filtering {s}"):
            reader = ds.get_reader(src_asmts[s])
            writer = ds.get_numeric_writer(flt_asmts, s, timestamp, 'uint8')
            writer.write(ds.apply_filter(asmt_not_orphaned_filter, reader[:] > 1))

    # filter assessment_ids
    asmt_core_fields = ('id', 'patient_id', 'created_at', 'updated_at')

    for k in asmt_core_fields:
        with utils.Timer(f"filtering '{k}'"):
            reader = ds.get_reader(src_asmts[k])
            ds.apply_filter(asmt_not_orphaned_filter, reader,
                            reader.get_writer(flt_asmts, k, timestamp))

    with utils.Timer("converting and filtering 'location' -> 'hospitalization' field"):
        asmt_locations = ds.get_reader(src_asmts['location'])
        ds.apply_filter(asmt_not_orphaned_filter, asmt_locations[:] > 1,
                        ds.get_numeric_writer(flt_asmts, 'hospitalization', timestamp, 'uint8'))

    with utils.Timer('flattening treatment field'):
        asmt_treatment = ds.get_reader(src_asmts['treatment'])
        ds.apply_filter(asmt_not_orphaned_filter, asmt_treatment[:] > 1,
                        ds.get_numeric_writer(flt_asmts, 'treatment', timestamp, 'uint8'))

    with utils.Timer('map assessment patient_ids to patient_ids', new_line=True):
        asmt_to_ptnt = ds.get_index(ds.get_reader(flt_ptnts['id']),
                                    ds.get_reader(flt_asmts['patient_id']))

    with utils.Timer('map patient age to assessment space'):
        ptnt_ages = ds.get_reader(flt_ptnts['age'])
        ptnt_ages.get_writer(flt_asmts, 'age', timestamp).write(ptnt_ages[:][asmt_to_ptnt])

    with utils.Timer('map patient gender to assessment gender'):
        ptnt_genders = ds.get_reader(flt_ptnts['gender'])
        ptnt_genders.get_writer(
            flt_asmts, 'gender', timestamp).write(ptnt_genders[:][asmt_to_ptnt])


    with utils.Timer('joining age and gender to filtered assessments'):
        pdf = pd.DataFrame({'id': ds.get_reader(flt_ptnts['id'])[:],
                            'age': ds.get_reader(flt_ptnts['age'])[:],
                            'gender': ds.get_reader(flt_ptnts['gender'])[:]})
        adf = pd.DataFrame({'patient_id': ds.get_reader(flt_asmts['patient_id'])[:]})
        rdf = pd.merge(adf, pdf, left_on='patient_id', right_on='id', how='left')



    print('running covid prediction')
    t0 = time.time()
    scores = nature_medicine_model_1(ds.get_reader(flt_asmts['persistent_cough']),
                                     ds.get_reader(flt_asmts['skipped_meals']),
                                     ds.get_reader(flt_asmts['loss_of_smell']),
                                     ds.get_reader(flt_asmts['fatigue']),
                                     ds.get_reader(flt_asmts['age']),
                                     ds.get_reader(flt_asmts['gender']))
    ds.get_numeric_writer(flt_asmts, 'prediction_score', timestamp, 'float32').write(scores)

    predictions = scores > 0.5
    print(f"covid prediction run in {time.time() - t0}s")
    print(f"covid prediction covid / not covid",
          np.count_nonzero(predictions == 1),
          np.count_nonzero(predictions == 0))
    ds.get_numeric_writer(flt_asmts, 'prediction', timestamp, 'uint8').write(predictions)

    # get the maximum prediction
    with utils.Timer("getting spans for assessment patient_ids"):
        spans = ds.get_spans(ds.get_reader(flt_asmts['patient_id']))

    with utils.Timer("getting max predictions for each patient"):
        max_score_indices = ds.apply_spans_index_of_max(spans, scores)

    print('len(flt_ptnts["id"]:', len(ds.get_reader(flt_ptnts['id'])))
    print('len(max_score_indices):', len(max_score_indices), max_score_indices[:10])


    # apply the indices to the filtered assessments
    for k in flt_asmts.keys():
        with utils.Timer(f"applying selected_assessment indices to {k} "):
            reader = ds.get_reader(flt_asmts[k])
            ds.apply_indices(max_score_indices, reader,
                             reader.get_writer(fnl_asmts, k, timestamp))
            # reader.get_writer(fnl_asmts, k, timestamp).write(reader[:][max_score_indices])

    print(flt_asmts.keys())
    print(fnl_asmts.keys())

    if not np.array_equal(ds.get_reader(flt_ptnts['id'])[:],
                          ds.get_reader(fnl_asmts['patient_id'])[:]):
        raise ValueError("filtered patients should map exactly with final assessments"
                         "but don't!")

    print("exporting")
    fnl_initial_fields = ('id', 'patient_id', 'created_at', 'updated_at')
    ordered_asmt_export_fields = ('age', 'gender', 'fatigue', 'loss_of_smell', 'persistent_cough',
                                  'skipped_meals', 'hospitalization', 'treatment',
                                  'prediction_score', 'prediction')
    fields =\
        [(k, fnl_asmts[k]) for k in fnl_initial_fields] +\
        [(k, flt_ptnts[k]) for k in supplement_fields] +\
        [(k, fnl_asmts[k]) for k in ordered_asmt_export_fields]


    for o in ordered_asmt_export_fields:
        print(o, ds.get_reader(fnl_asmts[o]).dtype())

    exporter.export_to_csv('/home/ben/covid/supplements_assessments.csv', ds, fields)


    # final filter on patients who have any kind of test with a negative / positive result

    tstd_ptnts = dest_data.create_group('tested_patients')
    with utils.Timer('filtering for negative/positive test results'):
        src_matrs = ds.get_reader(src_ptnts['max_assessment_test_result'])
        old_test_filter = src_matrs[:] > 0
        print(utils.build_histogram(src_matrs[:]))

        src_mtrs = ds.get_reader(src_ptnts['max_test_result'])
        new_test_filter = src_mtrs[:] > 0

        tested_patient_filter = patient_filter # & (old_test_filter | new_test_filter)
        print(np.count_nonzero(tested_patient_filter == True),
              np.count_nonzero(tested_patient_filter == False))
        print(utils.build_histogram(src_matrs[:][tested_patient_filter]))
        print(utils.build_histogram(src_mtrs[:][tested_patient_filter]))

        print(utils.build_histogram(src_matrs[:][tested_patient_filter] +
                                    src_mtrs[:][tested_patient_filter]))

    fields_to_export = patient_fields_to_export + ['max_assessment_test_result', 'max_test_result']

    with utils.Timer('filtering patient_fields', new_line=True):
        for k in fields_to_export:
            if k == 'max_assessment_test_result':
                with utils.Timer(f"filtering and flattening {k}"):
                    reader = ds.get_reader(src_ptnts[k])
                    writer = reader.get_writer(tstd_ptnts, 'old_test_result', timestamp)
                    ds.apply_filter(tested_patient_filter, reader, writer)
                    print(utils.build_histogram(ds.get_reader(tstd_ptnts['old_test_result'])[:]))
            elif k == 'max_test_result':
                with utils.Timer(f"filtering and flattening {k}"):
                    reader = ds.get_reader(src_ptnts[k])
                    writer = reader.get_writer(tstd_ptnts, 'new_test_result', timestamp)
                    ds.apply_filter(tested_patient_filter, reader, writer)
                    print(utils.build_histogram(ds.get_reader(tstd_ptnts['new_test_result'])[:]))
            elif k in supplement_fields:
                with utils.Timer(f"filtering and flattening {k}"):
                    reader = ds.get_reader(src_ptnts[k])
                    writer = ds.get_numeric_writer(tstd_ptnts, k, timestamp, 'int8')
                    filtered = ds.apply_filter(tested_patient_filter, reader)
                    filtered = filtered.astype('int8')
                    filtered = np.where((filtered - 1) < 0, 0, filtered - 1)
                    writer.write(filtered)
            else:
                with utils.Timer(f"filtering {k}"):
                    reader = ds.get_reader(src_ptnts[k])
                    writer = reader.get_writer(tstd_ptnts, k, timestamp)
                    ds.apply_filter(tested_patient_filter, reader, writer)

    print(utils.build_histogram(ds.get_reader(tstd_ptnts['old_test_result'])[:]))
    print(utils.build_histogram(ds.get_reader(tstd_ptnts['new_test_result'])[:]))
    print(utils.build_histogram(ds.get_reader(tstd_ptnts['old_test_result'])[:] +
                                ds.get_reader(tstd_ptnts['new_test_result'])[:]))

    asmt_fields = ['id', 'fatigue', 'loss_of_smell', 'persistent_cough', 'skipped_meals',
                   'hospitalization', 'treatment', 'prediction_score', 'prediction']
    patient_fields_plus_results = patient_fields_to_export + ['old_test_result', 'new_test_result']
    fields_to_export = [(k, tstd_ptnts[k]) for k in patient_fields_plus_results] + \
                       [("asmt_{}".format(k), fnl_asmts[k]) for k in asmt_fields]
    print(patient_fields_plus_results + asmt_fields)


    exporter.export_to_csv('/home/ben/covid/supplements_patients.csv', ds, fields_to_export)

    oldt = ds.get_reader(tstd_ptnts['old_test_result'])[:]
    newt = ds.get_reader(tstd_ptnts['new_test_result'])[:]
    print('both:', np.count_nonzero((oldt & newt) == True))


if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20200702_full.hdf5'
    dest_file = '/home/ben/covid/ds_20200702_supplements.hdf5'
    with h5py.File(src_file, 'r+') as src_data:
        with h5py.File(dest_file, 'w') as dest_data:
            supplement_paper_processing(datastore, src_data, dest_data)
