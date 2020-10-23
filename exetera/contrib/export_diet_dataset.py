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
from exetera.processing.healthy_diet_index import healthy_diet_index
from exetera.core import fields


def export_diet_dataset(s, src_data, dest_data):

    src_ptnts = src_data['patients']
    src_diet = src_data['diet']

    ffq_questions = ('ffq_chips', 'ffq_crisps_snacks', 'ffq_eggs', 'ffq_fast_food',
                     'ffq_fibre_rich_breakfast', 'ffq_fizzy_pop', 'ffq_fruit',
                     'ffq_fruit_juice', 'ffq_ice_cream', 'ffq_live_probiotic_fermented',
                     'ffq_oily_fish', 'ffq_pasta', 'ffq_pulses', 'ffq_red_meat',
                     'ffq_red_processed_meat', 'ffq_refined_breakfast', 'ffq_rice',
                     'ffq_salad', 'ffq_sweets', 'ffq_vegetables', 'ffq_white_bread',
                     'ffq_white_fish', 'ffq_white_fish_battered_breaded', 'ffq_white_meat',
                     'ffq_white_processed_meat', 'ffq_wholemeal_bread')

    ffq_dict = {k: s.get(src_diet[k]).data[:] for k in ffq_questions}
    scores = healthy_diet_index(ffq_dict)
    print(np.unique(scores, return_counts=True))

    p_ids = s.get(src_ptnts['id']).data[:]
    d_pids = s.get(src_diet['patient_id']).data[:]

    unique_d_pids = set(d_pids)
    p_filter = np.zeros(len(p_ids), np.bool)
    for i in range(len(p_ids)):
        p_filter[i] = p_ids[i] in unique_d_pids

    print(p_filter.sum(), len(p_filter))
    print(len(s.apply_filter(p_filter, p_ids)))


    # print("= patients =")
    # for k in src_ptnts.keys():
    #     print(k)
    #
    # print("= assessments =")
    # for k in src_data['assessments'].keys():
    #     print(k)
    #
    # print("= tests =")
    # for k in src_data['tests'].keys():
    #     print(k)
    #
    # print("= diet =")
    # for k in src_data['diet'].keys():
    #     print(k)


    patient_fields = ('110_to_220_cm', '15_to_55_bmi', '16_to_90_years', '40_to_200_kg',
                      'a1c_measurement_mmol', 'a1c_measurement_mmol_valid', 'a1c_measurement_percent', 'a1c_measurement_percent_valid',
                      'activity_change', 'age', 'age_filter', 'alcohol_change', 'already_had_covid',
                      # 'always_used_shortage',
                      'assessment_count', 'blood_group', 'bmi', 'bmi_clean', 'bmi_valid', 'cancer_clinical_trial_site',
                      'cancer_type', 'classic_symptoms', 'classic_symptoms_days_ago', 'classic_symptoms_days_ago_valid',
                      'clinical_study_institutions', 'clinical_study_names', 'clinical_study_nct_ids',
                      'contact_additional_studies', 'contact_health_worker', 'country_code', 'created_at',
                      'created_at_day', 'diabetes_diagnosis_year', 'diabetes_diagnosis_year_valid', 'diabetes_oral_biguanide',
                      'diabetes_oral_dpp4', 'diabetes_oral_meglitinides', 'diabetes_oral_other_medication',
                      'diabetes_oral_sglt2', 'diabetes_oral_sulfonylurea', 'diabetes_oral_thiazolidinediones',
                      'diabetes_treatment_basal_insulin', 'diabetes_treatment_insulin_pump', 'diabetes_treatment_lifestyle',
                      'diabetes_treatment_none', 'diabetes_treatment_other_injection', 'diabetes_treatment_other_oral',
                      'diabetes_treatment_pfnts', 'diabetes_treatment_rapid_insulin', 'diabetes_type',
                      # indexed string 'diabetes_type_other',
                      'diabetes_uses_cgm', 'diet_change', 'diet_counts', 'does_chemotherapy', 'ethnicity', 'ever_had_covid_test',
                      'first_assessment_day', 'gender', 'has_asthma', 'has_cancer', 'has_diabetes', 'has_eczema', 'has_hayfever',
                      'has_heart_disease', 'has_imd_data', 'has_kidney_disease', 'has_lung_disease', 'has_lung_disease_only',
                      # 'have_used_PPE', 'have_worked_in_hospital_care_facility', 'have_worked_in_hospital_clinic',
                      # 'have_worked_in_hospital_home_health', 'have_worked_in_hospital_inpatient', 'have_worked_in_hospital_other',
                      # 'have_worked_in_hospital_outpatient', 'have_worked_in_hospital_school_clinic',
                      'health_worker_with_contact',
                      'healthcare_professional', 'height_cm', 'height_cm_clean', 'height_cm_valid', 'help_available',
                      'housebound_problems', 'ht_combined_oral_contraceptive_pill', 'ht_depot_injection_or_implant', 'ht_hormone_treatment_therapy',
                      'ht_mirena_or_other_coil', 'ht_none', 'ht_oestrogen_hormone_therapy', 'ht_pfnts', 'ht_progestone_only_pill',
                      'ht_testosterone_hormone_therapy', 'id', 'imd_decile', 'imd_rank', 'interacted_patients_with_covid',
                      'interacted_with_covid', 'is_carer_for_community',
                      # 'is_in_uk_biobank', 'is_in_uk_guys_trust',
                      # 'is_in_uk_twins', 'is_in_us_agricultural_health', 'is_in_us_american_cancer_society_cancer_prevention_study_3',
                      # 'is_in_us_aspree_xt', 'is_in_us_bwhs', 'is_in_us_c19_human_genetics', 'is_in_us_california_teachers',
                      # 'is_in_us_chasing_covid', 'is_in_us_colocare', 'is_in_us_colon_cancer_family_registry',
                      # 'is_in_us_covid_flu_near_you', 'is_in_us_covid_siren', 'is_in_us_environmental_polymorphisms',
                      # 'is_in_us_growing_up_today', 'is_in_us_gulf', 'is_in_us_harvard_health_professionals', 'is_in_us_hispanic_colorectal_cancer',
                      # 'is_in_us_louisiana_state_university', 'is_in_us_mary_washington_healthcare', 'is_in_us_mass_eye_ear_infirmary',
                      # 'is_in_us_mass_general_brigham', 'is_in_us_md_anderson_d3code', 'is_in_us_multiethnic_cohort',
                      # 'is_in_us_northshore_genomic_health_initiative', 'is_in_us_nurses_study', 'is_in_us_partners_biobank',
                      # 'is_in_us_predetermine', 'is_in_us_predict2', 'is_in_us_promise_pcrowd', 'is_in_us_sister',
                      # 'is_in_us_stanford_diabetes', 'is_in_us_stanford_nutrition', 'is_in_us_stanford_well',
                      'is_pregnant',
                      'is_smoker',
                      # 'ladcd', 'last_asked_level_of_isolation', 'last_asked_level_of_isolation_day', 'last_asked_level_of_isolation_set',
                      'last_assessment_day', 'lifestyle_version', 'limited_activity',
                      'lsoa11cd', 'ruc11cd',
                      # 'lsoa11nm',
                      'max_assessment_test_result',
                      'max_test_result', 'mobility_aid',
                      # 'msoa11cd', 'msoa11nm',
                      'need_inside_help', 'need_outside_help',
                      'needs_help', 'never_used_shortage', 'on_cancer_clinical_trial',
                      # 'outward_postcode', 'outward_postcode_latitude',
                      # 'outward_postcode_latitude_valid', 'outward_postcode_longitude', 'outward_postcode_longitude_valid',
                      # 'outward_postcode_region', 'outward_postcode_town_area',
                      # 'past_symptom_abdominal_pain', 'past_symptom_anosmia',
                      # 'past_symptom_chest_pain', 'past_symptom_delirium', 'past_symptom_diarrhoea', 'past_symptom_fatigue',
                      # 'past_symptom_fever', 'past_symptom_hoarse_voice', 'past_symptom_persistent_cough', 'past_symptom_shortness_of_breath',
                      # 'past_symptom_skipped_meals', 'past_symptoms_changed', 'past_symptoms_days_ago', 'past_symptoms_days_ago_valid',
                      'period_frequency', 'period_status', 'period_stopped_age', 'period_stopped_age_valid', 'pregnant_weeks',
                      'pregnant_weeks_valid',
                      # NOTE: these have been removed from the schema! Handle well going forwards
                      # 'profile_attributes_updated_at', 'profile_attributes_updated_at_day',
                      # 'profile_attributes_updated_at_set',
                      'race_is_other', 'race_is_prefer_not_to_say', 'race_is_uk_asian', 'race_is_uk_black', 'race_is_uk_chinese',
                      'race_is_uk_middle_eastern', 'race_is_uk_mixed_other', 'race_is_uk_mixed_white_black', 'race_is_uk_white',
                      'race_is_us_asian', 'race_is_us_black', 'race_is_us_hawaiian_pacific', 'race_is_us_indian_native', 'race_is_us_white',
                      'race_other', 'reported_by_another', 'same_household_as_reporter', 'se_postcode', 'smoked_years_ago',
                      'smoked_years_ago_valid', 'smoker_status', 'snacking_change', 'sometimes_used_shortage',
                      'still_have_past_symptoms', 'takes_any_blood_pressure_medications', 'takes_aspirin', 'takes_blood_pressure_medications_pril',
                      'takes_blood_pressure_medications_sartan', 'takes_corticosteroids', 'takes_immunosuppressants', 'test_count',
                      # 'unwell_month_before',
                      # 'updated_at', 'updated_at_day', 'version', 'vs_asked_at', 'vs_asked_at_day',
                      'vs_asked_at_set', 'vs_garlic', 'vs_multivitamins', 'vs_none', 'vs_omega_3', 'vs_other', 'vs_pftns',
                      'vs_probiotics', 'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc', 'weight_change', 'weight_change_kg',
                      'weight_change_kg_valid', 'weight_change_pounds', 'weight_change_pounds_valid', 'weight_kg',
                      'weight_kg_clean', 'weight_kg_valid', 'year_of_birth', 'year_of_birth_valid',
                      # 'zipcode'
                      )

    flt_ptnts = dest_data.create_group('patients')
    with utils.Timer("filter patients"):
        for k in patient_fields:
            r = s.get(src_ptnts[k])
            w = r.create_like(flt_ptnts, k)
            s.apply_filter(p_filter, r, w)
            if isinstance(r, fields.IndexedStringField):
                print(len(w.data))

    # print("checking weight / height fields from patients")
    # for k in src_ptnts.keys():
    #     if "weight" in k or "height" in k:
    #         print(k)

    diet_keys = [''] # src_diet.keys()
    p_dict = {'id': s.apply_filter(p_filter, p_ids)}
    for k in flt_ptnts.keys():
        if "weight" in k or "height" in k:
            pkey = "patient_{}".format(k)
        else:
            pkey = k
        p_dict[pkey] = s.get(flt_ptnts[k]).data[:]

    # p_dict.update({k if k not in diet_keys else "patient_{}".format(k): s.get(flt_ptnts[k]).data[:]
    #                for k in patient_fields})
    for k, v in p_dict.items():
        print(k, len(v))
    pdf = pd.DataFrame(p_dict)

    d_dict = {'diet_id': s.get(src_diet['id']).data[:],
              'patient_id': s.get(src_diet['patient_id']).data[:]}
    d_dict.update({
        k: s.get(src_diet[k]).data[:] for k in src_diet.keys() if k not in ('id', 'patient_id')
    })
    d_dict.update({'scores': scores})
    ddf = pd.DataFrame(d_dict)

    tdf = pd.merge(left=ddf, right=pdf, left_on='patient_id', right_on='id')
    for k in tdf.keys():
        if k[-2:] == "_x" or k[-2:] == "_y":
            print(k)

    print(tdf)
    tdf.to_csv('/home/ben/covid/diet_export_20201014.csv', index=False)



if __name__ == '__main__':
    s = Session()
    src_file = '/home/ben/covid/ds_20201014_full.hdf5'
    dest_file = '/home/ben/covid/ds_20201014_diet_export.hdf5'
    with h5py.File(src_file, 'r') as src_data:
        with h5py.File(dest_file, 'w') as dest_data:
            export_diet_dataset(s, src_data, dest_data)
