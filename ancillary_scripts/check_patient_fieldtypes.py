import h5py

import dataset
import data_schemas
import utils

pfilename = '/home/ben/covid/patients_export_geocodes_20200604030001.csv'

# pcatfields  = (
#     'race_is_uk_white', 'need_inside_help', 'on_cancer_clinical_trial', 'race_is_uk_mixed_white_black',
#     'always_used_shortage', 'sometimes_used_shortage', 'race_is_uk_asian', 'is_in_uk_twins',
#     'have_worked_in_hospital_other', 'is_in_us_aspree_xt', 'help_available', 'have_worked_in_hospital_outpatient',
#     'has_cancer', 'past_symptom_hoarse_voice', 'reported_by_another', 'is_in_us_colocare', 'interacted_with_covid',
#     'race_is_other', 'period_frequency', 'have_used_PPE', 'is_in_us_predetermine', 'takes_aspirin',
#     'past_symptom_anosmia', 'past_symptom_delirium', 'same_household_as_reporter', 'ht_oestrogen_hormone_therapy',
#     'past_symptoms_changed', 'past_symptom_chest_pain', 'need_outside_help', 'is_in_us_bwhs', 'is_in_us_predict2',
#     'gender', 'have_worked_in_hospital_school_clinic', 'is_in_us_growing_up_today', 'race_is_uk_black',
#     'race_is_uk_chinese', 'race_is_uk_mixed_other', 'race_is_us_black', 'classic_symptoms', 'is_in_us_covid_siren',
#     'is_carer_for_community', 'is_in_uk_guys_trust', 'is_in_us_stanford_well', 'is_in_us_multiethnic_cohort',
#     'takes_blood_pressure_medications_pril', 'race_is_prefer_not_to_say', 'have_worked_in_hospital_inpatient',
#     'is_in_us_agricultural_health', 'interacted_patients_with_covid', 'past_symptom_fatigue',
#     'is_in_us_stanford_nutrition', 'race_is_us_white', 'is_in_us_gulf', 'ht_testosterone_hormone_therapy',
#     'unwell_month_before', 'has_lung_disease', 'housebound_problems',
#     'is_in_us_american_cancer_society_cancer_prevention_study_3', 'ht_depot_injection_or_implant',
#     'takes_corticosteroids', 'past_symptom_shortness_of_breath', 'race_is_us_hawaiian_pacific', 'has_heart_disease',
#     'is_in_us_harvard_health_professionals', 'is_in_us_colon_cancer_family_registry', 'is_in_us_nurses_study',
#     'ht_pfnts', 'is_in_us_chasing_covid', 'period_status', 'never_used_shortage', 'past_symptom_persistent_cough',
#     'already_had_covid', 'ht_progestone_only_pill', 'is_in_us_mass_eye_ear_infirmary',
#     'is_in_us_northshore_genomic_health_initiative', 'past_symptom_diarrhoea', 'past_symptom_fever',
#     'takes_immunosuppressants', 'is_in_us_promise_pcrowd', 'is_in_uk_biobank',
#     'have_worked_in_hospital_care_facility', 'limited_activity', 'is_pregnant', 'needs_help', 'ever_had_covid_test',
#     'have_worked_in_hospital_clinic', 'ht_none', 'contact_health_worker', 'smoker_status', 'has_kidney_disease',
#     'is_in_us_stanford_diabetes', 'race_is_us_indian_native', 'past_symptom_abdominal_pain',
#     'past_symptom_skipped_meals', 'race_is_uk_middle_eastern', 'is_in_us_sister', 'is_in_us_partners_biobank',
#     'does_chemotherapy', 'ht_hormone_treatment_therapy', 'healthcare_professional', 'race_is_us_asian',
#     'is_in_us_md_anderson_d3code', 'is_in_us_mass_general_brigham', 'is_in_us_louisiana_state_university',
#     'ethnicity', 'still_have_past_symptoms', 'takes_blood_pressure_medications_sartan',
#     'is_in_us_hispanic_colorectal_cancer', 'ht_combined_oral_contraceptive_pill', 'is_in_us_california_teachers',
#     'is_in_us_environmental_polymorphisms', 'is_smoker', 'is_in_us_covid_flu_near_you', 'mobility_aid',
#     'have_worked_in_hospital_home_health', 'takes_any_blood_pressure_medications', 'ht_mirena_or_other_coil',
#     'has_diabetes')
pcatfields = (
    'vs_vitamin_d', 'vs_other', 'vs_omega_3', 'vs_none', 'vs_vitamin_c', 'vs_pftns', 'vs_multivitamins',
    'vs_garlic', 'vs_probiotics', 'vs_zinc', 'vs_asked_at')
data_schema = data_schemas.DataSchema(1)
with open(pfilename) as f:
    ds = dataset.Dataset(f, keys=pcatfields, show_progress_every=100000)

    for n in ds.names_:
        if data_schema.patient_field_types.get(n, None) == 'categoricaltype':
            print(n)
            h = utils.build_histogram(ds.field_by_name(n))
            if len(h) > 100:
                print('not categorical!', h[:10])
            else:
                print(h)

