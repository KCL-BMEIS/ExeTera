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

import numpy as np

from hystore.core import persistence as pers
from hystore.core import readerwriter as rw
from hystore.core import fields


class FieldDesc:
    def __init__(self, field, strings_to_values, values_to_strings, to_datatype,
                 out_of_range_label):
        self.field = field
        self.to_datatype = to_datatype
        self.strings_to_values = strings_to_values
        self.values_to_strings = values_to_strings
        self.out_of_range_label = out_of_range_label

    def __str__(self):
        output = 'FieldDesc(field={}, strings_to_values={}, values_to_strings={})'
        return output.format(self.field, self.strings_to_values, self.values_to_strings)

    def __repr__(self):
        return self.__str__()


class FieldEntry:
    def __init__(self, field_desc, version_from, version_to=None):
        self.field_desc = field_desc
        self.version_from = version_from
        self.version_to = version_to

    def __str__(self):
        output = 'FieldEntry(field_desc={}, version_from={}, version_to={})'
        return output.format(self.field_desc, self.version_from, self.version_to)

    def __repr__(self):
        return self.__str__()


class DataSchemaVersionError(Exception):
    pass


def _build_map(value_list):
    inverse = dict()
    for ir, r in enumerate(value_list):
        inverse[r] = ir
    return inverse


class DataSchema:
    data_schemas = [1]


    field_writers = {
        'idtype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 32, ts),
        'datetimetype':
            lambda g, cs, n, ts: rw.DateTimeImporter(g, cs, n, False, ts),
        'optionaldatetimetype':
            lambda g, cs, n, ts: rw.DateTimeImporter(g, cs, n, True, ts),
        'datetype':
            lambda g, cs, n, ts: rw.OptionalDateImporter(g, cs, n, False, ts),
        'optionaldatetype':
            lambda g, cs, n, ts: rw.OptionalDateImporter(g, cs, n, True, ts),
        'versiontype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 10, ts),
        'indexedstringtype': lambda g, cs, n, ts: rw.IndexedStringWriter(g, cs, n, ts),
        'countrycodetype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 2, ts),
        'unittype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 1, ts),
        'categoricaltype':
            lambda g, cs, n, stv, ts: rw.CategoricalWriter(g, cs, n, stv, ts),
        'leakycategoricaltype':
            lambda g, cs, n, stv, oor, ts: rw.LeakyCategoricalImporter(g, cs, n, stv,
                                                                                oor, ts),
        'float32type': lambda g, cs, n, ts: rw.NumericImporter(
            g, cs, n, 'float32', pers.try_str_to_float, ts),
        'uint16type': lambda g, cs, n, ts: rw.NumericImporter(
            g, cs, n, 'uint16', pers.try_str_to_int, ts),
        'yeartype': lambda g, cs, n, ts: rw.NumericImporter(
            g, cs, n, 'uint32', pers.try_str_to_float_to_int, ts),
        'geocodetype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 9, ts)
    }


    new_field_writers = {
        'idtype': lambda s, g, n, ts=None, cs=None:
            fields.FixedStringImporter(s, g, n, 32, ts, cs),
        'datetimetype': lambda s, g, n, ts=None, cs=None:
            fields.DateTimeImporter(s, g, n, False, True, ts, cs),
        'optionaldatetimetype': lambda s, g, n, ts=None, cs=None:
            fields.DateTimeImporter(s, g, n, True, True, ts, cs),
        'datetype': lambda s, g, n, ts=None, cs=None:
            fields.DateImporter(s, g, n, False, ts, cs),
        'optionaldatetype': lambda s, g, n, ts=None, cs=None:
            fields.DateImporter(s, g, n, True, ts, cs),
        'versiontype': lambda s, g, n, ts=None, cs=None:
            fields.FixedStringImporter(s, g, n, 10, ts, cs),
        'indexedstringtype': lambda s, g, n, ts=None, cs=None:
            fields.IndexedStringImporter(s, g, n, ts, cs),
        'countrycodetype': lambda s, g, n, ts=None, cs=None:
            fields.FixedStringImporter(s, g, n, 2, ts, cs),
        'unittype': lambda s, g, n, ts=None, cs=None:
            fields.FixedStringImporter(s, g, n, 1, ts, cs),
        'categoricaltype': lambda s, g, n, vt, stv, ts=None, cs=None:
            fields.CategoricalImporter(s, g, n, vt, stv, ts, cs),
        'leakycategoricaltype': lambda s, g, n, vt, stv, oor, ts=None, cs=None:
            fields.LeakyCategoricalImporter(s, g, n, vt, stv, oor, ts, cs),
        'float32type': lambda s, g, n, ts=None, cs=None:
            fields.NumericImporter(s, g, n, 'float32', pers.try_str_to_float, ts, cs),
        'uint16type': lambda s, g, n, ts=None, cs=None:
            fields.NumericImporter(s, g, n, 'uint16', pers.try_str_to_int, ts, cs),
        'yeartype': lambda s, g, n, ts=None, cs=None:
            fields.NumericImporter(s, g, n, 'uint32', pers.try_str_to_float_to_int, ts, cs),
        'geocodetype': lambda s, g, n, ts=None, cs=None:
            fields.FixedStringImporter(s, g, n, 9, ts, cs)
    }


    _patient_field_types = {
        'id': 'idtype',
        'created_at': 'datetimetype',
        'updated_at': 'datetimetype',
        'version': 'versiontype',
        'country_code': 'countrycodetype',
        'a1c_measurement_mmol': 'float32type',
        'a1c_measurement_percent': 'float32type',
        'activity_change': 'categoricaltype',
        'alcohol_change': 'categoricaltype',
        'already_had_covid': 'categoricaltype',
        'always_used_shortage': 'categoricaltype',
        'blood_group': 'categoricaltype',
        'bmi': 'float32type',
        'cancer_clinical_trial_site': 'indexedstringtype',
        'cancer_type': 'indexedstringtype',
        'classic_symptoms': 'categoricaltype',
        'classic_symptoms_days_ago': 'uint16type',
        'clinical_study_institutions': 'indexedstringtype',
        'clinical_study_names': 'indexedstringtype',
        'clinical_study_nct_ids': 'indexedstringtype',
        'contact_additional_studies': 'categoricaltype',
        'contact_health_worker': 'categoricaltype',
        'diabetes_diagnosis_year': 'uint16type',
        'diabetes_oral_biguanide': 'categoricaltype',
        'diabetes_oral_dpp4': 'categoricaltype',
        'diabetes_oral_meglitinides': 'categoricaltype',
        'diabetes_oral_other_medication': 'indexedstringtype',
        'diabetes_oral_sglt2': 'categoricaltype',
        'diabetes_oral_sulfonylurea': 'categoricaltype',
        'diabetes_oral_thiazolidinediones': 'categoricaltype',
        'diabetes_treatment_basal_insulin': 'categoricaltype',
        'diabetes_treatment_insulin_pump': 'categoricaltype',
        'diabetes_treatment_lifestyle': 'categoricaltype',
        'diabetes_treatment_none': 'categoricaltype',
        'diabetes_treatment_other_injection': 'indexedstringtype',
        'diabetes_treatment_other_oral': 'indexedstringtype',
        'diabetes_treatment_pfnts': 'categoricaltype',
        'diabetes_treatment_rapid_insulin': 'categoricaltype',
        'diabetes_type': 'categoricaltype',
        'diabetes_type_other': 'indexedstringtype',
        'diabetes_uses_cgm': 'categoricaltype',
        'diet_change': 'categoricaltype',
        'does_chemotherapy': 'categoricaltype',
        'ethnicity': 'categoricaltype',
        'ever_had_covid_test': 'categoricaltype',
        'gender': 'categoricaltype',
        'has_asthma': 'categoricaltype',
        'has_cancer': 'categoricaltype',
        'has_diabetes': 'categoricaltype',
        'has_eczema': 'categoricaltype',
        'has_hayfever': 'categoricaltype',
        'has_heart_disease': 'categoricaltype',
        'has_kidney_disease': 'categoricaltype',
        'has_lung_disease': 'categoricaltype',
        'has_lung_disease_only': 'categoricaltype',
        'have_used_PPE': 'categoricaltype',
        'have_worked_in_hospital_care_facility': 'categoricaltype',
        'have_worked_in_hospital_clinic': 'categoricaltype',
        'have_worked_in_hospital_home_health': 'categoricaltype',
        'have_worked_in_hospital_inpatient': 'categoricaltype',
        'have_worked_in_hospital_other': 'categoricaltype',
        'have_worked_in_hospital_outpatient': 'categoricaltype',
        'have_worked_in_hospital_school_clinic': 'categoricaltype',
        'healthcare_professional': 'categoricaltype',
        'height_cm': 'float32type',
        'help_available': 'categoricaltype',
        'housebound_problems': 'categoricaltype',
        'ht_combined_oral_contraceptive_pill': 'categoricaltype',
        'ht_depot_injection_or_implant': 'categoricaltype',
        'ht_hormone_treatment_therapy': 'categoricaltype',
        'ht_mirena_or_other_coil': 'categoricaltype',
        'ht_none': 'categoricaltype',
        'ht_oestrogen_hormone_therapy': 'categoricaltype',
        'ht_pfnts': 'categoricaltype',
        'ht_progestone_only_pill': 'categoricaltype',
        'ht_testosterone_hormone_therapy': 'categoricaltype',
        'interacted_patients_with_covid': 'categoricaltype',
        'interacted_with_covid': 'categoricaltype',
        'is_carer_for_community': 'categoricaltype',
        'is_in_uk_biobank': 'categoricaltype',
        'is_in_uk_guys_trust': 'categoricaltype',
        'is_in_uk_twins': 'categoricaltype',
        'is_in_us_agricultural_health': 'categoricaltype',
        'is_in_us_american_cancer_society_cancer_prevention_study_3': 'categoricaltype',
        'is_in_us_aspree_xt': 'categoricaltype',
        'is_in_us_bwhs': 'categoricaltype',
        'is_in_us_c19_human_genetics': 'categoricaltype',
        'is_in_us_california_teachers': 'categoricaltype',
        'is_in_us_chasing_covid': 'categoricaltype',
        'is_in_us_colocare': 'categoricaltype',
        'is_in_us_colon_cancer_family_registry': 'categoricaltype',
        'is_in_us_covid_flu_near_you': 'categoricaltype',
        'is_in_us_covid_siren': 'categoricaltype',
        'is_in_us_environmental_polymorphisms': 'categoricaltype',
        'is_in_us_growing_up_today': 'categoricaltype',
        'is_in_us_gulf': 'categoricaltype',
        'is_in_us_harvard_health_professionals': 'categoricaltype',
        'is_in_us_hispanic_colorectal_cancer': 'categoricaltype',
        'is_in_us_louisiana_state_university': 'categoricaltype',
        'is_in_us_mass_eye_ear_infirmary': 'categoricaltype',
        'is_in_us_mass_general_brigham': 'categoricaltype',
        'is_in_us_md_anderson_d3code': 'categoricaltype',
        'is_in_us_multiethnic_cohort': 'categoricaltype',
        'is_in_us_northshore_genomic_health_initiative': 'categoricaltype',
        'is_in_us_nurses_study': 'categoricaltype',
        'is_in_us_partners_biobank': 'categoricaltype',
        'is_in_us_predetermine': 'categoricaltype',
        'is_in_us_predict2': 'categoricaltype',
        'is_in_us_promise_pcrowd': 'categoricaltype',
        'is_in_us_sister': 'categoricaltype',
        'is_in_us_stanford_diabetes': 'categoricaltype',
        'is_in_us_stanford_nutrition': 'categoricaltype',
        'is_in_us_stanford_well': 'categoricaltype',
        'is_pregnant': 'categoricaltype',
        'is_smoker': 'categoricaltype',
        'ladcd': 'geocodetype',
        'last_asked_level_of_isolation': 'optionaldatetimetype',
        'lifestyle_version': 'versiontype',
        'limited_activity': 'categoricaltype',
        'lsoa11cd': 'geocodetype',
        'lsoa11nm': 'indexedstringtype',
        'mobility_aid': 'categoricaltype',
        'msoa11cd': 'geocodetype',
        'msoa11nm': 'indexedstringtype',
        'need_inside_help': 'categoricaltype',
        'need_outside_help': 'categoricaltype',
        'needs_help': 'categoricaltype',
        'never_used_shortage': 'categoricaltype',
        'on_cancer_clinical_trial': 'categoricaltype',
        'outward_postcode': 'indexedstringtype',
        'outward_postcode_latitude': 'float32type',
        'outward_postcode_longitude': 'float32type',
        'outward_postcode_region': 'indexedstringtype',
        'outward_postcode_town_area': 'indexedstringtype',
        'past_symptom_abdominal_pain': 'categoricaltype',
        'past_symptom_anosmia': 'categoricaltype',
        'past_symptom_chest_pain': 'categoricaltype',
        'past_symptom_delirium': 'categoricaltype',
        'past_symptom_diarrhoea': 'categoricaltype',
        'past_symptom_fatigue': 'categoricaltype',
        'past_symptom_fever': 'categoricaltype',
        'past_symptom_hoarse_voice': 'categoricaltype',
        'past_symptom_persistent_cough': 'categoricaltype',
        'past_symptom_shortness_of_breath': 'categoricaltype',
        'past_symptom_skipped_meals': 'categoricaltype',
        'past_symptoms_changed': 'categoricaltype',
        'past_symptoms_days_ago': 'uint16type',
        'period_frequency': 'categoricaltype',
        'period_status': 'categoricaltype',
        'period_stopped_age': 'uint16type',
        'pregnant_weeks': 'uint16type',
        'profile_attributes_updated_at': 'optionaldatetimetype',
        'race_is_other': 'categoricaltype',
        'race_is_prefer_not_to_say': 'categoricaltype',
        'race_is_uk_asian': 'categoricaltype',
        'race_is_uk_black': 'categoricaltype',
        'race_is_uk_chinese': 'categoricaltype',
        'race_is_uk_middle_eastern': 'categoricaltype',
        'race_is_uk_mixed_other': 'categoricaltype',
        'race_is_uk_mixed_white_black': 'categoricaltype',
        'race_is_uk_white': 'categoricaltype',
        'race_is_us_asian': 'categoricaltype',
        'race_is_us_black': 'categoricaltype',
        'race_is_us_hawaiian_pacific': 'categoricaltype',
        'race_is_us_indian_native': 'categoricaltype',
        'race_is_us_white': 'categoricaltype',
        'race_other': 'indexedstringtype',
        'reported_by_another': 'categoricaltype',
        'same_household_as_reporter': 'categoricaltype',
        'se_postcode': 'indexedstringtype',
        'smoked_years_ago': 'uint16type',
        'smoker_status': 'categoricaltype',
        'snacking_change': 'categoricaltype',
        'sometimes_used_shortage': 'categoricaltype',
        'still_have_past_symptoms': 'categoricaltype',
        'takes_any_blood_pressure_medications': 'categoricaltype',
        'takes_aspirin': 'categoricaltype',
        'takes_blood_pressure_medications_pril': 'categoricaltype',
        'takes_blood_pressure_medications_sartan': 'categoricaltype',
        'takes_corticosteroids': 'categoricaltype',
        'takes_immunosuppressants': 'categoricaltype',
        'unwell_month_before': 'categoricaltype',
        'vs_asked_at': 'optionaldatetimetype',
        'vs_garlic': 'categoricaltype',
        'vs_multivitamins': 'categoricaltype',
        'vs_none': 'categoricaltype',
        'vs_omega_3': 'categoricaltype',
        'vs_other': 'indexedstringtype',
        'vs_pftns': 'categoricaltype',
        'vs_probiotics': 'categoricaltype',
        'vs_vitamin_c': 'categoricaltype',
        'vs_vitamin_d': 'categoricaltype',
        'vs_zinc': 'categoricaltype',
        'weight_change': 'categoricaltype',
        'weight_change_kg': 'float32type',
        'weight_change_pounds': 'float32type',
        'weight_kg': 'float32type',
        'year_of_birth': 'yeartype',
        'zipcode': 'indexedstringtype'
    }

    _assessment_field_types = {
        'id': 'idtype',
        'patient_id': 'idtype',
        'created_at': 'datetimetype',
        'updated_at': 'datetimetype',
        'version': 'versiontype',
        'country_code': 'countrycodetype',
        'abdominal_pain': 'categoricaltype',
        'always_used_shortage': 'categoricaltype',
        'blisters_on_feet': 'categoricaltype',
        'chest_pain': 'categoricaltype',
        'chills_or_shivers': 'categoricaltype',
        'date_test_occurred': 'optionaldatetype',
        'date_test_occurred_guess': 'categoricaltype',
        'delirium': 'categoricaltype',
        'diarrhoea': 'categoricaltype',
        'diarrhoea_frequency': 'categoricaltype',
        'dizzy_light_headed': 'categoricaltype',
        'eye_soreness': 'categoricaltype',
        'fatigue': 'categoricaltype',
        'fever': 'categoricaltype',
        'had_covid_test': 'categoricaltype',
        'have_used_PPE': 'categoricaltype',
        'headache': 'categoricaltype',
        'headache_frequency': 'categoricaltype',
        'health_status': 'categoricaltype',
        'hoarse_voice': 'categoricaltype',
        'interacted_any_patients': 'categoricaltype',
        'isolation_healthcare_provider': 'uint16type',
        'isolation_little_interaction': 'uint16type',
        'isolation_lots_of_people': 'uint16type',
        'level_of_isolation': 'categoricaltype',
        'location': 'categoricaltype',
        'loss_of_smell': 'categoricaltype',
        'mask_cloth_or_scarf': 'categoricaltype',
        'mask_n95_ffp': 'categoricaltype',
        'mask_not_sure_pfnts': 'categoricaltype',
        'mask_other': 'indexedstringtype',
        'mask_surgical': 'categoricaltype',
        'nausea': 'categoricaltype',
        'never_used_shortage': 'categoricaltype',
        'other_symptoms': 'indexedstringtype',
        'persistent_cough': 'categoricaltype',
        'red_welts_on_face_or_lips': 'categoricaltype',
        'shortness_of_breath': 'categoricaltype',
        'skipped_meals': 'categoricaltype',
        'sometimes_used_shortage': 'categoricaltype',
        'sore_throat': 'categoricaltype',
        'temperature': 'float32type',
        'temperature_unit': 'unittype',
        'tested_covid_positive': 'categoricaltype',
        'treated_patients_with_covid': 'categoricaltype',
        'treatment': 'leakycategoricaltype',
        'typical_hayfever': 'categoricaltype',
        'unusual_muscle_pains': 'categoricaltype',
        'worn_face_mask': 'categoricaltype'
    }

    _health_check_fields = (
        'fever', 'persistent_cough', 'fatigue', 'shortness_of_breath', 'diarrhoea',
        'delirium', 'skipped_meals', 'abdominal_pain', 'chest_pain', 'hoarse_voice',
        'loss_of_smell', 'headache', 'chills_or_shivers', 'eye_soreness', 'nausea',
        'dizzy_light_headed', 'red_welts_on_face_or_lips', 'blisters_on_feet', 'sore_throat',
        'unusual_muscle_pains'
    )

    _test_field_types = {
        'id': 'idtype',
        'patient_id': 'idtype',
        'created_at': 'datetimetype',
        'updated_at': 'datetimetype',
        'version': 'versiontype',
        'country_code': 'countrycodetype',
        'date_taken_between_end': 'datetype',
        'date_taken_between_start': 'datetype',
        'date_taken_specific': 'datetype',
        'deleted': 'categoricaltype',
        'invited_to_test': 'categoricaltype',
        'location': 'categoricaltype',
        'location_other': 'indexedstringtype',
        'mechanism': 'leakycategoricaltype',
        'result': 'categoricaltype',
        'trained_worker': 'categoricaltype',
    }

    na_value_from = ''
    na_value_to = ''
    leaky_boolean_to = [na_value_to, 'False', 'True']
    leaky_boolean_from = _build_map(leaky_boolean_to)
    leaky_boolean_delta = [na_value_to, 'pfnts', 'decreased', 'same', 'increased']
    # tuple entries
    # 0: name
    # 1: values_to_strings,
    # 2: string_to_values or None if it should be calculated from values_to_string
    # 3: inclusive version from
    # 4: exclusive version to
    _patient_categorical_fields = [
        ('activity_change', leaky_boolean_delta, None, np.uint8, 1, None),
        ('alcohol_change', leaky_boolean_delta + ['no_alcohol'], None, np.uint8, 1, None),
        ('already_had_covid', leaky_boolean_to, None, np.uint8, 1, None),
        ('always_used_shortage', ['', 'all_needed', 'reused'], None, np.uint8, 1, None),
        ('blood_group', ['', 'a', 'b', 'ab', 'o', 'unsure', 'pfnts'], None, np.uint8, 1, None),
        ('classic_symptoms', leaky_boolean_to, None, np.uint8, 1, None),
        ('contact_additional_studies', leaky_boolean_to, None, np.uint8, 1, None),
        ('contact_health_worker', leaky_boolean_to, None, np.uint8, 1, None),
        ('does_chemotherapy', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_oral_biguanide', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_oral_dpp4', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_oral_meglitinides', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_oral_sglt2', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_oral_sulfonylurea', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_oral_thiazolidinediones', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_treatment_basal_insulin', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_treatment_insulin_pump', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_treatment_lifestyle', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_treatment_none', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_treatment_pfnts', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_treatment_rapid_insulin', leaky_boolean_to, None, np.uint8, 1, None),
        ('diabetes_type', [na_value_to, 'pfnts', 'gestational', 'type_1', 'type_2', 'unsure', 'other'], None, np.uint8, 1, None),
        ('diabetes_uses_cgm', leaky_boolean_to, None, np.uint8, 1, None),
        ('diet_change', leaky_boolean_delta, None, np.uint8, 1, None),
        ('ethnicity', [na_value_to, 'prefer_not_to_say', 'not_hispanic', 'hispanic'], None, np.uint8, 1, None),
        ('gender', ['', '0', '1', '2', '3', '99999'], None, np.uint32, 1, None),
        ('ever_had_covid_test', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_asthma', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_cancer', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_diabetes', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_eczema', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_hayfever', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_heart_disease', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_kidney_disease', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_lung_disease', leaky_boolean_to, None, np.uint8, 1, None),
        ('has_lung_disease_only', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_used_PPE', ['', 'never', 'sometimes', 'always'], None, np.uint8, 1, None),
        ('have_worked_in_hospital_care_facility', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_worked_in_hospital_clinic', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_worked_in_hospital_home_health', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_worked_in_hospital_inpatient', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_worked_in_hospital_other', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_worked_in_hospital_outpatient', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_worked_in_hospital_school_clinic', leaky_boolean_to, None, np.uint8, 1, None),
        ('healthcare_professional', [na_value_to, 'no', 'yes_does_not_interact', 'yes_does_not_treat', 'yes_does_interact', 'yes_does_treat'], None, np.uint8, 1, None),
        ('help_available', leaky_boolean_to, None, np.uint8, 1, None),
        ('housebound_problems', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_combined_oral_contraceptive_pill', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_depot_injection_or_implant', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_hormone_treatment_therapy', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_mirena_or_other_coil', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_none', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_oestrogen_hormone_therapy', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_pfnts', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_progestone_only_pill', leaky_boolean_to, None, np.uint8, 1, None),
        ('ht_testosterone_hormone_therapy', leaky_boolean_to, None, np.uint8, 1, None),
        ('interacted_patients_with_covid', ['', 'no', 'yes_suspected', 'yes_documented_suspected', 'yes_documented'], None, np.uint8, 1, None),
        ('interacted_with_covid', ['', 'no', 'yes_suspected', 'yes_documented_suspected', 'yes_documented'], None, np.uint8, 1, None),
        ('is_carer_for_community', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_uk_biobank', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_uk_guys_trust', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_uk_twins', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_agricultural_health', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_american_cancer_society_cancer_prevention_study_3', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_aspree_xt', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_bwhs', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_c19_human_genetics', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_california_teachers', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_chasing_covid', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_colocare', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_colon_cancer_family_registry', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_covid_flu_near_you', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_covid_siren', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_environmental_polymorphisms', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_growing_up_today', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_gulf', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_harvard_health_professionals', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_hispanic_colorectal_cancer', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_louisiana_state_university', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_mass_eye_ear_infirmary', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_mass_general_brigham', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_md_anderson_d3code', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_multiethnic_cohort', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_northshore_genomic_health_initiative', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_nurses_study', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_partners_biobank', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_predetermine', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_predict2', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_promise_pcrowd', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_sister', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_stanford_diabetes', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_stanford_nutrition', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_in_us_stanford_well', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_pregnant', leaky_boolean_to, None, np.uint8, 1, None),
        ('is_smoker', leaky_boolean_to, None, np.uint8, 1, None),
        ('limited_activity', leaky_boolean_to, None, np.uint8, 1, None),
        ('mobility_aid', leaky_boolean_to, None, np.uint8, 1, None),
        ('need_inside_help', leaky_boolean_to, None, np.uint8, 1, None),
        ('need_outside_help', leaky_boolean_to, None, np.uint8, 1, None),
        ('needs_help', leaky_boolean_to, None, np.uint8, 1, None),
        ('never_used_shortage', ['', 'not_available', 'not_needed'], None, np.uint8, 1, None),
        ('on_cancer_clinical_trial', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_abdominal_pain', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_anosmia', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_chest_pain', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_delirium', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_diarrhoea', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_fatigue', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_fever', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_hoarse_voice', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_persistent_cough', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_shortness_of_breath', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptom_skipped_meals', leaky_boolean_to, None, np.uint8, 1, None),
        ('past_symptoms_changed', ['', 'much_better', 'little_better', 'same', 'little_worse', 'much_worse'], None, np.uint8, 1, None),
        ('period_frequency', ['', 'less_frequent', 'irregular', 'regular'], None, np.uint8, 1, None),
        ('period_status', ['', 'other', 'pfnts', 'never', 'not_currently', 'currently', 'pregnant', 'stopped'], None, np.uint8, 1, None),
        ('race_is_other', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_prefer_not_to_say', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_uk_asian', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_uk_black', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_uk_chinese', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_uk_middle_eastern', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_uk_mixed_other', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_uk_mixed_white_black', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_uk_white', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_us_asian', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_us_black', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_us_hawaiian_pacific', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_us_indian_native', ['False', 'True'], None, np.uint8, 1, None),
        ('race_is_us_white', ['False', 'True'], None, np.uint8, 1, None),
        ('reported_by_another', ['False', 'True'], None, np.uint8, 1, None),
        ('same_household_as_reporter', leaky_boolean_to, None, np.uint8, 1, None),
        ('smoker_status', ['', 'never', 'not_currently', 'yes'], None, np.uint8, 1, None),
        ('snacking_change', leaky_boolean_delta, None, np.uint8, 1, None),
        ('sometimes_used_shortage', ['', 'all_needed', 'reused', 'not_enough'], None, np.uint8, 1, None),
        ('still_have_past_symptoms', leaky_boolean_to, None, np.uint8, 1, None),
        ('takes_any_blood_pressure_medications', leaky_boolean_to, None, np.uint8, 1, None),
        ('takes_aspirin', leaky_boolean_to, None, np.uint8, 1, None),
        ('takes_blood_pressure_medications_pril', leaky_boolean_to, None, np.uint8, 1, None),
        ('takes_blood_pressure_medications_sartan', leaky_boolean_to, None, np.uint8, 1, None),
        ('takes_corticosteroids', leaky_boolean_to, None, np.uint8, 1, None),
        ('takes_immunosuppressants',  leaky_boolean_to, None, np.uint8, 1, None),
        ('unwell_month_before', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_garlic', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_multivitamins', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_none', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_omega_3', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_pftns', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_probiotics', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_vitamin_c', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_vitamin_d', leaky_boolean_to, None, np.uint8, 1, None),
        ('vs_zinc', leaky_boolean_to, None, np.uint8, 1, None),
        ('weight_change', leaky_boolean_delta, None, np.uint8, 1, None),
    ]
    _assessment_categorical_fields = [
        ('abdominal_pain', leaky_boolean_to, None, np.uint8, 1, None),
        ('always_used_shortage', [na_value_to, 'all_needed', 'reused'], None, np.uint8, 1, None),
        ('blisters_on_feet', leaky_boolean_to, None, np.uint8, 1, None),
        ('chest_pain', leaky_boolean_to, None, np.uint8, 1, None),
        ('chills_or_shivers', leaky_boolean_to, None, np.uint8, 1, None),
        ('date_test_occurred_guess', [na_value_to, 'less_than_7_days_ago', 'over_1_week_ago', 'over_2_week_ago', 'over_3_week_ago', 'over_1_month_ago'], None, np.uint8, 1, None),
        ('delirium', leaky_boolean_to, None, np.uint8, 1, None),
        ('diarrhoea', leaky_boolean_to, None, np.uint8, 1, None),
        ('diarrhoea_frequency', [na_value_to, 'one_to_two', 'three_to_four', 'five_or_more'], None, np.uint8, 1, None),
        ('dizzy_light_headed', leaky_boolean_to, None, np.uint8, 1, None),
        ('eye_soreness', leaky_boolean_to, None, np.uint8, 1, None),
        ('fatigue', [na_value_to, 'no', 'mild', 'significant', 'severe'], None, np.uint8, 1, None),
        ('fatigue_binary', leaky_boolean_to, {na_value_from: 0, 'no': 1, 'mild': 2, 'significant': 2, 'severe': 2}, np.uint8, 1, None),
        ('fever', leaky_boolean_to, None, np.uint8, 1, None),
        ('had_covid_test', leaky_boolean_to, None, np.uint8, 1, None),
        ('had_covid_test_clean', leaky_boolean_to, None, np.uint8, 1, None),
        ('have_used_PPE', [na_value_to, 'never', 'sometimes', 'always'], None, np.uint8, 1, None),
        ('headache', leaky_boolean_to, None, np.uint8, 1, None),
        ('headache_frequency', [na_value_to, 'some_of_day', 'most_of_day', 'all_of_the_day'], None, np.uint8, 1, None),
        ('health_status', [na_value_to, 'healthy', 'not_healthy'], None, np.uint8, 1, None),
        ('hoarse_voice', leaky_boolean_to, None, np.uint8, 1, None),
        ('interacted_any_patients', leaky_boolean_to, None, np.uint8, 1, None),
        ('level_of_isolation', [na_value_to, 'not_left_the_house', 'rarely_left_the_house', 'rarely_left_the_house_but_visited_lots', 'often_left_the_house'], None, np.uint8, 1, None),
        ('location', [na_value_to, 'home', 'hospital', 'back_from_hospital'], None, np.uint8, 1, None),
        ('loss_of_smell', leaky_boolean_to, None, np.uint8, 1, None),
        ('mask_cloth_or_scarf', leaky_boolean_to, None, np.uint8, 1, None),
        ('mask_n95_ffp', leaky_boolean_to, None, np.uint8, 1, None),
        ('mask_not_sure_pfnts', leaky_boolean_to, None, np.uint8, 1, None),
        ('mask_surgical', leaky_boolean_to, None, np.uint8, 1, None),
        ('nausea', leaky_boolean_to, None, np.uint8, 1, None),
        ('never_used_shortage', [na_value_to, 'not_needed', 'not_available'], None, np.uint8, 1, None),
        ('persistent_cough', leaky_boolean_to, None, np.uint8, 1, None),
        ('red_welts_on_face_or_lips', leaky_boolean_to, None, np.uint8, 1, None),
        ('shortness_of_breath', [na_value_to, 'no', 'mild', 'significant', 'severe'], None, np.uint8, 1, None),
        ('shortness_of_breath_binary', leaky_boolean_to, {na_value_from: 0, 'no': 1, 'mild': 2, 'significant': 2, 'severe': 2}, np.uint8, 1, None),
        ('skipped_meals', leaky_boolean_to, None, np.uint8, 1, None),
        ('sometimes_used_shortage', [na_value_to, 'all_needed', 'reused', 'not_enough'], None, np.uint8, 1, None),
        ('sore_throat', leaky_boolean_to, None, np.uint8, 1, None),
        ('tested_covid_positive', [na_value_to, 'waiting', 'no', 'yes'], None, np.uint8, 1, None),
        ('tested_covid_positive_clean', [na_value_to, 'waiting', 'no', 'yes'], None, np.uint8, 1, None),
        ('treated_patients_with_covid', [na_value_to, 'no', 'yes_suspected', 'yes_documented_suspected', 'yes_documented'], None, np.uint8, 1, None),
        ('treatment', [na_value_to, 'none', 'oxygen', 'nonInvasiveVentilation', 'invasiveVentilation'], None, np.uint8, 1, None, "freetext"),
        ('typical_hayfever', leaky_boolean_to, None, np.uint8, 1, None),
        ('unusual_muscle_pains', leaky_boolean_to, None, np.uint8, 1, None),
        ('worn_face_mask', [na_value_to, 'not_applicable', 'never', 'sometimes', 'most_of_the_time', 'always'], None, np.uint8, 1, None),
    ]
    _test_categorical_fields = [
        ('deleted', ['', 'False', 'True'], None, np.uint8, 1, None),
        ('invited_to_test', ['', 'False', 'True'], None, np.uint8, 1, None),
        ('location', ['', 'home', 'drive_through_rtc', 'hospital', 'gp', 'chemist', 'work', 'local_health_dept', 'drop_in_test_centre', 'other'], None, np.uint8, 1, None),
        ('mechanism', ['', 'nose_swab', 'throat_swab', 'nose_throat_swab', 'spit_tube', 'blood_sample',
                       'blood_sample_finger_prick', 'blood_sample_needle_draw'], None, np.uint8, 1, None, "freetext"),
        ('result', ['', 'waiting', 'failed', 'negative', 'positive'], None, np.uint8, 1, None),
        ('trained_worker', ['', 'trained', 'untrained', 'unsure'], None, np.uint8, 1, None),
    ]

    _assessment_field_entries = dict()
    for cf in _assessment_categorical_fields:
        desc = FieldDesc(cf[0], _build_map(cf[1]) if cf[2] is None else cf[2], cf[1], cf[3],
                         cf[6] if len(cf) == 7 else None)
        entry = FieldEntry(desc, cf[4], cf[5])
        entry_list = \
            list() if _assessment_field_entries.get(cf[0]) is None else _assessment_field_entries[cf[0]]
        entry_list.append(entry)
        _assessment_field_entries[cf[0]] = entry_list

    _patient_field_entries = dict()
    for cf in _patient_categorical_fields:
        desc = FieldDesc(cf[0], _build_map(cf[1]) if cf[2] is None else cf[2], cf[1], cf[3],
                         cf[6] if len(cf) == 7 else None)
        entry = FieldEntry(desc, cf[4], cf[5])
        entry_list = \
            list() if _patient_field_entries.get(cf[0]) is None else _patient_field_entries[cf[0]]
        entry_list.append(entry)
        _patient_field_entries[cf[0]] = entry_list


    _test_field_entries = dict()
    for cf in _test_categorical_fields:
        desc = FieldDesc(cf[0], _build_map(cf[1]) if cf[2] is None else cf[2], cf[1], cf[3],
                         cf[6] if len(cf) == 7 else None)
        entry = FieldEntry(desc, cf[4], cf[5])
        entry_list = \
            list() if _test_field_entries.get(cf[0]) is None else _test_field_entries[cf[0]]
        entry_list.append(entry)
        _test_field_entries[cf[0]] = entry_list


    def __init__(self, version):
        # TODO: field entries for patients!
        self.patient_categorical_maps = self._get_patient_categorical_maps(version)
        self.assessment_categorical_maps = self._get_assessment_categorical_maps(version)
        self.test_categorical_maps = self._get_test_categorical_maps(version)
        self.patient_field_types = DataSchema._patient_field_types
        self.assessment_field_types = DataSchema._assessment_field_types
        self.test_field_types = DataSchema._test_field_types


    def _validate_schema_number(self, schema):
        if schema not in DataSchema.data_schemas:
            raise DataSchemaVersionError(f'{schema} is not a valid cleaning schema value')


    def _get_patient_categorical_maps(self, version):
        return self._get_categorical_maps(DataSchema._patient_field_entries, version)


    def _get_assessment_categorical_maps(self, version):
        return self._get_categorical_maps(DataSchema._assessment_field_entries, version)


    def _get_test_categorical_maps(self, version):
        return self._get_categorical_maps(DataSchema._test_field_entries, version)


    def _get_categorical_maps(self, field_entries, version):
        self._validate_schema_number(version)

        selected_field_entries = dict()
        # get fields for which the schema number is in range
        for fe in field_entries.items():
            for e in fe[1]:
                if version >= e.version_from and (e.version_to is None or version < e.version_to):
                    selected_field_entries[fe[0]] = e.field_desc
                    break

        return selected_field_entries


    def string_field_desc(self):
        return FieldDesc('', None, None, str)
