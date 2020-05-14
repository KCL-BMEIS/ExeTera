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


class FieldDesc:
    def __init__(self, field, strings_to_values, values_to_strings, to_datatype):
        self.field = field
        self.to_datatype = to_datatype
        self.strings_to_values = strings_to_values
        self.values_to_strings = values_to_strings

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
    na_value_from = ''
    na_value_to = ''
    leaky_boolean_to = [na_value_to, 'False', 'True']
    leaky_boolean_from = _build_map(leaky_boolean_to)

    # tuple entries
    # 0: name
    # 1: values_to_strings,
    # 2: string_to_values or None if it should be calculated from values_to_string
    # 3: inclusive version from
    # 4: exclusive version to
    patient_categorical_fields = [
        ('age_filter', [na_value_to, 'bad', 'missing'], None, np.uint8, 1, None),
        ('weight_filter', [na_value_to, 'bad', 'missing'], None, np.uint8, 1, None),
        ('height_filter', [na_value_to, 'bad', 'missing'], None, np.uint8, 1, None),
        ('bmi_filter', [na_value_to, 'bad', 'missing'], None, np.uint8, 1, None)
    ]
    assessment_categorical_fields = [
        ('health_status', [na_value_to, 'healthy', 'not_healthy'], None, np.uint8, 1, None),
        ('fatigue', [na_value_to, 'no', 'mild', 'significant', 'severe'], None, np.uint8, 1, None),
        ('shortness_of_breath', [na_value_to, 'no', 'mild', 'significant', 'severe'], None, np.uint8, 1, None),
        ('abdominal_pain', leaky_boolean_to, None, np.uint8, 1, None),
        ('chest_pain', leaky_boolean_to, None, np.uint8, 1, None),
        ('delirium', leaky_boolean_to, None, np.uint8, 1, None),
        ('diarrhoea', leaky_boolean_to, None, np.uint8, 1, None),
        ('fever', leaky_boolean_to, None, np.uint8, 1, None),
        ('headache', leaky_boolean_to, None, np.uint8, 1, None),
        ('hoarse_voice', leaky_boolean_to, None, np.uint8, 1, None),
        ('loss_of_smell', leaky_boolean_to, None, np.uint8, 1, None),
        ('persistent_cough', leaky_boolean_to, None, np.uint8, 1, None),
        ('skipped_meals', leaky_boolean_to, None, np.uint8, 1, None),
        ('sore_throat', leaky_boolean_to, None, np.uint8, 1, None),
        ('unusual_muscle_pains', leaky_boolean_to, None, np.uint8, 1, None),
        ('always_used_shortage', [na_value_to, 'all_needed', 'reused'], None, np.uint8, 1, None),
        ('have_used_PPE', [na_value_to, 'never', 'sometimes', 'always'], None, np.uint8, 1, None),
        ('never_used_shortage', [na_value_to, 'not_needed', 'not_available'], None, np.uint8, 1, None),
        ('sometimes_used_shortage', [na_value_to, 'all_needed', 'reused', 'not_enough'], None, np.uint8, 1, None),
        ('treated_patients_with_covid', [na_value_to, 'no', 'yes_suspected', 'yes_documented_suspected', 'yes_documented'], None, np.uint8, 1, None),
        ('fatigue_binary', leaky_boolean_to, {na_value_from: 0, 'no': 1, 'mild': 2, 'significant': 2, 'severe': 2}, np.uint8, 1, None),
        ('shortness_of_breath_binary', leaky_boolean_to, {na_value_from: 0, 'no': 1, 'mild': 2, 'significant': 2, 'severe': 2}, np.uint8, 1, None),
        ('location', [na_value_to, 'home', 'hospital', 'back_from_hospital'], None, np.uint8, 1, None),
        ('level_of_isolation', [na_value_to, 'not_left_the_house', 'rarely_left_the_house', 'rarely_left_the_house_but_visited_lots', 'often_left_the_house'], None, np.uint8, 1, None),
        ('had_covid_test', leaky_boolean_to, None, np.uint8, 1, None),
        ('tested_covid_positive', [na_value_to, 'waiting', 'no', 'yes'], None, np.uint8, 1, None),
        ('had_covid_test_clean', leaky_boolean_to, None, np.uint8, 1, None),
        ('tested_covid_positive_clean', [na_value_to, 'waiting', 'no', 'yes'], None, np.uint8, 1, None)
    ]

    assessment_field_entries = dict()
    for cf in assessment_categorical_fields:
        entry = FieldEntry(FieldDesc(cf[0], _build_map(cf[1]) if cf[2] is None else cf[2], cf[1], cf[3]),
                           cf[4], cf[5])
        entry_list = \
            list() if assessment_field_entries.get(cf[0]) is None else assessment_field_entries[cf[0]]
        entry_list.append(entry)
        assessment_field_entries[cf[0]] = entry_list

    patient_field_entries = dict()
    for cf in patient_categorical_fields:
        entry = FieldEntry(FieldDesc(cf[0], _build_map(cf[1]) if cf[2] is None else cf[2], cf[1], cf[3]),
                           cf[4], cf[5])
        entry_list = \
            list() if patient_field_entries.get(cf[0]) is None else patient_field_entries[cf[0]]
        entry_list.append(entry)
        patient_field_entries[cf[0]] = entry_list


    def __init__(self, version):
        # TODO: field entries for patients!
        self.patient_categorical_maps = self._get_patient_categorical_maps(version)
        self.assessment_categorical_maps = self._get_assessment_categorical_maps(version)


    def _validate_schema_number(self, schema):
        if schema not in DataSchema.data_schemas:
            raise DataSchemaVersionError(f'{schema} is not a valid cleaning schema value')


    def _get_patient_categorical_maps(self, version):
        return self._get_categorical_maps(DataSchema.patient_field_entries, version)


    def _get_assessment_categorical_maps(self, version):
        return self._get_categorical_maps(DataSchema.assessment_field_entries, version)


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
