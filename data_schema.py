
class FieldEntry:
    def __init__(self, field, strings_to_values, values_to_strings, version_from, version_to=None):
        self.field = field
        self.strings_to_values = strings_to_values
        self.values_to_strings = values_to_strings
        self.version_from = version_from
        self.version_to = version_to

    def __str__(self):
        str = 'FieldEntry(field={}, strings_to_values={}, values_to_strings={}, version_from={}, version_to={})'
        return str.format(self.field, self.strings_to_values, self.values_to_string,
                          self.version_from, self.version_to)

    def __repr__(self):
        return self.__str__()


class CleaningSchemaError(Exception):
    pass


def validate_schema_number(schema):
    if schema not in data_schemas:
        raise CleaningSchemaError(f'{schema} is not a valid cleaning schema value')


def get_categorical_maps(version):
    validate_schema_number(version)

    selected_field_entries = dict()
    # get fields for which the schema number is in range
    for fe in field_entries.items():
        for e in fe[1]:
            if version >= e.version_from and (e.version_to is None or version < e.version_to):
                selected_field_entries[fe[0]] = e

    return selected_field_entries


def _build_map(value_list):
    inverse = dict()
    for ir, r in enumerate(value_list):
        inverse[r] = ir
    return inverse


data_schemas = [1]
cleaning_schemas = [1]
na_value_from = ''
na_value_to = ''
leaky_boolean_to = [na_value_to, 'False', 'True']
leaky_boolean_from = _build_map(leaky_boolean_to)

field_entries = dict()

# tuple entries
# 0: name
# 1: values_to_strings,
# 2: string_to_values or None if it should be calculated from values_to_string
# 3: inclusive version from
# 4: exclusive version to
categorical_fields = [
    ('fatigue', [na_value_to, 'no', 'mild', 'significant', 'severe'], None, 1, None),
    ('shortness_of_breath', [na_value_to, 'no', 'mild', 'significant', 'severe'], None, 1, None),
    ('abdominal_pain', leaky_boolean_to, None, 1, None),
    ('chest_pain', leaky_boolean_to, None, 1, None),
    ('delirium', leaky_boolean_to, None, 1, None),
    ('diarrhoea', leaky_boolean_to, None, 1, None),
    ('fever', leaky_boolean_to, None, 1, None),
    ('headache', leaky_boolean_to, None, 1, None),
    ('hoarse_voice', leaky_boolean_to, None, 1, None),
    ('loss_of_smell', leaky_boolean_to, None, 1, None),
    ('persistent_cough', leaky_boolean_to, None, 1, None),
    ('skipped_meals', leaky_boolean_to, None, 1, None),
    ('sore_throat', leaky_boolean_to, None, 1, None),
    ('unusual_muscle_pains', leaky_boolean_to, None, 1, None),
    ('always_used_shortage', [na_value_to, 'all_needed', 'reused'], None, 1, None),
    ('have_used_PPE', [na_value_to, 'never', 'sometimes', 'always'], None, 1, None),
    ('never_used_shortage', [na_value_to, 'not_needed', 'not_available'], None, 1, None),
    ('sometimes_used_shortage', [na_value_to, 'all_needed', 'reused', 'not_enough'], None, 1, None),
    ('treated_patients_with_covid', [na_value_to, 'no', 'yes_suspected', 'yes_documented_suspected', 'yes_documented'], None, 1, None),
    ('fatigue_binary', leaky_boolean_to, {na_value_from: 0, 'no': 1, 'mild': 2, 'severe': 2}, 1, None),
    ('shortness_of_breath_binary', leaky_boolean_to, {na_value_from: 0, 'no': 1, 'mild': 2, 'significant': 2, 'severe': 2}, 1, None),
    ('location', [na_value_to, 'home', 'hospital', 'back_from_hospital'], None, 1, None),
    ('level_of_isolation', [na_value_to, 'not_left_the_house', 'rarely_left_the_house',
                            'rarely_left_the_house_but_visited_lots', 'often_left_the_house'],
                            None, 1, None),
    ('had_covid_test', leaky_boolean_to, None, 1, None),
    ('tested_covid_positive', [na_value_to, 'waiting', 'no', 'yes'], None, 1, None)
]

field_entries = dict()
for cf in categorical_fields:
    entry = FieldEntry('str', _build_map(cf[1]) if cf[2] is None else cf[2], cf[1], cf[3], cf[4])
    entry_list = list() if field_entries.get(cf[0]) is None else field_entries[cf[0]]
    entry_list.append(entry)
    field_entries[cf[0]] = entry_list
