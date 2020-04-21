

class ParsingSchemaVersionError(Exception):
    pass


class ClassEntry:
    def __init__(self, key, class_definition, version_from, version_to=None):
        self.key = key
        self.class_definition = class_definition
        self.version_from = version_from
        self.version_to = version_to

    def __str__(self):
        output = 'ClassEntry(field={}, class_definition={}, version_from={}, version_to={})'
        return output.format(self.key, self.class_definition,
                             self.version_from, self.version_to)

    def __repr__(self):
        return self.__str__()


class ValidateCovidTestResultsFacVersion1:
    def __init__(self, dataset, filter_status, results_key, results, filter_flag):
        self.valid_transitions = {
            '': ('', 'waiting', 'yes', 'no'),
            'waiting': ('', 'waiting', 'yes', 'no'),
            'no': ('', 'no'),
            'yes': ('', 'yes')
        }
        self.upgrades = {
            '': ('waiting', 'yes', 'no'),
            'waiting': ('yes', 'no'),
            'no': (),
            'yes': ()
        }
        self.key_to_value = {
            '': 0,
            'waiting': 1,
            'no': 2,
            'yes': 3
        }
        self.dataset = dataset
        self.results = results
        self.filter_status = filter_status
        self.tcp_index = self.dataset.field_to_index('tested_covid_positive')
        self.filter_flag = filter_flag

    def __call__(self, fields, filter_status, start, end):
        # validate the subrange
        invalid = False
        max_value = ''
        for j in range(start, end + 1):
            # allowable transitions
            value = fields[j][self.tcp_index]
            if value not in self.valid_transitions[max_value]:
                invalid = True
                break
            if value in self.upgrades[max_value]:
                max_value = value
            self.results[j] = self.key_to_value[max_value]

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.key_to_value[fields[j][self.tcp_index]]
                filter_status[j] |= self.filter_flag


class ParsingSchema:
    def __init__(self, schema_number):

        self.parsing_schemas = [1, 2]
        self.functors = {
            'clean_covid_progression': [
                ClassEntry('validate_covid_fields', ValidateCovidTestResultsFacVersion1, 1, None)
            ]
        }

        self._validate_schema_number(schema_number)

        self.class_entries = dict()
        for f in self.functors.items():
            for e in f[1]:
                if schema_number >= e.version_from and\
                  (e.version_to is None or schema_number < e.version_to):
                    self.class_entries[f[0]] = e.class_definition
                    break

    def _validate_schema_number(self, schema_number):
        if schema_number not in self.parsing_schemas:
            raise ParsingSchemaVersionError(f'{schema} is not a valid cleaning schema value')
