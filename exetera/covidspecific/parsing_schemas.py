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

from exetera.processing.covid_test import ValidateCovidTestResultsFacVersion1, ValidateCovidTestResultsFacVersion2
from exetera.processing.temperature import ValidateTemperature1
from exetera.processing.weight_height_bmi import ValidateHeight1, ValidateHeight2


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


parsing_schemas = [1, 2]

class ParsingSchema:
    def __init__(self, schema_number):

        #self.parsing_schemas = [1, 2]
        self.functors = {
            'validate_weight_height_bmi': [
                ClassEntry('validate_weight_height_bmi', ValidateHeight1, 1, 2),
                ClassEntry('validate_weight_height_bmi', ValidateHeight2, 2, None)],
            'validate_temperature': [
                ClassEntry('validate_temperature', ValidateTemperature1, 1, None)],
            'clean_covid_progression': [
                ClassEntry('validate_covid_fields', ValidateCovidTestResultsFacVersion1, 1, 2),
                ClassEntry('validate_covid_fields', ValidateCovidTestResultsFacVersion2, 2, None)]
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
        if schema_number not in parsing_schemas:
            raise ParsingSchemaVersionError(
                f'{schema_number} is not a valid cleaning schema value')
