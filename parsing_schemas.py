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

class ValidateHeight1:
    def __init__(self,
                 min_weight_inc, max_weight_inc,
                 min_height_inc, max_height_inc,
                 min_bmi_inc, max_bmi_inc,
                 f_missing_weight, f_bad_weight,
                 f_missing_height, f_bad_height,
                 f_missing_bmi, f_bad_bmi,
                 weight_index, height_index, bmi_index):
        self.min_weight_inc = min_weight_inc
        self.max_weight_inc = max_weight_inc
        self.min_height_inc = min_height_inc
        self.max_height_inc = max_height_inc
        self.min_bmi_inc = min_bmi_inc
        self.max_bmi_inc = max_bmi_inc

        self.f_missing_weight = f_missing_weight
        self.f_bad_weight = f_bad_weight
        self.f_missing_height = f_missing_height
        self.f_bad_height = f_bad_height
        self.f_missing_bmi = f_missing_bmi
        self.f_bad_bmi = f_bad_bmi

        self.weight_index = weight_index
        self.height_index = height_index
        self.bmi_index = bmi_index

        self.weight_kg_clean = None
        self.height_cm_clean = None
        self.bmi_clean = None

    def valid(self, value):
        return self.min_height_inc <= value <= self.max_height_inc

    def __call__(self, fields, filter_list):
        self.weight_kg_clean = np.zeros(len(fields), dtype=np.float)
        self.height_cm_clean = np.zeros(len(fields), dtype=np.float)
        self.bmi_clean = np.zeros(len(fields), dtype=np.float)

        for ir, r in enumerate(fields):
            if r[self.weight_index] == '':
                if self.f_missing_weight != 0:
                    filter_list[ir] |= self.f_missing_weight
            else:
                weight_clean = float(r[self.weight_index])
                if weight_clean < self.min_weight_inc or weight_clean > self.max_weight_inc:
                    filter_list[ir] |= self.f_bad_weight
                else:
                    self.weight_kg_clean[ir] = weight_clean

            if r[self.height_index] == '':
                if self.f_missing_height != 0:
                    filter_list[ir] |= self.f_missing_height
            else:
                height_clean = float(r[self.height_index])
                if height_clean < self.min_height_inc or height_clean > self.max_height_inc:
                    filter_list[ir] |= self.f_bad_height
                else:
                    self.height_cm_clean[ir] = height_clean

            if r[self.bmi_index] == '':
                if self.f_missing_bmi != 0:
                    filter_list[ir] |= self.f_missing_bmi
            else:
                bmi_clean = float(r[self.bmi_index])
                if bmi_clean < self.min_bmi_inc or bmi_clean > self.max_bmi_inc:
                    filter_list[ir] |= self.f_bad_bmi
                else:
                    self.bmi_clean[ir] = bmi_clean
        return self.weight_kg_clean, self.height_cm_clean, self.bmi_clean


class ValidateHeight2:
    def __init__(self, min_weight_inc, max_weight_inc, min_height_inc, max_height_inc, min_bmi_inc, max_bmi_inc,
                 f_missing_weight, f_bad_weight, f_missing_height, f_bad_height, f_missing_bmi, f_bad_bmi,
                 weight_index, height_index, bmi_index):
        self.min_weight_inc = min_weight_inc
        self.max_weight_inc = max_weight_inc
        self.min_height_inc = min_height_inc
        self.max_height_inc = max_height_inc
        self.min_bmi_inc = min_bmi_inc
        self.max_bmi_inc = max_bmi_inc

        self.f_missing_weight = f_missing_weight
        self.f_bad_weight = f_bad_weight
        self.f_missing_height = f_missing_height
        self.f_bad_height = f_bad_height
        self.f_missing_bmi = f_missing_bmi
        self.f_bad_bmi = f_bad_bmi

        self.weight_index = weight_index
        self.height_index = height_index
        self.bmi_index = bmi_index

        self.kgs_per_stone = 6.35029
        self.kgs_per_lb = 0.453592
        self.stones_per_kg = 1 / self.kgs_per_stone

        self.cms_per_foot = 30.48
        self.cms_per_inch = 2.54
        self.feet_per_cm = 1 / self.cms_per_foot

        self.weight_kg_clean = None
        self.height_cm_clean = None
        self.bmi_clean = None

    def valid(self, value):
        return self.min_height_inc <= value <= self.max_height_inc

    def __call__(self, fields, filter_list):
        self.weight_kg_clean = np.zeros(len(fields), dtype=np.float)
        self.height_cm_clean = np.zeros(len(fields), dtype=np.float)
        self.bmi_clean = np.zeros(len(fields), dtype=np.float)

        for ir, r in enumerate(fields):
            if r[self.weight_index] == '':
                if self.f_missing_weight != 0:
                    filter_list[ir] |= self.f_missing_weight
            else:
                weight_clean = float(r[self.weight_index])
                if weight_clean < 25:
                    weight_clean *= self.kgs_per_stone
                elif 150 <= weight_clean < 300:
                    weight_clean *= self.kgs_per_lb
                elif 300 <= weight_clean < 450:
                    weight_clean *= self.stones_per_kg
                elif 450 <= weight_clean < 1500:
                    weight_clean *= 0.1
                # second pass on partially sanitised figure
                if 450 <= weight_clean < 600:
                    weight_clean *= self.stones_per_kg
                    
                if weight_clean < self.min_weight_inc or weight_clean > self.max_weight_inc:
                    filter_list[ir] |= self.f_bad_weight
                else:
                    self.weight_kg_clean[ir] = weight_clean

                # df_reg_new['weight_clean'] = df_reg_new['weight_kg']
                # # Stone range
                # df_reg_new['weight_clean'] = np.where(df_reg_new['weight_kg'] < 25, 6.35029 * df_reg_new['weight_kg'],
                #                                       df_reg_new['weight_clean'])
                # # Pounds range
                # df_reg_new['weight_clean'] = np.where(
                #     np.logical_and(df_reg_new['weight_kg'] > 150, df_reg_new['weight_kg'] < 300),
                #     0.453592 * df_reg_new['weight_kg'], df_reg_new['weight_clean'])
                # # Mix up imperial for metric
                # df_reg_new['weight_clean'] = np.where(
                #     np.logical_and(df_reg_new['weight_kg'] > 300, df_reg_new['weight_kg'] < 450),
                #     df_reg_new['weight_kg'] / 6.35029, df_reg_new['weight_clean'])
                # df_reg_new['weight_clean'] = np.where(
                #     np.logical_and(df_reg_new['weight_kg'] > 600, df_reg_new['weight_kg'] < 1500),
                #     df_reg_new['weight_kg'] / 10, df_reg_new['weight_clean'])
                # df_reg_new['weight_clean'] = np.where(
                #     np.logical_and(df_reg_new['weight_clean'] > 450, df_reg_new['weight_clean'] < 600),
                #     df_reg_new['weight_clean'] / 6.35029, df_reg_new['weight_clean'])

            if r[self.height_index] == '':
                if self.f_missing_height != 0:
                    filter_list[ir] |= self.f_missing_height
            else:
                height_clean = float(r[self.height_index])
                if height_clean < 2.4:
                    height_clean *= 100
                elif height_clean < 7.4:
                    height_clean *= self.cms_per_foot
                elif height_clean > 4000:
                    height_clean *= self.feet_per_cm

                if height_clean < self.min_height_inc or height_clean > self.max_height_inc:
                    filter_list[ir] |= self.f_bad_height
                else:
                    self.height_cm_clean[ir] = height_clean

                # df_reg_new['height_clean'] = df_reg_new['height_cm']
                # # Mix up meters for cm
                # df_reg_new['height_clean'] = np.where(df_reg_new['height_cm'] < 2.4, 100 * df_reg_new['height_cm'],
                #                                       df_reg_new['height_clean'])
                # # Mix up imperial for metric
                # df_reg_new['height_clean'] = np.where(df_reg_new['height_clean'] < 7.4,
                #                                       30.48 * np.round(df_reg_new['height_cm'] / 10, 0) + 2.54 *
                #                                       df_reg_new['height_cm'] % 10, df_reg_new['height_clean'])
                # # df_reg_new['height_clean'] = np.where(np.logical_and(df_reg_new['height_clean']>40, df_reg_new['height_clean']<75),
                # # Mix up metric for imperial
                # df_reg_new['height_clean'] = np.where(df_reg_new['height_clean'] > 4000,
                #                                       df_reg_new['height_clean'] / 30.48, df_reg_new['height_clean'])

                # Cleaning up bmi
                if r[self.weight_index] == '' or r[self.height_index] == '' or height_clean == 0.0:
                    if self.f_missing_bmi != 0:
                        filter_list[ir] |= self.f_missing_bmi
                else:
                    bmi_clean = weight_clean / ((height_clean / 100) ** 2)

                    if bmi_clean < self.min_bmi_inc or bmi_clean > self.max_bmi_inc:
                        filter_list[ir] |= self.f_bad_bmi
                    else:
                        self.bmi_clean[ir] = bmi_clean

                # df_reg_new['bmi_clean'] = df_reg_new['weight_clean'] / np.square(df_reg_new['height_clean'] / 100)
                # return df_reg_new
                # value = float(r[index])
                # if not self.valid(value):
                #     if self.f_bad != 0:
                #         filter_list[ir] |= self.f_bad
        return self.weight_kg_clean, self.height_cm_clean, self.bmi_clean


class ValidateCovidTestResultsFacVersion1:
    def __init__(self, dataset, filter_status, results_key, results, filter_flag, show_debug=False):
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
        self.show_debug = show_debug

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
            if self.show_debug:
                print(fields[j][1], end=': ')
                for j in range(start, end + 1):
                    if j > start:
                        print(' ->', end=' ')
                    value = fields[j][self.tcp_index]
                    print('na' if value == '' else value, end='')
                print('')


class ValidateCovidTestResultsFacVersion2:
    def __init__(self, dataset, filter_status, results_key, results, filter_flag, show_debug=False):
        # TODO: this is all actually dependent on the data schema so that must be checked
        self.valid_transitions = {
            '': ('', 'waiting', 'yes', 'no'),
            'waiting': ('', 'waiting', 'yes', 'no'),
            'no': ('', 'no'),
            'yes': ('', 'yes')
        }
        self.valid_transitions_before_yes = {
            '': ('', 'waiting', 'yes', 'no'),
            'waiting': ('', 'waiting', 'yes', 'no'),
            'no': ('', 'waiting', 'no', 'yes'),
            'yes': ('', 'yes')
        }
        self.upgrades = {
            '': ('waiting', 'yes', 'no'),
            'waiting': ('yes', 'no'),
            'no': (),
            'yes': ()
        }
        self.upgrades_before_yes = {
            '': ('waiting', 'yes', 'no'),
            'waiting': ('yes',),
            'no': ('waiting', 'yes'),
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
        self.show_debug = show_debug

    def __call__(self, fields, filter_status, start, end):
        # validate the subrange
        invalid = False
        max_value = ''
        first_waiting = -1
        first_no = -1
        first_yes = -1

        for j in range(start, end + 1):
            value = fields[j][self.tcp_index]
            if value == 'waiting' and first_waiting == -1:
                first_waiting = j
            if value == 'no' and first_no == -1:
                first_no = j
            if value == 'yes' and first_yes == -1:
                first_yes = j

        for j in range(start, end + 1):
            valid_transitions = self.valid_transitions_before_yes if j <= first_yes else self.valid_transitions
            upgrades = self.upgrades_before_yes if j <= first_yes else self.upgrades
            # allowable transitions
            value = fields[j][self.tcp_index]
            if value not in valid_transitions[max_value]:
                invalid = True
                break
            if j < first_yes and value == 'no':
                value == 'waiting'
            if value in upgrades[max_value]:
                max_value = value
            self.results[j] = self.key_to_value[max_value]

        #rescue na -> waiting -> no -> waiting
        if invalid and first_yes == -1 and fields[end][self.tcp_index] == 'waiting':
            invalid = False
            max_value = ''
            for j in range(start, end+1):
                value = fields[j][self.tcp_index]
                if max_value == '' and value != '':
                    max_value = 'waiting'
                self.results[j] = self.key_to_value[max_value]

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.key_to_value[fields[j][self.tcp_index]]
                filter_status[j] |= self.filter_flag
            if self.show_debug:
                print(fields[j][1], end=': ')
                for j in range(start, end + 1):
                    if j > start:
                        print(' ->', end=' ')
                    value = fields[j][self.tcp_index]
                    print('na' if value == '' else value, end='')
                print('')



class ParsingSchema:
    def __init__(self, schema_number):

        self.parsing_schemas = [1, 2]
        self.functors = {
            'validate_weight_height_bmi': [
                ClassEntry('validate_weight_height_bmi', ValidateHeight1, 1, 2),
                ClassEntry('validate_weight_height_bmi', ValidateHeight2, 2, None)],
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
        if schema_number not in self.parsing_schemas:
            raise ParsingSchemaVersionError(f'{schema} is not a valid cleaning schema value')
