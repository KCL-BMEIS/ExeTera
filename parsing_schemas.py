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
                 f_missing_bmi, f_bad_bmi):
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

        self.weight_kg_clean = None
        self.height_cm_clean = None
        self.bmi_clean = None

    def valid(self, value):
        return self.min_height_inc <= value <= self.max_height_inc

    def __call__(self, weights, heights, bmis, filter_list):
        if len(weights) != len(heights):
            raise ValueError("'weights' and 'heights' are different lengths")
        if len(weights) != len(bmis):
            raise ValueError("'weights' and 'bmis' are different lengths")
        self.weight_kg_clean = np.zeros(len(weights), dtype=np.float)
        self.height_cm_clean = np.zeros(len(heights), dtype=np.float)
        self.bmi_clean = np.zeros(len(bmis), dtype=np.float)

        for ir in range(len(weights)):
            if weights[ir] == '':
                if self.f_missing_weight != 0:
                    filter_list[ir] |= self.f_missing_weight
            else:
                weight_clean = float(weights[ir])
                if weight_clean < self.min_weight_inc or weight_clean > self.max_weight_inc:
                    filter_list[ir] |= self.f_bad_weight
                else:
                    self.weight_kg_clean[ir] = weight_clean

            if heights[ir] == '':
                if self.f_missing_height != 0:
                    filter_list[ir] |= self.f_missing_height
            else:
                height_clean = float(heights[ir])
                if height_clean < self.min_height_inc or height_clean > self.max_height_inc:
                    filter_list[ir] |= self.f_bad_height
                else:
                    self.height_cm_clean[ir] = height_clean

            if bmis[ir] == '':
                if self.f_missing_bmi != 0:
                    filter_list[ir] |= self.f_missing_bmi
            else:
                bmi_clean = float(bmis[ir])
                if bmi_clean < self.min_bmi_inc or bmi_clean > self.max_bmi_inc:
                    filter_list[ir] |= self.f_bad_bmi
                else:
                    self.bmi_clean[ir] = bmi_clean
        return self.weight_kg_clean, self.height_cm_clean, self.bmi_clean


class ValidateHeight2:
    def __init__(self,
                 min_weight_inc, max_weight_inc,
                 min_height_inc, max_height_inc,
                 min_bmi_inc, max_bmi_inc,
                 f_missing_weight, f_bad_weight,
                 f_missing_height, f_bad_height,
                 f_missing_bmi, f_bad_bmi):
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

    def __call__(self, weights, heights, bmis, filter_list):
        if len(weights) != len(heights):
            raise ValueError("'weights' and 'heights' are different lengths")
        if len(weights) != len(bmis):
            raise ValueError("'weights' and 'bmis' are different lengths")
        self.weight_kg_clean = np.zeros(len(weights), dtype=np.float)
        self.height_cm_clean = np.zeros(len(heights), dtype=np.float)
        self.bmi_clean = np.zeros(len(bmis), dtype=np.float)

        for ir in range(len(weights)):
            if weights[ir] == '':
                if self.f_missing_weight != 0:
                    filter_list[ir] |= self.f_missing_weight
            else:
                weight_clean = float(weights[ir])
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

            if heights[ir] == '':
                if self.f_missing_height != 0:
                    filter_list[ir] |= self.f_missing_height
            else:
                height_clean = float(heights[ir])
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

                # Cleaning up bmi
                if weights[ir] == '' or heights[ir] == '' or height_clean == 0.0:
                    if self.f_missing_bmi != 0:
                        filter_list[ir] |= self.f_missing_bmi
                else:
                    bmi_clean = weight_clean / ((height_clean / 100) ** 2)

                    if bmi_clean < self.min_bmi_inc or bmi_clean > self.max_bmi_inc:
                        filter_list[ir] |= self.f_bad_bmi
                    else:
                        self.bmi_clean[ir] = bmi_clean

        return self.weight_kg_clean, self.height_cm_clean, self.bmi_clean


class ValidateHeight3:
    def __init__(self,
                 min_weight_inc, max_weight_inc,
                 min_height_inc, max_height_inc,
                 min_bmi_inc, max_bmi_inc,
                 f_missing_weight, f_bad_weight,
                 f_missing_height, f_bad_height,
                 f_missing_bmi, f_bad_bmi):
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

        self.m_to_cm = 100
        self.ft_to_cm = 30.48
        self.in_to_cm = 2.55
        self.mm_to_cm = 0.1

        self.mean_heights_by_age = [(64.1, 67.5),
                                    (80.7, 82.2),
                                    (85.5, 86.8),
                                    (94.0, 95.2),
                                    (100.3, 102.3),
                                    (107.9, 109.2),
                                    (115.5, 115.5),
                                    (121.1, 121.9),
                                    (128.2, 128),
                                    (133.3, 133.3),
                                    (138.4, 138.4),
                                    (144.0, 143.5),
                                    (149.8, 149.1),
                                    (156.7, 156.2),
                                    (158.7, 163.8),
                                    (159.7, 170.1),
                                    (162.5, 173.4),
                                    (162.5, 175.2),
                                    (163.0, 175.7),
                                    (163.0, 176.5),
                                    (163.3, 177.0)]
        self.mean_weights_by_age = [(7.5, 7.9),
                                    (10.6, 10.9),
                                    (12.0, 12.5),
                                    (14.2, 14.0),
                                    (15.4, 16.3),
                                    (17.9, 18.4),
                                    (19.9, 20.6),
                                    (22.4, 22.9),
                                    (25.8, 25.6),
                                    (28.1, 28.6),
                                    (31.9, 32.0),
                                    (36.9, 35.6),
                                    (41.5, 39.9),
                                    (45.8, 45.3),
                                    (47.6, 50.8),
                                    (52.1, 56.0),
                                    (53.5, 60.8),
                                    (54.4, 64.4),
                                    (56.7, 66.9),
                                    (57.1, 68.9),
                                    (58.0, 70.3)]

        self.mean_height_mults = (1.0, 1.0)
        min_height_mult = 110 / self.mean_heights_by_age[-1][1]
        max_height_mult = 220 / self.mean_heights_by_age[-1][1]
        self.min_height_mults = (self.mean_heights_by_age[-1][0] * min_height_mult,
                                 self.mean_heights_by_age[-1][1] * min_height_mult)
        self.max_height_mults = (self.mean_heights_by_age[-1][0] * max_height_mult,
                                 self.mean_heights_by_age[-1][1] * max_height_mult)
        print('max_height_mults =', self.max_height_mults)
        print('min_height_mults =', self.min_height_mults)
        # self.kgs_per_stone = 6.35029
        # self.kgs_per_lb = 0.453592
        # self.stones_per_kg = 1 / self.kgs_per_stone
        #
        # self.cms_per_foot = 30.48
        # self.cms_per_inch = 2.54
        # self.feet_per_cm = 1 / self.cms_per_foot

        self.weight_kg_clean = None
        self.height_cm_clean = None
        self.bmi_clean = None

    def _in_range(self, minv, maxv, v):
        return minv <= v <= maxv

    def __call__(self, sexes, yobs, weights, heights, bmis, filter_list):
        if len(weights) != len(heights):
            raise ValueError("'weights' and 'heights' are different lengths")
        if len(weights) != len(bmis):
            raise ValueError("'weights' and 'bmis' are different lengths")
        self.weight_kg_clean = np.zeros(len(weights), dtype=np.float)
        self.height_cm_clean = np.zeros(len(heights), dtype=np.float)
        self.bmi_clean = np.zeros(len(bmis), dtype=np.float)

        under_16 = 0
        missing_height_count = 0
        valid_heights = 0
        height_alternatives_histogram = [0, 0, 0, 0, 0]
        height_units_histogram = {'none': 0, 'm': 0, 'ft': 0, 'in': 0, 'cm': 0, 'mm': 0}
        for ir in range(len(heights)):
            if yobs[ir] is not '' and int(float(yobs[ir])) > 2004:
                under_16 += 1
                continue

            missing_height = False
            if heights[ir] == '':
                missing_height = True
                missing_height_count += 1
            else:
                height_cm = float(heights[ir])
                height_cm_valid = self._in_range(self.min_height_inc, self.max_height_inc, height_cm)
                if height_cm_valid:
                    valid_heights += 1
                    height_units_histogram['cm'] += 1
                    pass
                else:
                    valid_alternatives = 0
                    height_m = height_cm * self.m_to_cm
                    height_m_valid = self._in_range(self.min_height_inc, self.max_height_inc, height_m)
                    if height_m_valid:
                        valid_alternatives += 1
                        height_units_histogram['m'] += 1
                    height_ft = height_cm * self.ft_to_cm
                    height_ft_valid = self._in_range(self.min_height_inc, self.max_height_inc, height_ft)
                    if height_ft_valid:
                        valid_alternatives += 1
                        height_units_histogram['ft'] += 1
                    height_in = height_cm * self.in_to_cm
                    height_in_valid = self._in_range(self.min_height_inc, self.max_height_inc, height_in)
                    if height_in_valid:
                        valid_alternatives += 1
                        height_units_histogram['in'] += 1
                    height_mm = height_cm * self.mm_to_cm
                    height_mm_valid = self._in_range(self.min_height_inc, self.max_height_inc, height_mm)
                    if height_mm_valid:
                        valid_alternatives += 1
                        height_units_histogram['mm'] += 1
                    height_alternatives_histogram[valid_alternatives] += 1
                    if valid_alternatives == 0:
                        height_units_histogram['none'] += 1
        print('under_16 =', under_16)
        print('missing height count =', missing_height_count)
        print('valid heights =', valid_heights)
        print('valid alternatives =', height_alternatives_histogram)
        print('height units histogram =', height_units_histogram)

        for ir in range(len(weights)):
            if weights[ir] == '':
                if self.f_missing_weight != 0:
                    filter_list[ir] |= self.f_missing_weight
            else:
                weight_clean = float(weights[ir])
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

            if heights[ir] == '':
                if self.f_missing_height != 0:
                    filter_list[ir] |= self.f_missing_height
            else:
                height_clean = float(heights[ir])
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

                # Cleaning up bmi
                if weights[ir] == '' or heights[ir] == '' or height_clean == 0.0:
                    if self.f_missing_bmi != 0:
                        filter_list[ir] |= self.f_missing_bmi
                else:
                    bmi_clean = weight_clean / ((height_clean / 100) ** 2)

                    if bmi_clean < self.min_bmi_inc or bmi_clean > self.max_bmi_inc:
                        filter_list[ir] |= self.f_bad_bmi
                    else:
                        self.bmi_clean[ir] = bmi_clean

        return self.weight_kg_clean, self.height_cm_clean, self.bmi_clean

class ValidateTemperature1:
    def __init__(self, min_temp_incl, max_temp_incl, f_missing_temp, f_bad_temp):
        self.min_temp_incl = min_temp_incl
        self.max_temp_incl = max_temp_incl
        self.f_missing_temp = f_missing_temp
        self.f_bad_temp = f_bad_temp

    def __call__(self, temps, filter_list):
        temperature_c = np.zeros_like(temps, dtype=np.float)
        for ir, t in enumerate(temps):
            if t == '':
                dest_temp = 0.0
                filter_list[ir] |= self.f_missing_temp
            else:
                t = float(t)
                dest_temp = (t - 32) / 1.8 if t > self.max_temp_incl else t
                if dest_temp == 0.0:
                    filter_list[ir] |= self.f_missing_temp
                    temperature_c[ir] = dest_temp
                elif dest_temp <= self.min_temp_incl or dest_temp >= self.max_temp_incl:
                    temperature_c[ir] = 0.0
                    filter_list[ir] |= self.f_bad_temp
                else:
                    temperature_c[ir] = dest_temp

        return temperature_c


class ValidateCovidTestResultsFacVersion1PreHCTFix:
    def __init__(self, hcts, tcps, filter_status, results_key, hct_results, results, filter_flag, show_debug=False):
        self.valid_transitions = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 2), 3: (0, 3)}
        self.upgrades = {0: (0, 1, 2, 3), 1: (2, 3), 2: tuple(), 3: tuple()}
        self.hcts = hcts
        self.tcps = tcps
        self.hct_results = hct_results
        self.results = results
        self.filter_status = filter_status
        self.filter_flag = filter_flag
        self.show_debug = show_debug

    def __call__(self, patient_id, filter_status, start, end):
        # validate the subrange
        invalid = False
        max_value = 0
        for j in range(start, end + 1):
            # allowable transitions
            value = self.tcps[j]
            if value not in self.valid_transitions[max_value]:
                invalid = True
                break
            if value in self.upgrades[max_value]:
                max_value = value
            self.results[j] = max_value

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.tcps[j]
                filter_status[j] |= self.filter_flag

        self.hct_results[start:end+1] = self.hcts[start:end+1]
        if invalid:
            for j in range(start, end + 1):
                filter_status[j] |= self.filter_flag

        if self.show_debug == True:
            if invalid or not np.array_equal(self.hcts[start:end+1], self.hct_results[start:end+1])\
               or not np.array_equal(self.tcps[start:end+1], self.results[start:end+1]):
                reason = 'inv' if invalid else 'diff'
                print(reason, start, 'hct:', self.hcts[start:end+1], self.hct_results[start:end+1])
                print(reason, start, 'tcp:', self.tcps[start:end+1], self.results[start:end+1])

        # TODO: remove before v0.1.8
        # for j in range(start, end + 1):
        #     self.hct_results[j] = self.hcts[j]

        # if invalid:
        #     for j in range(start, end + 1):
        #         if self.hct_results[j] == 1 and self.results[j] != 0:
        #             print('hct:', start, self.hcts[start:end+1], self.hct_results[start:end+1])
        #             print('tcp:', start, self.tcps[start:end+1], self.results[start:end+1])
        #             break


class ValidateCovidTestResultsFacVersion1:
    def __init__(self, hcts, tcps, filter_status, results_key, hct_results, results, filter_flag, show_debug=False):
        self.valid_transitions = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 2), 3: (0, 3)}
        self.upgrades = {0: (0, 1, 2, 3), 1: (2, 3), 2: tuple(), 3: tuple()}
        self.hcts = hcts
        self.tcps = tcps
        self.hct_results = hct_results
        self.results = results
        self.filter_status = filter_status
        self.filter_flag = filter_flag
        self.show_debug = show_debug

    def __call__(self, patient_id, filter_status, start, end):
        # validate the subrange
        invalid = False
        max_value = 0
        for j in range(start, end + 1):
            # allowable transitions
            value = self.tcps[j]
            if value not in self.valid_transitions[max_value]:
                invalid = True
                break
            if value in self.upgrades[max_value]:
                max_value = value
            self.results[j] = max_value

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.tcps[j]
                filter_status[j] |= self.filter_flag

        if not invalid:
            first_hct_false = -1
            first_hct_true = -1
            for j in range(start, end + 1):
                if self.hcts[j] == 1:
                    if first_hct_false == -1:
                        first_hct_false = j
                elif self.hcts[j] == 2:
                    if first_hct_true == -1:
                        first_hct_true = j

            max_value = 0
            for j in range(start, end + 1):
                if j == first_hct_false:
                    max_value = max(max_value, 1)
                if j == first_hct_true:
                    max_value = 2

                self.hct_results[j] = max_value
        else:
            for j in range(start, end + 1):
                self.hct_results[j] = self.hcts[j]
                filter_status[j] |= self.filter_flag

        if self.show_debug == True:
            if invalid or not np.array_equal(self.hcts[start:end+1], self.hct_results[start:end+1])\
               or not np.array_equal(self.tcps[start:end+1], self.results[start:end+1]):
                reason = 'inv' if invalid else 'diff'
                print(reason, start, 'hct:', self.hcts[start:end+1], self.hct_results[start:end+1])
                print(reason, start, 'tcp:', self.tcps[start:end+1], self.results[start:end+1])

        # TODO: remove before v0.1.8
        # for j in range(start, end + 1):
        #     self.hct_results[j] = self.hcts[j]

        # if invalid:
        #     for j in range(start, end + 1):
        #         if self.hct_results[j] == 1 and self.results[j] != 0:
        #             print('hct:', start, self.hcts[start:end+1], self.hct_results[start:end+1])
        #             print('tcp:', start, self.tcps[start:end+1], self.results[start:end+1])
        #             break


class ValidateCovidTestResultsFacVersion2:
    def __init__(self, tcps, filter_status, results_key, results, filter_flag, show_debug=False):
        # TODO: this is all actually dependent on the data schema so that must be checked
        # self.valid_transitions = {
        #     '': ('', 'waiting', 'yes', 'no'),
        #     'waiting': ('', 'waiting', 'yes', 'no'),
        #     'no': ('', 'no'),
        #     'yes': ('', 'yes')
        # }
        # self.valid_transitions_before_yes = {
        #     '': ('', 'waiting', 'yes', 'no'),
        #     'waiting': ('', 'waiting', 'yes', 'no'),
        #     'no': ('', 'waiting', 'no', 'yes'),
        #     'yes': ('', 'yes')
        # }
        # self.upgrades = {
        #     '': ('waiting', 'yes', 'no'),
        #     'waiting': ('yes', 'no'),
        #     'no': (),
        #     'yes': ()
        # }
        # self.upgrades_before_yes = {
        #     '': ('waiting', 'yes', 'no'),
        #     'waiting': ('yes',),
        #     'no': ('waiting', 'yes'),
        #     'yes': ()
        # }
        # self.key_to_value = {
        #     '': 0,
        #     'waiting': 1,
        #     'no': 2,
        #     'yes': 3
        # }
        self.valid_transitions = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 2), 3: (0, 3)}
        self.valid_transitions_before_yes =\
            {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 1, 2, 3), 3: (0, 3)}
        self.upgrades = {0: (0, 1, 2, 3), 1: (2, 3), 2: tuple(), 3: tuple()}
        self.upgrades_before_yes = {0: (1, 2, 3), 1: (3,), 2: (1, 3), 3: tuple()}

        self.tcps = tcps
        self.results = results
        self.filter_status = filter_status
        self.filter_flag = filter_flag
        self.show_debug = show_debug

    def __call__(self, patient_id, filter_status, start, end):
        # validate the subrange
        invalid = False
        max_value = 0
        first_waiting = -1
        first_no = -1
        first_yes = -1

        for j in range(start, end + 1):
            value = self.tcps[j]
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
            value = self.tcps[j]
            if value not in valid_transitions[max_value]:
                invalid = True
                break
            if j < first_yes and value == 'no':
                value == 'waiting'
            if value in upgrades[max_value]:
                max_value = value
            self.results[j] = max_value

        #rescue na -> waiting -> no -> waiting
        if invalid and first_yes == -1 and self.tcps[end] == 'waiting':
            invalid = False
            max_value = ''
            for j in range(start, end+1):
                value = self.tcps[j]
                if max_value == '' and value != '':
                    max_value = 'waiting'
                self.results[j] = max_value

        if invalid:
            for j in range(start, end + 1):
                self.results[j] = self.tcps[j]
                filter_status[j] |= self.filter_flag
            if self.show_debug:
                print(self.tcps[j], end=': ')
                for j in range(start, end + 1):
                    if j > start:
                        print(' ->', end=' ')
                    value = self.tcps[j]
                    print('na' if value == '' else value, end='')
                print('')

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
