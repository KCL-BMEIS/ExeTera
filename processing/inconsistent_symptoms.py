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

from utils import check_input_lengths

class CheckInconsistentSymptoms:
    def __init__(self, f_healthy_but_symptoms, f_not_healthy_but_no_symptoms):
        self.f_healthy_but_symptoms = f_healthy_but_symptoms
        self.f_not_healthy_but_no_symptoms = f_not_healthy_but_no_symptoms

    def __call__(self, healthy, symptoms, flags, i_healthy, i_not_healthy):
        check_input_lengths(('healthy', 'symptoms', 'flags'), (healthy, symptoms, flags))
        for i_r in range(len(healthy)):
            if healthy[i_r] == i_healthy and symptoms[i_r]:
                flags[i_r] |= self.f_healthy_but_symptoms
            elif healthy[i_r] == i_not_healthy and not symptoms[i_r]:
                flags[i_r] |= self.f_not_healthy_but_no_symptoms

# src_health_status = asmt_ds.field_by_name('health_status')
# i_healthy = categorical_maps['health_status'].strings_to_values['healthy']
# i_not_healthy = categorical_maps['health_status'].strings_to_values['not_healthy']
# for ir in range(asmt_ds.row_count()):
#     if src_health_status[ir] == i_healthy and any_symptoms[ir]:
#         asmt_filter_status[ir] |= FILTER_INCONSISTENT_SYMPTOMS
#     elif src_health_status[ir] == i_not_healthy and not any_symptoms[ir]:
#         asmt_filter_status[ir] |= FILTER_INCONSISTENT_NO_SYMPTOMS
#
# for f in (FILTER_INCONSISTENT_SYMPTOMS, FILTER_INCONSISTENT_NO_SYMPTOMS):
#     print(f'{assessment_flag_descs[f]}: {count_flag_set(asmt_filter_status, f)}')