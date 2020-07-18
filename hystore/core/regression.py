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

from hystore.core import utils


def na_or_value(value):
    if value == '':
        return 'na'
    return value

def na_compare(value1, value2):
    lv1 = value1.lower()
    lv2 = value2.lower()
    if lv1 == 'na' and lv2 in ('', 'na'):
        return True
    return lv1 == lv2

def datetime_compare_to_secs(value1, value2):
    dt1 = utils.datetime_to_seconds(value1)
    dt2 = utils.datetime_to_seconds(value2)
    return dt1 == dt2

def check_row(exp_ds, exp_index, act_ds, act_index, keys, custom_checks):
    disparities = None
    for k in keys:
        if isinstance(k, str):
            kexp, kact = k, k
        else:
            kexp, kact = k
        if kexp in custom_checks:
            check = custom_checks[kexp]
        else:
            check = na_compare
        exp_value = exp_ds.field_by_name(kexp)[exp_index]
        act_value = act_ds.field_by_name(kact)[act_index]
        if not check(exp_value, act_value):
            if disparities is None:
                disparities = list()
            if kexp == kact:
                disparities.append(f'{kexp} : {str(na_or_value(exp_value))} / {str(na_or_value(act_value))}')
            else:
                disparities.append(f'{kexp} / {kact} : {str(na_or_value(exp_value))} / {str(na_or_value(act_value))}')
    return disparities