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


def validate_temperature_1(min_temp, max_temp,
                           temps, temp_units, temp_set,
                           dest_temps, dest_temps_valid, dest_temps_modified):
    raw_temps = temps[:]
    raw_dest_temps = np.where(raw_temps > max_temp, (raw_temps - 32) / 1.8, raw_temps)
    raw_dest_temps_valid = temp_set[:] & (min_temp <= raw_dest_temps) & (raw_dest_temps <= max_temp)
    raw_dest_temps_modified = raw_temps != raw_dest_temps
    dest_temps.write(raw_dest_temps)
    dest_temps_valid.write(raw_dest_temps_valid)
    dest_temps_modified.write(raw_dest_temps_modified)
