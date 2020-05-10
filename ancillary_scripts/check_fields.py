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

import dataset
import pipeline
import utils

filename = 'test_patients.csv'
with open(filename) as f:
    ds = dataset.Dataset(f)

for ir in range(ds.row_count()):
    utils.print_diagnostic_row(f'{ir}', ds, ir,
                               ('id', 'weight_kg', 'height_cm', 'bmi', 'weight_clean', 'height_clean', 'bmi_clean'))
