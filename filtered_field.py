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

def filtered_field(field, filter):
    for f in filter:
        yield field[f]


class FilteredField:
    def __init__(self, field, filter):
        self.field = field
        self.filter = filter
        self.dtype = self.field.dtype if isinstance(self.field, np.ndarray) else None

    def __getitem__(self, item):
        return self.field[self.filter[item]]

    def __len__(self):
        return len(self.filter)

