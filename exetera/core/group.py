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




class Group:
    """
    Group / Field semantics
    -----------------------

    Location semantics
     * Fields can be created without a logical location. Such fields are written to a 'temp' location when required
     * Fields can be assigned a logical location or created with a logical location
     * Fields have a physical location at the point they are written to the dataset. Fields that are assigned to a logical
    location are also guaranteed to be written to a physical location
    """

    def __init__(self, parent):
        self.parent = parent

    def create_group(self, group_name):
        self.parent