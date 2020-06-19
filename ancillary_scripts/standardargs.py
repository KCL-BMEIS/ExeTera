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


def standard_args(use_patients=False, use_assessments=False, use_tests=False):
    import argparse
    parser = argparse.ArgumentParser()
    if use_patients:
        parser.add_argument('-p', '--patient_data',
                            help='the location and name of the patient data csv file')
    if use_assessments:
        parser.add_argument('-a', '--assessment_data',
                            help='the location and name of the test data csv file')
    if use_tests:
        parser.add_argument('-t', '--test_data',
                            help='the location and name of the test data csv file')

    parser.add_argument('-v', '--verbosity', default=1)
    parser.add_argument('-te', '--territories', default=None,
                        help='territories to include (None means include all)')

    return parser
