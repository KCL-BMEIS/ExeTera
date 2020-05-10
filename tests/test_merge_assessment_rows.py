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

import unittest


from processing.assessment_merge import MergeAssessmentRows
from utils import concatenate_maybe_strs


class MockDataset:
    def __init__(self, **kwargs):
        self.fields_ = kwargs

    def row_count(self):
        return len(self.fields_['id'])

    def field_by_name(self, field_name):
        return self.fields_[field_name]


class TestMergeAssessmentRows(unittest.TestCase):


    def test_merge_assessment_rows(self):

        output_row_count = 4
        resulting_fields = {'id': [None] * output_row_count,
                            'patient_id': [None] * output_row_count,
                            'created_at': [None] * output_row_count,
                            'updated_at': [None] * output_row_count,
                            'treatment': [None] * output_row_count
                           }

        created_fields = dict()
        existing_field_indices = [('id', 0), ('patient_id', 1), ('created_at', 2),
                                  ('updated_at', 3), ('treatment', 4)]
        custom_field_aggregators = {'treatment': concatenate_maybe_strs}

        ids = ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah']
        patient_ids = ['za', 'za', 'zb', 'zb', 'zc', 'zc', 'zd', 'zd']
        created_ats = ['2020-04-01 08:00:00', '2020-04-01 12:00:00',
                       '2020-04-01 09:00:00', '2020-04-01 13:00:00',
                       '2020-04-01 10:00:00', '2020-04-01 14:00:00',
                       '2020-04-01 11:00:00', '2020-04-01 15:00:00']
        updated_ats = ['2020-04-01 08:00:00', '2020-04-01 12:00:00',
                       '2020-04-01 09:00:00', '2020-04-01 13:00:00',
                       '2020-04-01 10:00:00', '2020-04-01 14:00:00',
                       '2020-04-01 11:00:00', '2020-04-01 15:00:00']
        treatments = ['', '', 'x', '', '', 'y', 'x', 'y']

        source_fields = [ids, patient_ids, created_ats, updated_ats, treatments]

        # ds = MockDataset({'id': ids, 'patient_id': patient_ids,
        #                   'created_at': created_ats, 'updated_at': updated_ats,
        #                   'treatment': treatments})
        mar = MergeAssessmentRows([4],
                                  resulting_fields, created_fields,
                                  existing_field_indices, custom_field_aggregators)

        mar(source_fields, None, 0, 1)
        mar(source_fields, None, 2, 3)
        mar(source_fields, None, 4, 5)
        mar(source_fields, None, 6, 7)

        self.assertListEqual(resulting_fields['id'],
                             ['ab', 'ad', 'af', 'ah'])
        self.assertListEqual(resulting_fields['patient_id'],
                             ['za', 'zb', 'zc', 'zd'])
        self.assertListEqual(resulting_fields['created_at'],
                             ['2020-04-01 12:00:00', '2020-04-01 13:00:00',
                              '2020-04-01 14:00:00', '2020-04-01 15:00:00'])
        self.assertListEqual(resulting_fields['updated_at'],
                             ['2020-04-01 12:00:00', '2020-04-01 13:00:00',
                              '2020-04-01 14:00:00', '2020-04-01 15:00:00'])
        self.assertListEqual(resulting_fields['treatment'],
                             ['', 'x', 'y', 'x,y'])


        resulting_fields = {'id': [None] * output_row_count,
                            'patient_id': [None] * output_row_count,
                            'created_at': [None] * output_row_count,
                            'updated_at': [None] * output_row_count,
                            'treatment': [None] * output_row_count}
        treatments = ['', '', '', 'a,b', 'c,d', '', 'a,b', 'c,d']

        source_fields = [ids, patient_ids, created_ats, updated_ats, treatments]

        mar = MergeAssessmentRows([4],
                                  resulting_fields, created_fields,
                                  existing_field_indices, custom_field_aggregators)

        mar(source_fields, None, 0, 1)
        mar(source_fields, None, 2, 3)
        mar(source_fields, None, 4, 5)
        mar(source_fields, None, 6, 7)

        self.assertListEqual(resulting_fields['id'],
                             ['ab', 'ad', 'af', 'ah'])
        self.assertListEqual(resulting_fields['patient_id'],
                             ['za', 'zb', 'zc', 'zd'])
        self.assertListEqual(resulting_fields['created_at'],
                             ['2020-04-01 12:00:00', '2020-04-01 13:00:00',
                              '2020-04-01 14:00:00', '2020-04-01 15:00:00'])
        self.assertListEqual(resulting_fields['updated_at'],
                             ['2020-04-01 12:00:00', '2020-04-01 13:00:00',
                              '2020-04-01 14:00:00', '2020-04-01 15:00:00'])
        self.assertListEqual(resulting_fields['treatment'],
                             ['', '"a,b"', '"c,d"', '"a,b","c,d"'])
