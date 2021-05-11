import unittest
import json
import tempfile
import os
import h5py
from datetime import datetime, timezone
from exetera.core import importer
from exetera.core.load_schema import NewDataSchema
import numpy as np
from io import BytesIO

TEST_SCHEMA = json.dumps({
    "exetera": {
      "version": "1.1.0"
    },
    'schema': {
        'schema_key': {
            "primary_keys": [
                'id'
            ],
            "fields": {
                'name': {
                    'field_type': 'string'
                },
                'id': {
                    'field_type': 'numeric',
                    'value_type': 'int32',
                    'validation_mode': 'strict'
                },
                'age': {
                    'field_type': 'numeric',
                    'value_type': 'int32',                  
                },
                'height': {
                    'field_type': 'numeric',
                    'value_type': 'float32',
                    'invalid_value' : 160.5,
                    'validation_mode': 'relaxed',
                    'flag_field_name': '_valid_test'
                },
                'weight_change': {
                    'field_type': 'numeric',
                    'value_type': 'float32',
                    'invalid_value' : 'min',
                    'create_flag_field': False,
                },
                'BMI': {
                    'field_type': 'numeric',
                    'value_type': 'float64',
                    'validation_mode': 'relaxed'
                },
                'updated_at':{
                    'field_type': 'datetime',
                    'create_day_field': True
                },
                'birthday':{
                    'field_type':'date',
                },
                "postcode": {
                    "field_type": "categorical",
                    "categorical": {
                    "value_type": "int8",
                    "strings_to_values": {
                        ""    : 0,
                        "NW1" : 1,
                        "E1"  : 2,
                        "SW1P": 3,
                        "NW3" : 4,
                        }
                    }
                },
            }
        }
    }
})



TEST_CSV_CONTENTS = '\n'.join((
    'name, id, age, birthday,  height, weight_change, BMI,  postcode, updated_at',
    'a,     1, 30, 1990-01-01, 170.9,    21.2,        20.5,      NW1, 2020-05-12 07:00:00',
    'b,     2, 40, 1980-03-04, 180.2,        ,        25.4,     SW1P, 2020-05-13 01:00:00',
    'c,     3, 50, 1970-04-05,      ,   -17.5,        27.2,       E1, 2020-05-14 03:00:00',
    'd,     4, 60, 1960-04-05,      ,   -17.5,        27.2,         , 2020-05-15 03:00:00',
    'e,     5, 70, 1950-04-05, 161.0,     2.5,        20.2,      NW3, 2020-05-16 03:00:00',

))

class TestImporter(unittest.TestCase):
    def setUp(self):
        self.ts = str(datetime.now(timezone.utc))

        self.fd_schema, self.schema_file_name = tempfile.mkstemp(suffix='.json')
        with open(self.schema_file_name, 'w') as fschema:
            fschema.write(TEST_SCHEMA)

        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(self.csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)

        self.files = {'schema_key': self.csv_file_name}


    def test_importer_with_arg_include(self):
        include, exclude = {'schema_key': ['id', 'name']}, {}

        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, include, exclude)
        with h5py.File(bio, 'r') as hf:
            self.assertListEqual(list(hf.keys()), ['schema_key'])
            self.assertTrue(set(hf['schema_key'].keys()) >= set(['id', 'name']))
            self.assertEqual(hf['schema_key']['id']['values'].shape[0], 5)


    def test_importer_with_wrong_arg_include(self):
        bio = BytesIO()
        include, exclude = {'schema_wrong_key': ['id', 'name']}, {}

        with self.assertRaises(Exception) as context:
            importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, include, exclude)
            self.assertEqual(str(context.exception), "-n/--include: the following include table(s) are not part of any input files: {'schema_wrong_key'}")
                    

    def test_importer_with_arg_exclude(self):
        bio = BytesIO()
        include, exclude = {}, {'schema_key':['updated_at']}

        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, include, exclude)
        with h5py.File(bio, 'r') as hf:
            self.assertListEqual(list(hf.keys()), ['schema_key'])
            self.assertTrue('updated_at' not in set(hf['schema_key'].keys()))
            self.assertEqual(hf['schema_key']['id']['values'].shape[0], 5)


    def test_date_importer_without_create_day_field(self):     
        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {})
        with h5py.File(bio, 'r') as hf:
            self.assertTrue('birthday' in set(hf['schema_key'].keys()))  
            self.assertEqual(datetime.fromtimestamp(hf['schema_key']['birthday']['values'][1]).strftime("%Y-%m-%d"), '1980-03-04')
            
            self.assertTrue('birthday_day' not in set(hf['schema_key'].keys()))       


    def test_datetime_importer_with_create_day_field_True(self):
        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {})
        with h5py.File(bio, 'r') as hf:
            self.assertTrue('updated_at' in set(hf['schema_key'].keys()))                
            self.assertEqual(datetime.fromtimestamp(hf['schema_key']['updated_at']['values'][1]).strftime("%Y-%m-%d %H:%M:%S"), '2020-05-13 01:00:00')  

            self.assertTrue('updated_at_day' in set(hf['schema_key'].keys()))         
            self.assertEqual(hf['schema_key']['updated_at_day']['values'][1], b'2020-05-13')


    def test_numeric_field_importer_with_small_chunk_size(self):
        chunk_size = 1000

        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {})

        # numeric int field
        expected_age_list = list(np.array([30,40,50,60,70], dtype = np.int32 ))
        # numeric float field with default value
        expected_height_list = list(np.array([170.9,180.2,160.5,160.5,161.0], dtype = np.float32))
        # numeric float field with min_default_value
        expected_weight_change_list = list(np.array([21.2, NewDataSchema._get_min_max('float32')[0], -17.5, -17.5, 2.5], dtype = np.float32))

        with h5py.File(bio, 'r') as hf:
            self.assertListEqual(list(hf['schema_key']['age']['values'][:]), expected_age_list)
            self.assertListEqual(list(hf['schema_key']['height']['values'][:]), expected_height_list)
            self.assertListEqual(list(hf['schema_key']['weight_change']['values'][:]), expected_weight_change_list)


    def test_numeric_importer_with_empty_value_in_strict_mode(self):
        TEST_CSV_CONTENTS_EMPTY_VALUE = '\n'.join((
            'name, id',
            'a,     1',
            'c,     '
        ))

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS_EMPTY_VALUE)

        files = {'schema_key': csv_file_name}
        
        bio = BytesIO()
        with self.assertRaises(Exception) as context:
            importer.import_with_schema(self.ts, bio, self.schema_file_name, files, False, {}, {})
            self.assertEqual(str(context.exception), "Numeric value in the field 'id' can not be empty in strict mode")

        
    def test_numeric_importer_with_non_numeric_value_in_strict_mode(self):
        TEST_CSV_CONTENTS_EMPTY_VALUE = '\n'.join((
            'name, id',
            'a,     1',
            'c,     5@'
        ))

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS_EMPTY_VALUE)

        files = {'schema_key': csv_file_name}
        
        bio = BytesIO()
        with self.assertRaises(Exception) as context:
            importer.import_with_schema(self.ts, bio, self.schema_file_name, files, False, {}, {})
            self.assertEqual(str(context.exception), "The following numeric value in the field 'id' can not be parsed: 5@")
        


    def test_numeric_importer_with_non_empty_valid_value_in_strict_mode(self):
        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {})
        with h5py.File(bio, 'r') as hf:
            self.assertTrue('id' in set(hf['schema_key'].keys()))
            self.assertTrue('id_valid' not in set(hf['schema_key'].keys()))


    def test_numeric_importer_in_allow_empty_mode(self):
        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {})
        with h5py.File(bio, 'r') as hf:
            self.assertTrue('age' in set(hf['schema_key'].keys()))
            self.assertTrue('age_valid' in set(hf['schema_key'].keys()))
            self.assertTrue('weight_change' in set(hf['schema_key'].keys()))
            self.assertTrue('weight_change_valid' not in set(hf['schema_key'].keys()))            


    def test_numeric_importer_in_relaxed_mode(self):
        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {})
        with h5py.File(bio, 'r') as hf:
            self.assertTrue('height' in set(hf['schema_key'].keys()))
            self.assertTrue('height_valid' not in set(hf['schema_key'].keys()))
            self.assertTrue('height_valid_test' in set(hf['schema_key'].keys()))
            self.assertTrue('BMI' in set(hf['schema_key'].keys()))
            self.assertTrue('BMI_valid' in set(hf['schema_key'].keys()))


    def test_indexed_string_importer_with_small_chunk_size(self):
        chunk_size = 400 # < total_bytes

        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {}, chunk_size = chunk_size)
        with h5py.File(bio, 'r') as hf:
            indices = hf['schema_key']['name']['index'][:]
            values = hf['schema_key']['name']['values'][:]

        self.assertListEqual(list(indices), [0,1,2,3,4])
        self.assertEqual(values[indices[0]:indices[1]].tobytes(), b'a')


    def test_categorical_field_importer_with_small_chunk_size(self):
        chunk_size = 400 # < total_bytes
        
        bio = BytesIO()
        importer.import_with_schema(self.ts, bio, self.schema_file_name, self.files, False, {}, {}, chunk_size = chunk_size)
        with h5py.File(bio, 'r') as hf:
            expected_postcode_value_list = [1, 3, 2, 0, 4]
            self.assertEqual(list(hf['schema_key']['postcode']['values'][:]), expected_postcode_value_list)


    def tearDown(self):
        os.close(self.fd_schema)
        os.close(self.fd_csv)

