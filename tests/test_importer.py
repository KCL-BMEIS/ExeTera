import unittest
import json
import tempfile
import os
import h5py
from datetime import datetime, timezone
from exetera.core import importer
from exetera.core.load_schema import NewDataSchema

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
                }
            }
        }
    }
})



TEST_CSV_CONTENTS = '\n'.join((
    'name, id, age,birthday,  height, weight_change, BMI,  updated_at',
    'a,     1, 30,1990-01-01, 170.9,    21.2,        20.5, 2020-05-12 07:00:00',
    'b,     2, 40,1980-03-04, 180.2,        ,        25.4, 2020-05-13 01:00:00',
    'c,     3, 50,1970-04-05,      ,   -17.5,        27.2, 2020-05-14 03:00:00'
))

class TestImporter(unittest.TestCase):
    def setUp(self):
        self.fd_schema, self.schema_file_name = tempfile.mkstemp(suffix='.json')
        with open(self.schema_file_name, 'w') as fschema:
            fschema.write(TEST_SCHEMA)

        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(self.csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)

        self.files = {'schema_key': self.csv_file_name}


    def test_importer_with_arg_include(self):

        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include, exclude = {'schema_key': ['id', 'name']}, {}

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include, exclude)
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue(set(f['schema_key'].keys()) >= set(['id', 'name']))
            self.assertEqual(f['schema_key']['id']['values'].shape[0], 3)
        finally:
            os.close(fd_dest)

    def test_importer_with_wrong_arg_include(self):

        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include, exclude = {'schema_wrong_key': ['id', 'name']}, {}

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include, exclude)
        except Exception as e:
            self.assertEqual(str(e), "-n/--include: the following include table(s) are not part of any input files: {'schema_wrong_key'}")
        finally:
            os.close(fd_dest)
            

    def test_importer_with_arg_exclude(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        include, exclude = {}, {'schema_key':['updated_at']}

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, include, exclude)
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue('updated_at' not in set(f['schema_key'].keys()))
            self.assertEqual(f['schema_key']['id']['values'].shape[0], 3)
        finally:
            os.close(fd_dest)       

    def test_importer_without_create_day_field(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, {}, {})
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue('birthday' in set(f['schema_key'].keys()))            
            self.assertTrue('birthday_day' not in set(f['schema_key'].keys()))            

        finally:
            os.close(fd_dest)  

    def test_importer_with_create_day_field_True(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, {}, {})
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue('updated_at' in set(f['schema_key'].keys()))            
            self.assertTrue('updated_at_day' in set(f['schema_key'].keys()))            

        finally:
            os.close(fd_dest)  


    def test_numeric_importer_with_default_value(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, {}, {})
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertEqual(f['schema_key']['height']['values'].shape[0], 3)
            self.assertEqual(f['schema_key']['height']['values'][2], 160.5)
        finally:
            os.close(fd_dest)

    def test_numeric_importer_with_min_default_value(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, {}, {})
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertEqual(f['schema_key']['weight_change']['values'].shape[0], 3)
            self.assertEqual(f['schema_key']['weight_change']['values'][1], NewDataSchema._get_min_max('float32')[0])
        finally:
            os.close(fd_dest)


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

        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        
        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, files, False, {}, {})
        except Exception as e:
            self.assertEqual(str(e), "Numeric value in the field 'id' can not be empty in strict mode")
        finally:
            os.close(fd_dest)
        
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

        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')
        
        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, files, False, {}, {})
        except Exception as e:
            self.assertEqual(str(e), "The following numeric value in the field 'id' can not be parsed: 5@")
        finally:
            os.close(fd_dest)

    def test_numeric_importer_with_non_empty_valid_value_in_strict_mode(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, {}, {})
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue('id' in set(f['schema_key'].keys()))
            self.assertTrue('id_valid' not in set(f['schema_key'].keys()))
        finally:
            os.close(fd_dest)

    def test_numeric_importer_in_allow_empty_mode(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, {}, {})
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue('age' in set(f['schema_key'].keys()))
            self.assertTrue('age_valid' in set(f['schema_key'].keys()))
            self.assertTrue('weight_change' in set(f['schema_key'].keys()))
            self.assertTrue('weight_change_valid' not in set(f['schema_key'].keys()))            
        finally:
            os.close(fd_dest)

    def test_numeric_importer_in_relaxed_mode(self):
        ts = str(datetime.now(timezone.utc))
        fd_dest, dest_file_name = tempfile.mkstemp(suffix='.hdf5')

        try:
            importer.import_with_schema(ts, dest_file_name, self.schema_file_name, self.files, False, {}, {})
            f = h5py.File(dest_file_name, 'r')
            self.assertListEqual(list(f.keys()), ['schema_key'])
            self.assertTrue('height' in set(f['schema_key'].keys()))
            self.assertTrue('height_valid' not in set(f['schema_key'].keys()))
            self.assertTrue('height_valid_test' in set(f['schema_key'].keys()))
            self.assertTrue('BMI' in set(f['schema_key'].keys()))
            self.assertTrue('BMI_valid' in set(f['schema_key'].keys()))
        finally:
            os.close(fd_dest)

    def tearDown(self):
        os.close(self.fd_schema)
        os.close(self.fd_csv)

