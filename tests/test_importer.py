import unittest
import json
import tempfile
import os
import h5py
from datetime import datetime, timezone
from exetera.io import importer
from exetera.core import utils, session
import numpy as np
from io import BytesIO, StringIO

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
                'updated_at': {
                    'field_type': 'datetime',
                    'create_day_field': True
                },
                'updated_at_26': {
                    'field_type': 'datetime'
                },
                'updated_at_27': {
                    'field_type': 'datetime'
                },
                'updated_at_32': {
                    'field_type': 'datetime'
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
                "patient_id": {
                    "field_type": "fixed_string",
                    "length": 4
                },
                "degree": {
                    "field_type": "categorical",
                    "categorical": {
                        "value_type": "int8",
                        "strings_to_values": {
                            ""         : 0,
                            "bachelor" : 1,
                            "master"   : 2,
                            "doctor"   : 3,
                            },
                        "out_of_range": "freetext"    
                    }                
                }
            }
        }
    }
})



TEST_CSV_CONTENTS = '\n'.join((
    'name, id, age, birthday,  height, weight_change, BMI,  postcode, patient_id,   degree, updated_at, updated_at_26, updated_at_27, updated_at_32',
    'a,     1, 30, 1990-01-01, 170.9,    21.2,        20.5,      NW1,         E1, bachelor, 2020-05-12 07:00:00, 2022-02-03 12:00:00.12 UST, 2022-02-03 12:00:00.123 UST, 2022-02-03 12:00:01.123456+00.00',
    'bb,    2, 40, 1980-03-04, 180.2,        ,        25.4,     SW1P,       E123,   master, 2020-05-13 01:00:00, 2022-02-03 12:00:00.12 UST, 2022-02-03 12:00:00.123 UST, 2022-02-03 12:00:01.123456+00.00',
    'ccc,   3, 50, 1970-04-05,      ,   -17.5,        27.2,       E1,       E234,         , 2020-05-14 03:00:00, 2022-02-03 12:00:00.12 UST, 2022-02-03 12:00:00.123 UST, 2022-02-03 12:00:01.123456+00.00',
    'dddd,  4, 60, 1960-04-05,      ,   -17.5,        27.2,         ,           ,     prof, 2020-05-15 03:00:00, 2022-02-03 12:00:00.12 UST, 2022-02-03 12:00:00.123 UST, 2022-02-03 12:00:01.123456+00.00',
    'eeeee, 5, 70, 1950-04-05, 161.0,     2.5,        20.2,      NW3,    E456789,   doctor, 2020-05-16 03:00:00, 2022-02-03 12:00:00.12 UST, 2022-02-03 12:00:00.123 UST, 2022-02-03 12:00:01.123456+00.00',
))

class TestImporter(unittest.TestCase):
    def setUp(self):
        self.ts = str(datetime.now(timezone.utc))
        self.ds_name = 'test_ds'
        self.chunk_row_size = 100

        # csv file
        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(self.csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)
        self.files = {'schema_key': self.csv_file_name}

        # schema can use StringIO to replace csv file
        self.schema = StringIO(TEST_SCHEMA)


    def test_importer_with_arg_include(self):
        include, exclude = {'schema_key': ['id', 'name']}, {}

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        include,
                                        exclude,
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)

            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['id'].data[:].tolist(), [1,2,3,4,5])
            self.assertEqual(df['name'].data[:], ['a','bb','ccc','dddd','eeeee'])

        with h5py.File(bio, 'r') as hf:
            self.assertListEqual(list(hf.keys()), ['schema_key'])
            self.assertTrue(set(hf['schema_key'].keys()) >= set(['id', 'name']))
            self.assertEqual(hf['schema_key']['id']['values'][:].tolist(), [1,2,3,4,5])
            self.assertEqual(hf['schema_key']['name']['index'][:].tolist(), [0,1,3,6,10,15])

    def test_importer_with_wrong_arg_include(self):
        bio = BytesIO()
        include, exclude = {'schema_wrong_key': ['id', 'name']}, {}

        s = session.Session()
        with self.assertRaises(Exception) as context:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        include,
                                        exclude,
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)

        self.assertEqual(str(context.exception),
                         "-n/--include: the following include table(s) are not part of "
                         "any input files: {'schema_wrong_key'}")
                    

    def test_importer_with_arg_exclude(self):
        bio = BytesIO()
        include, exclude = {}, {'schema_key':['updated_at']}

        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        include,
                                        exclude,
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertTrue('updated_at' not in df)

        with h5py.File(bio, 'r') as hf:
            self.assertTrue('updated_at' not in set(hf['schema_key'].keys()))


    def test_importer_date(self):
        expected_birthday_date = ['1990-01-01', '1980-03-04', '1970-04-05', '1960-04-05', '1950-04-05']

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        {},
                                        {},
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['birthday'].data[:].tolist(), [datetime.strptime(x, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() for x in expected_birthday_date])

        with h5py.File(bio, 'r') as hf:
            self.assertEqual(hf['schema_key']['birthday']['values'][:].tolist(), [datetime.strptime(x, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() for x in expected_birthday_date])


    def test_importer_datetime_with_create_day_field(self):
        expected_updated_at_list = ['2020-05-12 07:00:00', '2020-05-13 01:00:00', '2020-05-14 03:00:00', '2020-05-15 03:00:00', '2020-05-16 03:00:00']
        expected_updated_at_date_list = [b'2020-05-12', b'2020-05-13', b'2020-05-14', b'2020-05-15', b'2020-05-16']

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        {},
                                        {},
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['updated_at'].data[:].tolist(), [datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() for x in expected_updated_at_list])
            self.assertEqual(df['updated_at_day'].data[:].tolist(), expected_updated_at_date_list )

        with h5py.File(bio, 'r') as hf:
            #print(hf['schema_key']['updated_at']['values'][:])
            self.assertAlmostEqual(hf['schema_key']['updated_at']['values'][:].tolist(), [datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() for x in expected_updated_at_list])
            self.assertEqual(hf['schema_key']['updated_at_day']['values'][:].tolist(), expected_updated_at_date_list)


    def test_numeric_field_importer_with_small_chunk_size(self):
        # numeric int field
        expected_age_list = list(np.array([30,40,50,60,70], dtype = np.int32 ))
        # numeric float field with default value
        expected_height_list = list(np.array([170.9,180.2,160.5,160.5,161.0], dtype = np.float32))
        # numeric float field with min_default_value
        expected_weight_change_list = list(np.array([21.2, utils.get_min_max('float32')[0], -17.5, -17.5, 2.5], dtype = np.float32))

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        {},
                                        {},
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['age'].data[:].tolist(), expected_age_list)
            self.assertEqual(df['height'].data[:].tolist(), expected_height_list)
            self.assertEqual(df['weight_change'].data[:].tolist(), expected_weight_change_list)

        with h5py.File(bio, 'r') as hf:
            self.assertListEqual(hf['schema_key']['age']['values'][:].tolist(), expected_age_list)
            self.assertListEqual(hf['schema_key']['height']['values'][:].tolist(), expected_height_list)
            self.assertListEqual(hf['schema_key']['weight_change']['values'][:].tolist(), expected_weight_change_list)


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
        with self.assertRaises(ValueError) as context:
            with session.Session() as s:
                importer.import_with_schema(s,
                                            bio,
                                            self.ds_name,
                                            self.schema,
                                            files,
                                            False,
                                            {},
                                            {},
                                            self.ts,
                                            chunk_row_size=self.chunk_row_size)

        self.assertEqual(str(context.exception), "Field 'id' contains values that cannot be converted to float in 'strict' mode")
        
        os.close(fd_csv)


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
        with self.assertRaises(ValueError) as context:
            with session.Session() as s:
                importer.import_with_schema(s,
                                            bio,
                                            self.ds_name,
                                            self.schema,
                                            files,
                                            False,
                                            {},
                                            {},
                                            self.ts,
                                            chunk_row_size=self.chunk_row_size)

        self.assertEqual(str(context.exception), "Field 'id' contains values that cannot be converted to float in 'strict' mode")

        os.close(fd_csv)


    def test_numeric_importer_with_non_empty_valid_value_in_strict_mode(self):
        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        {},
                                        {},
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['id'].data[:].tolist(), [1,2,3,4,5])
            self.assertTrue('id_valid' not in df)

        with h5py.File(bio, 'r') as hf:
            self.assertEqual(hf['schema_key']['id']['values'][:].tolist(), [1,2,3,4,5])  
            self.assertTrue('id_valid' not in set(hf['schema_key'].keys()))


    def test_numeric_importer_in_allow_empty_mode(self):
        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        {},
                                        {},
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['age_valid'].data[:].tolist(),  [True, True, True, True, True])
            self.assertTrue('weight_change_valid' not in df)

        with h5py.File(bio, 'r') as hf:
            self.assertTrue(hf['schema_key']['age']['values'][:].tolist(), [30,40,50,60,70])
            self.assertTrue(hf['schema_key']['age_valid']['values'][:].tolist(), [True, True, True, True, True])
            self.assertTrue('weight_change_valid' not in set(hf['schema_key'].keys()))            


    def test_numeric_importer_in_relaxed_mode(self):
        expected_height_list = list(np.asarray([170.9, 180.2, 160.5, 160.5, 161.0], dtype=np.float32))

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        None,
                                        None,
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['height'].data[:].tolist(), expected_height_list)
            self.assertTrue('height_valid' not in df)
            self.assertEqual(df['height_valid_test'].data[:].tolist(), [True, True, False, False, True])
            self.assertEqual(df['BMI_valid'].data[:].tolist(), [True, True, True, True, True])

        with h5py.File(bio, 'r') as hf:
            self.assertEqual(hf['schema_key']['height']['values'][:].tolist(), expected_height_list)
            self.assertTrue('height_valid' not in set(hf['schema_key'].keys()))
            self.assertTrue(hf['schema_key']['height_valid_test']['values'][:].tolist(), [True, True, False, False, True])
            self.assertTrue(hf['schema_key']['BMI']['values'][:].tolist(), [20.5, 25.4, 27.2, 27.2, 20.2])
            self.assertTrue(hf['schema_key']['BMI_valid']['values'][:].tolist(), [True, True, True, True, True])


    def test_indexed_string_importer_with_small_chunk_size(self):
        chunk_row_size = 20 # chunk_row_size * column_count < total_bytes

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        None,
                                        None,
                                        self.ts,
                                        chunk_row_size=chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['name'].data[:], ['a','bb','ccc','dddd','eeeee'])

        with h5py.File(bio, 'r') as hf:
            indices = hf['schema_key']['name']['index'][:]
            values = hf['schema_key']['name']['values'][:]

            self.assertListEqual(list(indices), [0,1,3,6,10,15])
            self.assertEqual(values[indices[0]:indices[1]].tobytes(), b'a')
            self.assertEqual(values[indices[3]:indices[4]].tobytes(), b'dddd')


    def test_categorical_field_importer_with_small_chunk_size(self):
        chunk_row_size = 20 # chunk_row_size * column_count < total_bytes

        expected_postcode_value_list = [1, 3, 2, 0, 4]
        expected_key_names = [b'', b'NW1', b'E1', b'SW1P', b'NW3']
        expected_key_values = [0,1,2,3,4]
        
        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        True,
                                        None,
                                        None,
                                        self.ts,
                                        chunk_row_size=chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['postcode'].data[:].tolist(), expected_postcode_value_list)
            self.assertEqual(list(df['postcode'].keys.values()), expected_key_names)

        with h5py.File(bio, 'r') as hf:
            self.assertEqual(hf['schema_key']['postcode']['values'][:].tolist(), expected_postcode_value_list)
            #self.assertEqual(hf['schema_key']['postcode']['key_names'][:].tolist(), expected_key_names)
            self.assertEqual(hf['schema_key']['postcode']['key_values'][:].tolist(), expected_key_values)


    def test_fixed_string_field_importer(self):
        expected_patient_id_value_list = [b'E1', b'E123', b'E234', b'', b'E456']

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        None,
                                        None,
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['patient_id'].data[:].tolist(), expected_patient_id_value_list)            

        with h5py.File(bio, 'r') as hf:
            self.assertEqual(hf['schema_key']['patient_id']['values'][:].tolist(), expected_patient_id_value_list)


    def test_leaky_categorical_field_importer(self):
        expected_degree_value_list = [1, 2, 0, -1, 3]
        expected_degree_freetext_index_list = [0, 0, 0, 0, 4, 4]
        expected_degree_freetext_value_list = list(np.frombuffer(b'prof', dtype = np.uint8))

        bio = BytesIO()
        with session.Session() as s:
            importer.import_with_schema(s,
                                        bio,
                                        self.ds_name,
                                        self.schema,
                                        self.files,
                                        False,
                                        None,
                                        None,
                                        self.ts,
                                        chunk_row_size=self.chunk_row_size)
            ds = s.get_dataset(self.ds_name)
            df = ds.get_dataframe('schema_key')
            self.assertEqual(df['degree'].data[:].tolist(), expected_degree_value_list)
            self.assertEqual(df['degree_freetext'].indices[:].tolist(), expected_degree_freetext_index_list)
            self.assertEqual(df['degree_freetext'].values[:].tolist(), expected_degree_freetext_value_list)

        with h5py.File(bio, 'r') as hf:
            self.assertEqual(list(hf['schema_key']['degree']['values'][:]), expected_degree_value_list)
            self.assertEqual(list(hf['schema_key']['degree_freetext']['index'][:]), expected_degree_freetext_index_list)
            self.assertEqual(list(hf['schema_key']['degree_freetext']['values'][:]), expected_degree_freetext_value_list)
        

    def tearDown(self):
        os.close(self.fd_csv)
