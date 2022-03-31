from unittest import TestCase
import tempfile
import warnings
import os
from io import BytesIO, StringIO
import numpy as np
from datetime import datetime, timezone

from exetera.io.field_importers import Categorical, Numeric, String, DateTime, Date
from exetera.core import utils, session, operations as ops
from exetera.io import parsers
from .test_importer import TEST_SCHEMA, TEST_CSV_CONTENTS



class TestReadCSV(TestCase):

    def setUp(self):
        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(self.csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)

        self.schema_dict = {  
                        'name': String(),
                        'id': Numeric('int32', validation_mode='strict'),
                        'age': Numeric('int32'),
                        'height': Numeric('float32', invalid_value = 160.5, validation_mode='relaxed', flag_field_name= '_valid_test'),
                        'weight_change': Numeric('float32', invalid_value= 'min', create_flag_field= False),
                        'BMI': Numeric('float64', validation_mode='relaxed'),
                        'updated_at': DateTime(create_day_field= True),
                        'birthday': Date(create_day_field=True),
                        'postcode': Categorical(categories={"": 0, "NW1":1, "E1":2, "SW1P":3, "NW3":4}),
                        'patient_id': String(fixed_length=4),
                        'degree': Categorical(categories={"":0, "bachelor":1, "master":2, "doctor":3}, allow_freetext=True)
                    }
        
        self.schema_file = StringIO(TEST_SCHEMA)
        
    
    def test_read_csv_with_empty_schema_dict_and_file(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

        with self.assertRaises(Exception) as context:
            parsers.read_csv(self.csv_file_name, df)
        
        self.assertEqual(str(context.exception), "'schema_dict' and 'schema_file', one and only one of them should be provided.")


    def test_read_csv_with_non_emtpy_schema_dict_and_file(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

        with self.assertRaises(Exception) as context:
            parsers.read_csv(self.csv_file_name, df, self.schema_dict, self.schema_file)
        
        self.assertEqual(str(context.exception), "'schema_dict' and 'schema_file', one and only one of them should be provided.")


    def tearDown(self):
        os.close(self.fd_csv)


class TestSchemaDictionaryReadCSV(TestReadCSV):

    def test_read_csv_only_categorical_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, schema_dictionary = self.schema_dict, include=['postcode'])

            expected_postcode_value_list = [1, 3, 2, 0, 4]
            self.assertListEqual(df['postcode'].data[:].tolist(), expected_postcode_value_list)
            self.assertEqual(list(df['postcode'].keys.keys()), [0,1,2,3,4])
            self.assertEqual(list(df['postcode'].keys.values()), [b'', b'NW1', b'E1', b'SW1P', b'NW3'])


    def test_read_csv_only_leaky_categorical_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['degree'])

            expected_degree_value_list = [1, 2, 0, -1, 3]
            expected_degree_freetext_index_list = [0, 0, 0, 0, 4, 4]
            expected_degree_freetext_value_list = list(np.frombuffer(b'prof', dtype = np.uint8))
            self.assertEqual(df['degree'].data[:].tolist(), expected_degree_value_list)
            self.assertEqual(df['degree_freetext'].indices[:].tolist(), expected_degree_freetext_index_list)
            self.assertEqual(df['degree_freetext'].values[:].tolist(), expected_degree_freetext_value_list)
        

    def test_read_csv_only_numeric_int_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['id', 'age'])

            self.assertListEqual(df['id'].data[:].tolist(), [1,2,3,4,5])
            self.assertTrue('id_valid' not in df)

            self.assertListEqual(df['age'].data[:].tolist(), [30,40,50,60,70])
            self.assertListEqual(df['age_valid'].data[:].tolist(), [True, True, True, True, True])

    
    def test_read_csv_only_numeric_float_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['height', 'weight_change', 'BMI'])

            expected_height_list = list(np.asarray([170.9, 180.2, 160.5, 160.5, 161.0], dtype=np.float32))
            expected_height_valid_list = [True, True, False, False, True]
            self.assertEqual(list(df['height'].data[:]), expected_height_list)
            self.assertEqual(list(df['height_valid_test'].data[:]), expected_height_valid_list)

            expected_weight_change_list = list(np.asarray([21.2, utils.get_min_max('float32')[0], -17.5, -17.5, 2.5], dtype = np.float32))
            self.assertEqual(list(df['weight_change'].data[:]), expected_weight_change_list)
            self.assertTrue('weight_change_valid' not in df)

            expected_BMI_list = list(np.asarray([20.5, 25.4, 27.2, 27.2, 20.2], dtype=np.float64))
            expected_BMI_valid_list = [True, True, True, True, True]
            self.assertEqual(list(df['BMI'].data[:]), expected_BMI_list)
            self.assertEqual(list(df['BMI_valid'].data[:]), expected_BMI_valid_list)


    def test_read_csv_with_fields_out_of_order(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['weight_change', 'height', 'BMI'])

            expected_height_list = list(np.asarray([170.9, 180.2, 160.5, 160.5, 161.0], dtype=np.float32))
            expected_height_valid_list = [True, True, False, False, True]
            self.assertEqual(list(df['height'].data[:]), expected_height_list)
            self.assertEqual(list(df['height_valid_test'].data[:]), expected_height_valid_list)

            expected_weight_change_list = list(np.asarray([21.2, utils.get_min_max('float32')[0], -17.5, -17.5, 2.5], dtype = np.float32))
            self.assertEqual(list(df['weight_change'].data[:]), expected_weight_change_list)
            self.assertTrue('weight_change_valid' not in df)

            expected_BMI_list = list(np.asarray([20.5, 25.4, 27.2, 27.2, 20.2], dtype=np.float64))
            expected_BMI_valid_list = [True, True, True, True, True]
            self.assertEqual(list(df['BMI'].data[:]), expected_BMI_list)
            self.assertEqual(list(df['BMI_valid'].data[:]), expected_BMI_valid_list)

    
    def test_read_csv_only_indexed_string_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['name'])
            self.assertListEqual(df['name'].indices[:].tolist(), [0, 1, 3, 6, 10, 15])
            self.assertListEqual(df['name'].values[:].tolist(), [97, 98, 98, 99, 99, 99, 100, 100, 100, 100, 101, 101, 101, 101, 101])
            self.assertListEqual(df['name'].data[:], ['a', 'bb', 'ccc', 'dddd', 'eeeee'])


    def test_read_csv_only_fixed_string_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['patient_id'])

            expected_patient_id_value_list = [b'E1', b'E123', b'E234', b'', b'E456']
            self.assertListEqual(df['patient_id'].data[:].tolist(), expected_patient_id_value_list)


    def test_read_csv_only_datetime_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['updated_at'])    

            expected_updated_at_list = ['2020-05-12 07:00:00', '2020-05-13 01:00:00', '2020-05-14 03:00:00', '2020-05-15 03:00:00', '2020-05-16 03:00:00']
            expected_updated_at_date_list = [b'2020-05-12', b'2020-05-13', b'2020-05-14',b'2020-05-15',b'2020-05-16']
            self.assertEqual(df['updated_at'].data[:].tolist(), [datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() for x in expected_updated_at_list])
            self.assertEqual(df['updated_at_day'].data[:].tolist(),expected_updated_at_date_list )


    def test_read_csv_only_date_field(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, include=['birthday'])  

            expected_birthday_date = [b'1990-01-01', b'1980-03-04', b'1970-04-05', b'1960-04-05', b'1950-04-05']
            self.assertEqual(df['birthday'].data[:].tolist(), [datetime.strptime(x.decode(), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() for x in expected_birthday_date])
            self.assertEqual(df['birthday_day'].data[:].tolist(), expected_birthday_date)


    def test_read_csv_check_j_valid_from_to(self):
        bio = BytesIO()
        ts = datetime.now(timezone.utc).timestamp()

        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            parsers.read_csv(self.csv_file_name, df, self.schema_dict, timestamp=ts)  

            self.assertEqual(df['j_valid_from'].data[:].tolist(), [ts]*5)
            self.assertEqual(df['j_valid_to'].data[:].tolist(), [ops.MAX_DATETIME.timestamp()]*5)


    def test_read_csv_with_schema_missing_field(self):
        bio = BytesIO()
        with session.Session() as s:
            with warnings.catch_warnings(record = True ) as w:
                warnings.simplefilter("always")
                dst = s.open_dataset(bio, 'w', 'dst')
                df = dst.create_dataframe('df')
    
                missing_schema_dict = {'name': String()}
                parsers.read_csv(self.csv_file_name, df, missing_schema_dict)
                self.assertListEqual(df['id'].data[:], ['1','2','3','4','5']) 
                self.assertEqual([i.replace('\r', '') for i in df['updated_at'].data[:]],  # remove \r due to windows
                                 ['2020-05-12 07:00:00', '2020-05-13 01:00:00', '2020-05-14 03:00:00', '2020-05-15 03:00:00', '2020-05-16 03:00:00'])
                self.assertEqual(df['birthday'].data[:], ['1990-01-01', '1980-03-04', '1970-04-05', '1960-04-05', '1950-04-05'])
                self.assertEqual(df['postcode'].data[:], ['NW1', 'SW1P', 'E1', '', 'NW3'])
                self.assertEqual(len(w), 1)
        

class TestSchemaJsonFileReadCSV(TestReadCSV):
    
    def test_read_csv_all_fields(self):
        bio = BytesIO()
        with session.Session() as s:
            dst = s.open_dataset(bio, 'w', 'dst')
            df = dst.create_dataframe('df')

            #print('csv_file_name', self.csv_file_name)

            parsers.read_csv(self.csv_file_name, df, schema_file=self.schema_file)

            expected_postcode_value_list = [1, 3, 2, 0, 4]
            self.assertListEqual(df['postcode'].data[:].tolist(), expected_postcode_value_list)
            self.assertEqual(list(df['postcode'].keys.keys()), [0,1,2,3,4])
            self.assertEqual(list(df['postcode'].keys.values()), [b'', b'NW1', b'E1', b'SW1P', b'NW3'])

