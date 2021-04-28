from unittest import TestCase
from exetera.core.csv_reader_speedup import my_fast_csv_reader, file_read_line_fast_csv, get_byte_map, my_fast_categorical_mapper, read_file_using_fast_csv_reader, get_file_stat, \
                                            ESCAPE_VALUE,SEPARATOR_VALUE,NEWLINE_VALUE,WHITE_SPACE_VALUE
import tempfile
import numpy as np
import os
import pandas as pd


TEST_SCHEMA = [{'name': 'a', 'type': 'cat', 'vals': ('','a', 'bb', 'ccc', 'dddd', 'eeeee'), 
                    'strings_to_values': {'':0,'a':1, 'bb':2, 'ccc':3, 'dddd':4, 'eeeee':5}},
                {'name': 'b', 'type': 'int'},
                {'name': 'c', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False'),  
                                             'strings_to_values': {"": 0, "False": 1, "True": 2}},
                {'name': 'd', 'type': 'int'},
                {'name': 'e', 'type': 'int'},
                {'name': 'f', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False'),
                                             'strings_to_values': {"": 0, "False": 1, "True": 2}},
                {'name': 'g', 'type': 'cat', 'vals': ('', '', '', '', 'True', 'False'),
                                             'strings_to_values': {"": 0, "False": 1, "True": 2}},
                {'name': 'h', 'type': 'cat', 'vals': ('', '', '', 'No', 'Yes'),
                                             'strings_to_values': {"": 0, "No": 1, "Yes": 2}}]



ESCAPE_VALUE = np.frombuffer(b'"', dtype='S1')[0][0]
SEPARATOR_VALUE = np.frombuffer(b',', dtype='S1')[0][0]
NEWLINE_VALUE = np.frombuffer(b'\n', dtype='S1')[0][0]
WHITE_SPACE_VALUE = np.frombuffer(b' ', dtype='S1')[0][0]

class DummyWriter:
    def __init__(self):
        self.data = []
    
    def write_part(self, chunk):
        self.data.extend(chunk)


class TestFastCSVReader(TestCase):
    
    def _make_test_data(self, count, schema, csv_file_name):
        """
        [ {'name':name, 'type':'cat'|'float'|'fixed', 'values':(vals)} ]
        """
        import pandas as pd
        rng = np.random.RandomState(12345678)
        columns = {}
        cat_columns_v = {}
        cat_map_dict = {}
        for s in schema: 
            if s['type'] == 'cat':
                vals = s['vals']
                arr = rng.randint(low=0, high=len(vals), size=count)
                larr = [None] * count
                arr_str_to_val = [None] * count
                for i in range(len(arr)):
                    larr[i] = vals[arr[i]]
                    arr_str_to_val[i] = s['strings_to_values'][vals[arr[i]]]

                columns[s['name']] = larr
                cat_columns_v[s['name']] = arr_str_to_val
                cat_map_dict[s['name']] = s['strings_to_values']
            elif s['type'] == 'float':
                arr = rng.uniform(size=count)
                columns[s['name']] = arr
            elif s['type'] == 'int':
                arr = rng.randint(10, size = count)
                columns[s['name']] = arr

        # create csv file 
        df = pd.DataFrame(columns)
        df.to_csv(csv_file_name, index = False)

        # create byte map for each categorical field 
        fieldnames = list(df)
        categorical_map_list = [None] * len(fieldnames)
        writer_list = [None] * len(fieldnames)

        for i, fn in enumerate(fieldnames):
            if fn in cat_map_dict:
                string_map = cat_map_dict[fn]
                categorical_map_list[i] = get_byte_map(string_map)
            
            writer_list[i] = DummyWriter()

        return df, cat_columns_v, categorical_map_list, writer_list


    def test_escape_well_formed_csv(self):
        TEST_CSV_CONTENTS = '\n'.join((
            'id, f1, f2, f3',
            '1, "abc", "a""b""c", """ab"""'
        ))
        
        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)
            fcsv.write('\n')

        total_byte_size, count_columns, count_rows, val_row_count, val_threshold = get_file_stat(csv_file_name, chunk_size=10)

        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)
        
        content = np.fromfile(csv_file_name, dtype=np.uint8)    

        _, written_row_count = my_fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)
        self.assertEqual(written_row_count, 1)
        self.assertEqual(column_inds[1, 1], 3)
        self.assertEqual(column_inds[2, 1], 5)
        self.assertEqual(column_inds[3, 1], 4)

        os.close(fd_csv)


    def test_escape_bad_formed_csv_1(self):
        TEST_CSV_CONTENTS = '\n'.join((
            'id, f1 ',
            '1, abc"',
        ))
        
        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)

        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(csv_file_name, chunk_size=10)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)
        
        content = np.fromfile(csv_file_name, dtype=np.uint8)

        try:
            my_fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)
        except Exception as e:
            self.assertEqual(str(e), 'double quote should start at the beginning of the cell')
        finally:
            os.close(fd_csv)


    def test_escape_bad_formed_csv_2(self):
        TEST_CSV_CONTENTS = '\n'.join((
            'id, f1 ',
            '1, "abc"de',
        ))
        
        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        with open(csv_file_name, 'w') as fcsv:
            fcsv.write(TEST_CSV_CONTENTS)

        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(csv_file_name, chunk_size=10)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)
        
        content = np.fromfile(csv_file_name, dtype=np.uint8)

        try:
            my_fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)
        except Exception as e:
            self.assertEqual(str(e), 'invalid double quote')
        finally:
            os.close(fd_csv)       
                                

    def test_csv_fast_correctness(self):
        file_lines, chunk_size = 3, 100

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        df, _, _, _ = self._make_test_data(file_lines, TEST_SCHEMA, csv_file_name)

        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(csv_file_name, chunk_size=chunk_size)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)

        val_threshold = int(count_rows * 10 * 0.8)
        
        content = np.fromfile(csv_file_name, dtype=np.uint8)
        offset, written_row_count = my_fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)

        self.assertEqual(offset, 70)
        self.assertEqual(written_row_count, 3)
        
        self.assertListEqual(list(column_inds[0][:written_row_count + 1]), [0, 3, 5, 9])
        self.assertListEqual(list(column_inds[1][:written_row_count + 1]), [0, 1, 2, 3])
        self.assertListEqual(list(column_vals[0][:9]), [99, 99, 99, 98, 98, 100, 100, 100, 100])
        self.assertListEqual(list(column_vals[1][:3]), [49, 48, 49])

        os.close(fd_csv)


    def test_read_file_using_fast_csv_reader_file_lines_smaller_than_chunk_size(self):
        file_lines, chunk_size = 3, 100

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        df, cat_columns_v, categorical_map_list, writer_list = self._make_test_data(file_lines, TEST_SCHEMA, csv_file_name)

        read_file_using_fast_csv_reader(source = csv_file_name, chunk_size=chunk_size, categorical_map_list = categorical_map_list, writer_list = writer_list)
        
        field_names = list(df)
        for i_c, field in enumerate(field_names):
            if categorical_map_list[i_c] is not None:
                data = writer_list[i_c].data
                self.assertEqual(len(data), len(cat_columns_v[field]))
                self.assertListEqual(data, cat_columns_v[field])

        os.close(fd_csv)
        

    def test_read_file_using_fast_csv_reader_file_lines_larger_than_chunk_size(self):
        file_lines, chunk_size = 3, 100

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        df, cat_columns_v, categorical_map_list, writer_list = self._make_test_data(file_lines, TEST_SCHEMA, csv_file_name)

        read_file_using_fast_csv_reader(source = csv_file_name, chunk_size=chunk_size, categorical_map_list = categorical_map_list, writer_list = writer_list)
        
        field_names = list(df)
        for i_c, field in enumerate(field_names):
            if categorical_map_list[i_c] is not None:
                data = writer_list[i_c].data
                self.assertEqual(len(data), len(cat_columns_v[field]))
                self.assertListEqual(data, cat_columns_v[field])
            
        os.close(fd_csv)