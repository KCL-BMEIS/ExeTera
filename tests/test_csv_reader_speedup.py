from unittest import TestCase
from exetera.core.csv_reader_speedup import my_fast_csv_reader, file_read_line_fast_csv, get_byte_map, my_fast_categorical_mapper, read_file_using_fast_csv_reader
import tempfile
import numpy as np
import os
import pandas as pd


TEST_SCHEMA = [{'name': 'a', 'type': 'cat', 'vals': ('','a', 'bb', 'ccc', 'dddd', 'eeeee'), 
                    'strings_to_values': {'':0,'a':1, 'bb':2, 'ccc':3, 'dddd':4, 'eeeee':5}},]
                # {'name': 'b', 'type': 'float'},
                # {'name': 'c', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False'),  
                #                              'strings_to_values': {"": 0, "False": 1, "True": 2}},
                # {'name': 'd', 'type': 'float'},
                # {'name': 'e', 'type': 'float'},
                # {'name': 
                # 'f', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False'),
                #                              'strings_to_values': {"": 0, "False": 1, "True": 2}},
                # {'name': 'g', 'type': 'cat', 'vals': ('', '', '', '', 'True', 'False'),
                #                              'strings_to_values': {"": 0, "False": 1, "True": 2}},
                # {'name': 'h', 'type': 'cat', 'vals': ('', '', '', 'No', 'Yes'),
                #                              'strings_to_values': {"": 0, "No": 1, "Yes": 2}}]


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


    def test_csv_fast_correctness(self):
        file_lines, chunk_size = 3, 10

        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        df, cat_columns_v, categorical_map_list = self._make_test_data(file_lines, TEST_SCHEMA, self.csv_file_name)

        count_columns = len(list(df))
        
        column_inds = np.zeros((count_columns, 30 + 1), dtype=np.int64) # add one more row for initial index 0
        column_vals = np.zeros((count_columns, 3 * 100), dtype=np.uint8)
        print('====== initialize =====')
        print(column_inds)
        print(column_vals)
        
        # fast csv reader reads chunk size of file content
        content = np.fromfile(self.csv_file_name, dtype=np.uint8)
        output = my_fast_csv_reader(content, column_inds, column_vals, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE)
        
        print(column_inds, column_vals)
        #my_fast_csv_reader(csv_file_name)



    def test_read_file_using_fast_csv_reader_file_lines_smaller_than_chunk_size(self):
        file_lines, chunk_size = 3, 10

        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        df, cat_columns_v, categorical_map_list, writer_list = self._make_test_data(file_lines, TEST_SCHEMA, self.csv_file_name)

        read_file_using_fast_csv_reader(source = self.csv_file_name, chunk_size=chunk_size, categorical_map_list = categorical_map_list, writer_list = writer_list)
        
        field_names = list(df)
        for i_c, field in enumerate(field_names):
            data = writer_list[i_c].data
            self.assertEqual(len(data), len(cat_columns_v[field]))
            self.assertListEqual(data, cat_columns_v[field])

        os.close(self.fd_csv)
        

    def test_read_file_using_fast_csv_reader_file_lines_larger_than_chunk_size(self):
        file_lines, chunk_size = 105, 10

        self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
        df, cat_columns_v, categorical_map_list, writer_list = self._make_test_data(file_lines, TEST_SCHEMA, self.csv_file_name)

        read_file_using_fast_csv_reader(source = self.csv_file_name, chunk_size=chunk_size, categorical_map_list = categorical_map_list, writer_list = writer_list)
        
        field_names = list(df)
        for i_c, field in enumerate(field_names):
            data = writer_list[i_c].data
            self.assertEqual(len(data), len(cat_columns_v[field]))
            self.assertListEqual(data, cat_columns_v[field])
            
        os.close(self.fd_csv)


    # def test_file_lines_smaller_than_chunk_size(self):
    #     file_lines, chunk_size = 3, 10

    #     # create temp csv file
    #     self.fd_csv, self.csv_file_name = tempfile.mkstemp(suffix='.csv')
    #     df, cat_columns_v, categorical_map_list = self._make_test_data(file_lines, TEST_SCHEMA, self.csv_file_name)

    #     column_inds, column_vals = file_read_line_fast_csv(self.csv_file_name)
        
    #     field_to_use = list(df)
    #     for i_c, field in enumerate(field_to_use):
    #         if categorical_map_list[i_c] is None:
    #             continue

    #         cat_keys, _, cat_index, cat_values = categorical_map_list[i_c]
    #         print(cat_keys, cat_index, cat_values)

    #         chunk = np.zeros(chunk_size, dtype=np.uint8)
    #         pos = my_fast_categorical_mapper(chunk, i_c, column_inds, column_vals, cat_keys, cat_index, cat_values)
    #         chunk = list(chunk[:pos])

    #         self.assertListEqual(chunk, cat_columns_v[field])

    #     # delete the temp csv file
    #     os.close(self.fd_csv)