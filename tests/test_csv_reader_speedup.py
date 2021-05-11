from unittest import TestCase
from exetera.core.csv_reader_speedup import fast_csv_reader, read_file_using_fast_csv_reader, get_file_stat
import tempfile
import numpy as np
import os
import pandas as pd
import exetera.core.operations as ops
from io import StringIO


TEST_SCHEMA = [{'name': 'a', 'type': 'cat', 'vals': ('','a', 'bb', 'ccc', 'dddd', 'eeeee'), 
                    'strings_to_values': {'':0,'a':1, 'bb':2, 'ccc':3, 'dddd':4, 'eeeee':5}},
                {'name': 'b', 'type': 'int'},
                {'name': 'c', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False'),  
                                             'strings_to_values': {"": 0, "False": 1, "True": 2}},
                {'name': 'd', 'type': 'float'},
                {'name': 'e', 'type': 'str', 'vals': ('hello world','python', 'KCL Biomedical and Image Science')},
                {'name': 'f', 'type': 'fixed_str', 'vals': ('aaa', 'bbb', 'ccc', 'ddd', 'eee'), 'length': 3},
                {'name': 'g', 'type': 'str', 'vals': ('numba', 'numpy', 'pandas', 'pytorch')},
                {'name': 'h', 'type': 'cat', 'vals': ('', '', '', 'No', 'Yes'),
                                             'strings_to_values': {"": 0, "No": 1, "Yes": 2}}]


ESCAPE_VALUE = np.frombuffer(b'"', dtype='S1')[0][0]
SEPARATOR_VALUE = np.frombuffer(b',', dtype='S1')[0][0]
NEWLINE_VALUE = np.frombuffer(b'\n', dtype='S1')[0][0]
WHITE_SPACE_VALUE = np.frombuffer(b' ', dtype='S1')[0][0]


class DummyWriter:
    def __init__(self, field_type):
        self.type = field_type
        self.data = []
        self.result = []
    
    def transform_and_write_part(self, column_inds, column_vals, col_idx, written_row_count):
        self.data.extend(ops.transform_to_values(column_inds, column_vals, col_idx, written_row_count))
            
        if self.type == 'int':
            self.result = [int(x.tobytes().decode('utf-8')) for x in self.data]
        elif self.type == 'float':
            self.result = [float(x.tobytes().decode('utf-8')) for x in self.data]
        else:
            self.result = [x.tobytes().decode('utf-8') for x in self.data]

    def flush(self):
        pass
        

 
class TestFastCSVReader(TestCase):
    
    def _make_test_data(self, count, schema, fields_to_use=None, csv_file_name=None):
        """
        [ {'name':name, 'type':'cat'|'float'|'fixed', 'values':(vals)} ]
        """
        rng = np.random.RandomState(12345678)
        columns = {}
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
                cat_map_dict[s['name']] = s['strings_to_values']
            elif s['type'] == 'float':
                arr = rng.uniform(size=count)
                columns[s['name']] = arr
            elif s['type'] == 'int':
                arr = rng.randint(10, size = count)
                columns[s['name']] = arr
            elif s['type'] == 'str':
                vals = s['vals']
                arr = rng.randint(low=0, high=len(vals), size=count)
                str_arr = [None] * count
                for i in range(len(arr)):
                    str_arr[i] = vals[arr[i]]
                columns[s['name']] = str_arr

        # create csv file 
        df = pd.DataFrame(columns)

        csv_buffer = StringIO()
        if csv_file_name:
            df.to_csv(csv_file_name, index=False)
        else:
            df.to_csv(csv_buffer, index = False)

        # create byte map for each categorical field 
        fieldnames = list(df)

        fields_to_use = fieldnames if fields_to_use is None else fields_to_use
        writer_list = [None] * len(fields_to_use)

        print('initial', fields_to_use)

        for i, fn in enumerate(fields_to_use):
            for s in schema:
                if s['name'] == fn:
                    writer_list[i] = DummyWriter(s['type'])
                    break

        # index map for field_to_use
        index_map = [fieldnames.index(k) for k in fields_to_use]

        return csv_buffer, df, writer_list, index_map


    def test_escape_well_formed_csv(self):
        open_csv = StringIO('id,f1,f2,f3\n1,"abc","a""b""c","""ab"""\n')
        content = np.frombuffer(open_csv.getvalue().encode(), dtype=np.uint8)

        total_byte_size, count_columns, count_rows, val_row_count, val_threshold = get_file_stat(open_csv, chunk_size=10)

        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)

        column_inds, _, _, written_row_count = fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)
        self.assertEqual(written_row_count, 1)
        self.assertEqual(column_inds[1, 1], 3)  # abc
        self.assertEqual(column_inds[2, 1], 5)  # a"b"c
        self.assertEqual(column_inds[3, 1], 4)  # "ab"


    def test_escape_bad_formed_csv_1(self):
        open_csv = StringIO('id,f1\n1,abc"\n')
        content = np.frombuffer(open_csv.getvalue().encode(), dtype=np.uint8)

        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(open_csv, chunk_size=10)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)

        with self.assertRaises(Exception) as context:
            fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)    
            self.assertEqual(str(context.exception), 'double quote should start at the beginning of the cell') 


    def test_escape_bad_formed_csv_2(self):
        open_csv = StringIO('id,f1\n1,"abc"de\n')
        content = np.frombuffer(open_csv.getvalue().encode(), dtype=np.uint8)

        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(open_csv, chunk_size=10)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)

        with self.assertRaises(Exception) as context:
            fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)    
            self.assertEqual(str(context.exception), 'invalid double quote')   

    
    def test_escape_bad_formed_csv_3(self):
        open_csv = StringIO('id,f1\n1, "abc\n')
        content = np.frombuffer(open_csv.getvalue().encode(), dtype=np.uint8)

        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(open_csv, chunk_size=10)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)

        with self.assertRaises(Exception) as context:
            fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)    
            self.assertEqual(str(context.exception), 'invalid double quote')   
                                

    def test_csv_fast_correctness(self):
        file_lines, chunk_size = 3, 100

        csv_buffer, df, _, _ = self._make_test_data(file_lines, TEST_SCHEMA)
        content = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)
        
        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(csv_buffer, chunk_size=chunk_size)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)
        
        column_inds, column_vals, offset, written_row_count = fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)

        self.assertEqual(written_row_count, 3)
        self.assertListEqual(list(column_inds[0][:written_row_count + 1]), [0, 3, 5, 9])
        self.assertListEqual(list(column_inds[1][:written_row_count + 1]), [0, 1, 2, 3])
        self.assertListEqual(list(column_vals[0][:9]), [99, 99, 99, 98, 98, 100, 100, 100, 100])
        self.assertListEqual(list(column_vals[1][:3]), [49, 48, 49])


    def test_fast_csv_reader_on_only_categorical_field(self):
        file_lines, chunk_size = 1003, 100
        fields_to_use = [s['name'] for s in TEST_SCHEMA if s['type'] == 'cat']

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        _, df, writer_list, index_map = self._make_test_data(file_lines, TEST_SCHEMA, fields_to_use, csv_file_name)
        
        read_file_using_fast_csv_reader(source=csv_file_name, chunk_size=chunk_size, index_map=index_map, field_importer_list=writer_list)

        for ith, field in enumerate(fields_to_use):
            result = writer_list[ith].result
            self.assertEqual(len(result), len(df[field]))
            self.assertListEqual(result, list(df[field]))

        os.close(fd_csv)


    def test_fast_csv_reader_on_only_indexed_string_field(self):
        file_lines, chunk_size = 53, 100
        fields_to_use = [s['name'] for s in TEST_SCHEMA if s['type'] == 'str']

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        _, df, writer_list, index_map = self._make_test_data(file_lines, TEST_SCHEMA, fields_to_use, csv_file_name)

        read_file_using_fast_csv_reader(source = csv_file_name, chunk_size=chunk_size, index_map=index_map, field_importer_list = writer_list)
        
        for ith, field in enumerate(fields_to_use):
            result = writer_list[ith].result
            self.assertEqual(len(result), len(df[field]))
            self.assertListEqual(result, list(df[field]))

        os.close(fd_csv) 

    
    def test_fast_csv_reader_on_only_numeric_field(self):
        file_lines, chunk_size = 998, 100
        fields_to_use = [s['name'] for s in TEST_SCHEMA if s['type'] in ('int', 'float')]

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        _, df, writer_list, index_map = self._make_test_data(file_lines, TEST_SCHEMA, fields_to_use, csv_file_name)

        read_file_using_fast_csv_reader(source=csv_file_name, chunk_size=chunk_size, index_map=index_map, field_importer_list=writer_list)

        for ith, field in enumerate(fields_to_use):
            result = writer_list[ith].result
            self.assertEqual(len(result), len(df[field]))
            self.assertListEqual(result, list(df[field]))

        os.close(fd_csv)


    def test_fast_csv_reader_chunksize_too_small(self):
        file_lines, chunk_size = 10, 10

        fd_csv, csv_file_name = tempfile.mkstemp(suffix='.csv')
        _, df, writer_list, index_map = self._make_test_data(file_lines, TEST_SCHEMA, fields_to_use=None, csv_file_name=csv_file_name)

        with self.assertRaises(Exception) as context:
            read_file_using_fast_csv_reader(source=csv_file_name, chunk_size=chunk_size, index_map=index_map, field_importer_list=writer_list)
            self.assertEqual(context.exception, 'The length of one line is too large, please modify the chunksize and make it larger')

        os.close(fd_csv)


    def _make_empty_test_data(self):
        columns = {}
        count_row = 10
        count_col = 3

        for i in range(count_col):
            fieldname = 'f' + str(i)
            columns[fieldname] = ['']*count_row

        df = pd.DataFrame(columns)
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index = False)

        index_map = [i for i in range(count_row)]
        return csv_buffer, df, index_map


    def test_fast_csv_reader_column_inds_full(self):
        chunk_size = 50

        csv_buffer, df, index_map  = self._make_empty_test_data()
        content = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)
        
        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(csv_buffer, chunk_size=chunk_size)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)

        indices, values, offset_pos, _ = fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)

        self.assertGreater(indices.shape[1], column_inds.shape[1])  # resized column_inds
        self.assertEqual(values.shape[1], column_vals.shape[1])     # column_val didn't resizes
        self.assertEqual(offset_pos, len(content))                  # whole chunk of content got read


    def _make_column_val_full_data(self):
        columns = {}
        count_row = 100
        columns['f1'] = [str(i) for i in range(2)] + [str(i*10000000) for i in range(2, count_row)]

        df = pd.DataFrame(columns)

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index = False)

        index_map = [i for i in range(count_row)]
        return csv_buffer, df, index_map


    def test_fast_csv_reader_column_vals_full(self):
        chunk_size = 1000

        csv_buffer, df, index_map = self._make_column_val_full_data()        
        content = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)
        
        total_byte_size, count_columns, count_rows,  val_row_count, val_threshold = get_file_stat(csv_buffer, chunk_size=chunk_size)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)

        indices, values, offset_pos, _ = fast_csv_reader(content, column_inds, column_vals, True, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)

        self.assertGreater(indices.shape[1], column_inds.shape[1])  # resized column_inds
        self.assertGreater(values.shape[1], column_vals.shape[1])   # resized column_vals
        self.assertEqual(offset_pos, len(content))                  # whole chunk of content got read

