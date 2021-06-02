from unittest import TestCase
from exetera.core.csv_reader_speedup import fast_csv_reader, read_file_using_fast_csv_reader, get_file_stat
import tempfile
import numpy as np
import os
import pandas as pd
import exetera.core.operations as ops
from io import StringIO
import mock

TEST_SCHEMA = [{'name': 'a', 'type': 'cat', 'vals': ('','a', 'bb', 'ccc', 'dddd', 'eeeee'), 'field_size':5,
                                            'strings_to_values': {'':0,'a':1, 'bb':2, 'ccc':3, 'dddd':4, 'eeeee':5}},
                {'name': 'b', 'type': 'int', 'field_size':20,},
                {'name': 'c', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False'), 'field_size':5,
                                             'strings_to_values': {"": 0, "False": 1, "True": 2}},
                {'name': 'd', 'type': 'float', 'field_size':30},
                {'name': 'e', 'type': 'str', 'field_size':10, 'vals': ('hello world','python', 'KCL Biomedical and Image Science')},
                {'name': 'f', 'type': 'fixed_str', 'field_size':3, 'vals': ('aaa', 'bbb', 'ccc', 'ddd', 'eee'), 'length': 3},
                {'name': 'g', 'type': 'str', 'field_size':10, 'vals': ('numba', 'numpy', 'pandas', 'pytorch')},
                {'name': 'h', 'type': 'cat', 'field_size':3, 'vals': ('', '', '', 'No', 'Yes'),
                                                             'strings_to_values': {"": 0, "No": 1, "Yes": 2}}]


ESCAPE_VALUE = np.frombuffer(b'"', dtype='S1')[0][0]
SEPARATOR_VALUE = np.frombuffer(b',', dtype='S1')[0][0]
NEWLINE_VALUE = np.frombuffer(b'\n', dtype='S1')[0][0]
WHITE_SPACE_VALUE = np.frombuffer(b' ', dtype='S1')[0][0]


def _make_test_data(schema, count, chunk_row_size, fields_to_use=None):
    """
    [ {'name':name, 'type':'cat'|'float'|'fixed', 'values':(vals)} ]
    """
    rng = np.random.RandomState(12345678)
    columns = {}
    cat_map_dict = {}
    column_offsets = np.zeros(len(schema) + 1, dtype = np.int64)
    for i_s, s in enumerate(schema): 
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
            for x in arr:
                if len(str(x)) > 30:
                    print('float')
                    print(x)

        elif s['type'] == 'int':
            arr = rng.randint(10, size = count)
            columns[s['name']] = arr
        elif s['type'] == 'str' or s['type'] == 'fixed_str':
            vals = s['vals']
            arr = rng.randint(low=0, high=len(vals), size=count)
            str_arr = [None] * count
            for i in range(len(arr)):
                str_arr[i] = vals[arr[i]]
            columns[s['name']] = str_arr

        column_offsets[i_s + 1] = column_offsets[i_s] + s['field_size']*chunk_row_size

    # create csv file 
    df = pd.DataFrame(columns)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index = False)

    # create byte map for each categorical field 
    fieldnames = list(df)

    fields_to_use = fieldnames if fields_to_use is None else fields_to_use
    writer_list = [None] * len(fields_to_use)

    for i, fn in enumerate(fields_to_use):
        for s in schema:
            if s['name'] == fn:
                writer_list[i] = DummyWriter(s['type'])
                break

    # index map for field_to_use
    index_map = [fieldnames.index(k) for k in fields_to_use]

    return csv_buffer, df, writer_list, index_map, column_offsets


class DummyWriter:
    def __init__(self, field_type):
        self.type = field_type
        self.data = []
        self.result = []
    
    def transform_and_write_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        self.data.extend(ops.transform_to_values(column_inds, column_vals, column_offsets, col_idx, written_row_count))
            
        if self.type == 'int':
            self.result = [int(x.tobytes().decode('utf-8')) for x in self.data]
        elif self.type == 'float':
            self.result = [float(x.tobytes().decode('utf-8')) for x in self.data]
        else:
            self.result = [x.tobytes().decode('utf-8') for x in self.data]

    def flush(self):
        pass
 
class TestFastCSVReader(TestCase):
    
    def test_escape_well_formed_csv(self):
        open_csv = StringIO('id,f1,f2,f3\n1,"abc","a""b""c","""ab"""\n')
        content = np.frombuffer(open_csv.getvalue().encode(), dtype=np.uint8)
        
        chunk_row_size = 10
        _, count_columns, count_rows, _ = get_file_stat(open_csv, chunk_row_size)
        column_offsets = np.array([0, 1, 11, 21, 31], dtype = np.int64) * chunk_row_size

        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
        column_vals = np.zeros(column_offsets[-1], dtype=np.uint8)

        _, written_row_count, _, _, _ = fast_csv_reader(content, 0, column_inds, column_vals, column_offsets, True, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)
        self.assertEqual(written_row_count, 1)
        self.assertEqual(column_inds[1, 1], 3)  # abc
        self.assertEqual(column_inds[2, 1], 5)  # a"b"c
        self.assertEqual(column_inds[3, 1], 4)  # "ab"


    def test_escape_bad_formed_csv_1(self):
        open_csv = StringIO('id,f1\n1,abc"\n')
        content = np.frombuffer(open_csv.getvalue().encode(), dtype=np.uint8)

        chunk_row_size = 10
        _, count_columns, count_rows, _ = get_file_stat(open_csv, chunk_row_size)
        column_offsets = np.array([0, 1, 11], dtype = np.int64) * chunk_row_size

        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
        column_vals = np.zeros(column_offsets[-1], dtype=np.uint8)

        with self.assertRaises(Exception) as context:
            fast_csv_reader(content, 0, column_inds, column_vals, column_offsets, True, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)    
            
        self.assertEqual(str(context.exception), 'double quote should start at the beginning of the cell') 


    def test_escape_bad_formed_csv_2(self):
        open_csv = StringIO('id,f1\n1,"abc"de\n')
        content = np.frombuffer(open_csv.getvalue().encode(), dtype=np.uint8)

        chunk_row_size = 10
        _, count_columns, count_rows, _ = get_file_stat(open_csv, chunk_row_size)
        column_offsets = np.array([0, 1, 11], dtype = np.int64) * chunk_row_size

        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
        column_vals = np.zeros(column_offsets[-1], dtype=np.uint8)

        with self.assertRaises(Exception) as context:
            fast_csv_reader(content, 0, column_inds, column_vals, column_offsets, True, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)    
        
        self.assertEqual(str(context.exception), 'invalid double quote')   


    def test_csv_fast_correctness(self):
        file_lines, chunk_row_size = 3, 100

        csv_buffer, df, _, _, column_offsets = _make_test_data(TEST_SCHEMA, file_lines, chunk_row_size)
        content = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)
        
        _, count_columns, count_rows, _ = get_file_stat(csv_buffer, chunk_row_size)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros(column_offsets[-1], dtype=np.uint8)
        
        _, written_row_count, _, _, _ = fast_csv_reader(content, 0, column_inds, column_vals, column_offsets, True, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)

        self.assertEqual(written_row_count, 3)
        self.assertListEqual(list(column_inds[0][:written_row_count + 1]), [0, 3, 5, 9])
        self.assertListEqual(list(column_inds[1][:written_row_count + 1]), [0, 1, 2, 3])
        self.assertListEqual(list(column_vals[column_offsets[0]: column_offsets[0] + 9]), [99, 99, 99, 98, 98, 100, 100, 100, 100])
        self.assertListEqual(list(column_vals[column_offsets[1]: column_offsets[1] + 3]), [49, 48, 49])


    def test_fast_csv_reader_column_inds_full(self):
        
        def _make_column_inds_full_data(chunk_row_size):
            columns = {}
            count_row = chunk_row_size
            count_col = 3
            column_offsets = np.zeros(count_col + 1, dtype = np.int64)

            for i in range(count_col):
                fieldname = 'f' + str(i)
                columns[fieldname] = ['abcd'] + ['']*(count_row - 1)
                column_offsets[i + 1] = column_offsets[i] + 10 * chunk_row_size

            df = pd.DataFrame(columns)
            
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index = False)

            return csv_buffer, df, column_offsets


        chunk_row_size = 10

        csv_buffer, df, column_offsets = _make_column_inds_full_data(chunk_row_size)
        content = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)
        
        _, count_columns, count_rows, _ = get_file_stat(csv_buffer, chunk_row_size)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros( column_offsets[-1], dtype=np.uint8)

        _, _, is_indices_full, is_values_full, _ = fast_csv_reader(content, 0, column_inds, column_vals, column_offsets, True, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)

        self.assertTrue(is_indices_full)
        self.assertFalse(is_values_full)


    def test_fast_csv_reader_column_vals_full(self):
        def _make_column_val_full_data(chunk_row_size):
            columns = {}
            count_row = chunk_row_size
            columns['f1'] = ['a' for _ in range(count_row // 2)] + ['b'*1000 for _ in range(count_row // 2, count_row)]
            df = pd.DataFrame(columns)
            column_offsets = np.array([0, 10], dtype = np.int64) * chunk_row_size

            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index = False)

            return csv_buffer, df, column_offsets

        chunk_row_size = 100

        csv_buffer, df, column_offsets = _make_column_val_full_data(chunk_row_size)        
        content = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)
        
        _, count_columns, count_rows, _ = get_file_stat(csv_buffer, chunk_row_size)
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) 
        column_vals = np.zeros( column_offsets[-1], dtype=np.uint8)

        _, _, is_indices_full, is_values_full, val_full_col_idx = fast_csv_reader(content, 0, column_inds, column_vals, column_offsets, True, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)
        self.assertFalse(is_indices_full)
        self.assertTrue(is_values_full)
        self.assertEqual(val_full_col_idx, 0)


class TestReadFile(TestCase):

    @mock.patch("numpy.fromfile")
    def test_read_file_on_only_categorical_field(self, mock_fromfile):
        file_lines, chunk_row_size = 100, 100
        fields_to_use = [s['name'] for s in TEST_SCHEMA if s['type'] == 'cat']

        csv_buffer, df, writer_list, index_map, column_offsets = _make_test_data(TEST_SCHEMA, file_lines, chunk_row_size, fields_to_use)
        mock_fromfile.return_value = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)
        # print(csv_buffer.getvalue().encode())

        read_file_using_fast_csv_reader(csv_buffer, chunk_row_size, column_offsets, index_map, field_importer_list=writer_list)
        
        for ith, field in enumerate(fields_to_use):
            result = writer_list[ith].result
            # print(result)
            # print(df[field])
            self.assertEqual(len(result), len(df[field]))
            self.assertListEqual(result, list(df[field]))


    @mock.patch("numpy.fromfile")
    def test_read_file_on_only_indexed_string_field(self, mock_fromfile):
        file_lines, chunk_row_size = 50, 100
        fields_to_use = [s['name'] for s in TEST_SCHEMA if s['type'] == 'str']

        csv_buffer, df, writer_list, index_map, column_offsets = _make_test_data(TEST_SCHEMA, file_lines, chunk_row_size, fields_to_use)
        mock_fromfile.return_value = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)

        read_file_using_fast_csv_reader(csv_buffer, chunk_row_size, column_offsets, index_map, field_importer_list=writer_list)
        
        for ith, field in enumerate(fields_to_use):
            result = writer_list[ith].result
            self.assertEqual(len(result), len(df[field]))
            self.assertListEqual(result, list(df[field]))


    @mock.patch("numpy.fromfile")
    def test_read_file_on_only_fixed_string_field(self, mock_fromfile):
        file_lines, chunk_row_size = 50, 100
        fields_to_use = [s['name'] for s in TEST_SCHEMA if s['type'] == 'fixed_str']

        csv_buffer, df, writer_list, index_map, column_offsets = _make_test_data(TEST_SCHEMA, file_lines, chunk_row_size, fields_to_use)
        mock_fromfile.return_value = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)

        read_file_using_fast_csv_reader(csv_buffer, chunk_row_size, column_offsets, index_map, field_importer_list=writer_list)
        
        for ith, field in enumerate(fields_to_use):
            result = writer_list[ith].result
            self.assertEqual(len(result), len(df[field]))
            self.assertListEqual(result, list(df[field]))

    
    @mock.patch("numpy.fromfile")
    def test_read_file_on_only_numeric_field(self, mock_fromfile):
        file_lines, chunk_row_size = 100, 100
        fields_to_use = [s['name'] for s in TEST_SCHEMA if s['type'] in ('int', 'float')]

        csv_buffer, df, writer_list, index_map, column_offsets = _make_test_data(TEST_SCHEMA, file_lines, chunk_row_size, fields_to_use)
        mock_fromfile.return_value = np.frombuffer(csv_buffer.getvalue().encode(), dtype=np.uint8)

        read_file_using_fast_csv_reader(csv_buffer, chunk_row_size, column_offsets, index_map, field_importer_list=writer_list)

        for ith, field in enumerate(fields_to_use):
            result = writer_list[ith].result
            self.assertEqual(len(result), len(df[field]))
            self.assertListEqual(result, list(df[field]))


