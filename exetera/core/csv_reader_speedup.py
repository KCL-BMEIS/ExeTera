import csv
import time
from numba import njit,jit
import numpy as np


class Timer:
    def __init__(self, start_msg, new_line=False, end_msg=''):
        print(start_msg + ': ' if new_line is False else '\n')
        self.end_msg = end_msg

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.end_msg + f' {time.time() - self.t0} seconds')


# def generate_test_arrays(count):
#     strings = [b'one', b'two', b'three', b'four', b'five', b'six', b'seven']
#     raw_values = np.random.RandomState(12345678).randint(low=1, high=7, size=count)
#     total_len = 0
#     for r in raw_values:
#         total_len += len(strings[r])
#     indices = np.zeros(count+1, dtype=np.int64)
#     values = np.zeros(total_len, dtype=np.int8)
#     for i_r, r in enumerate(raw_values):
#         indices[i_r+1] = indices[i_r] + len(strings[r])
#         for i_c in range(len(strings[r])):
#             values[indices[i_r]+i_c] = strings[r][i_c]
#
#     for i_r in range(20):
#         start, end = indices[i_r], indices[i_r+1]
#         print(values[start:end].tobytes())


def main():
    # generate_test_arrays(1000)
    col_dicts = [{'name': 'a', 'type': 'cat', 'vals': ('a', 'bb', 'ccc', 'dddd', 'eeeee')},
                 {'name': 'b', 'type': 'float'},
                 {'name': 'c', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False')},
                 {'name': 'd', 'type': 'float'},
                 {'name': 'e', 'type': 'float'},
                 {'name': 'f', 'type': 'cat', 'vals': ('', '', '', '', '', 'True', 'False')},
                 {'name': 'g', 'type': 'cat', 'vals': ('', '', '', '', 'True', 'False')},
                 {'name': 'h', 'type': 'cat', 'vals': ('', '', '', 'No', 'Yes')}]
    # make_test_data(100000, col_dicts)
    source = 'resources/assessment_input_small_data.csv'

    with Timer("Original csv reader took:"):
        original_csv_read(source)


    file_read_line_fast_csv(source)

    file_read_line_fast_csv(source)


# original csv reader
def original_csv_read(source, column_inds=None, column_vals=None):
    time0 = time.time()
    with open(source) as f:
        csvf = csv.reader(f, delimiter=',', quotechar='"')
        for i_r, row in enumerate(csvf):
            if i_r == 0:
                print(len(row))
            for i_c in range(len(row)):
                entry = row[i_c].encode()
                column_inds[i_c][i_r+1] = column_inds[i_c][i_r] + len(entry)
                column_vals[column_inds[i_c][i_r]:column_inds[i_c][i_r+1]] = entry

    # print('Original csv reader took {} s'.format(time.time() - time0))


# FAST open file read line
def file_read_line_fast_csv(source):

    with open(source) as f:
        header = csv.DictReader(f)
        count_columns = len(header.fieldnames)
        content = f.read()
        count_rows = content.count('\n') + 1

    content = np.fromfile(source, dtype='|S1')#np.uint8)
    column_inds = np.zeros((count_columns, count_rows), dtype=np.int64)
    column_vals = np.zeros((count_columns, count_rows * 25), dtype=np.uint8)


    ESCAPE_VALUE = np.frombuffer(b'"', dtype='S1')[0][0]
    SEPARATOR_VALUE = np.frombuffer(b',', dtype='S1')[0][0]
    NEWLINE_VALUE = np.frombuffer(b'\n', dtype='S1')[0][0]

    with Timer("my_fast_csv_reader_int"):

        content = np.fromfile(source, dtype=np.uint8)
        my_fast_csv_reader_int(content, column_inds, column_vals, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE)

    return column_inds, column_vals


def get_cell(row,col, column_inds, column_vals):
    start_row_index = column_inds[col][row]
    end_row_index = column_inds[col][row+1]
    return column_vals[col][start_row_index:end_row_index].tobytes()


def make_test_data(count, schema):
    """
    [ {'name':name, 'type':'cat'|'float'|'fixed', 'values':(vals)} ]
    """
    import pandas as pd
    rng = np.random.RandomState(12345678)
    columns = {}
    for s in schema:
        if s['type'] == 'cat':
            vals = s['vals']
            arr = rng.randint(low=0, high=len(vals), size=count)
            larr = [None] * count
            for i in range(len(arr)):
                larr[i] = vals[arr[i]]
            columns[s['name']] = larr
        elif s['type'] == 'float':
            arr = rng.uniform(size=count)
            columns[s['name']] = arr

    df = pd.DataFrame(columns)
    df.to_csv('/home/ben/covid/benchmark_csv.csv', index_label='index')


def make_test_data(count, schema):
    """
    [ {'name':name, 'type':'cat'|'float'|'fixed', 'values':(vals)} ]
    """
    import pandas as pd
    rng = np.random.RandomState(12345678)
    columns = {}
    for s in schema:
        if s['type'] == 'cat':
            vals = s['vals']
            arr = rng.randint(low=0, high=len(vals), size=count)
            larr = [None] * count
            for i in range(len(arr)):
                larr[i] = vals[arr[i]]
            columns[s['name']] = larr
        elif s['type'] == 'float':
            arr = rng.uniform(size=count)
            columns[s['name']] = arr

    df = pd.DataFrame(columns)
    df.to_csv('/home/ben/covid/benchmark_csv.csv', index_label='index')



@njit
def my_fast_csv_reader_int(source, column_inds, column_vals, escape_value, separator_value, newline_value):
    colcount = len(column_inds[0])

    index = np.int64(0)
    line_start = np.int64(0)
    cell_start_idx = np.int64(0)
    cell_end_idx = np.int64(0)
    col_index = np.int64(0)
    row_index = np.int64(-1)
    current_char_count = np.int32(0)

    escaped = False
    end_cell = False
    end_line = False
    escaped_literal_candidate = False
    cur_cell_start = column_inds[col_index, row_index] if row_index >= 0 else 0
    cur_cell_char_count = 0
    while True:
        write_char = False
        end_cell = False
        end_line = False

        c = source[index]
        if c == separator_value:
            if not escaped:
                end_cell = True
            else:
                write_char = True

        elif c == newline_value:
            if not escaped:
                end_cell = True
                end_line = True
            else:
                write_char = True
        elif c == escape_value:
            escaped = not escaped
        else:
            write_char = True

        if write_char and row_index >= 0:
            column_vals[col_index, cur_cell_start + cur_cell_char_count] = c
            cur_cell_char_count += 1

        if end_cell:
            if row_index >= 0:
                column_inds[col_index, row_index+1] = cur_cell_start + cur_cell_char_count
            if end_line:
                row_index += 1
                col_index = 0

            else:
                col_index += 1

            cur_cell_start = column_inds[col_index, row_index]
            cur_cell_char_count = 0

        index += 1

        if index == len(source):
            break

if __name__ == "__main__":
    main()
