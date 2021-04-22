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

        count_rows = sum(1 for _ in f) # w/o header row

        f.seek(0)
        print(f.read())
        # count_rows = content.count('\n') + 1  # +1: for the case that last line doesn't have \n
        
    column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
    # change it to longest key 
    column_vals = np.zeros((count_columns, count_rows * 100), dtype=np.uint8)

    print('====initialize=====')
    print(column_inds, column_vals)

    ESCAPE_VALUE = np.frombuffer(b'"', dtype='S1')[0][0]
    SEPARATOR_VALUE = np.frombuffer(b',', dtype='S1')[0][0]
    NEWLINE_VALUE = np.frombuffer(b'\n', dtype='S1')[0][0]

    #print(lineterminator.tobytes())
    #print("hello")
    #CARRIAGE_RETURN_VALUE = np.frombuffer(b'\r', dtype='S1')[0][0]
    # print("test")
    with Timer("my_fast_csv_reader"):
        content = np.fromfile(source, dtype=np.uint8)
        print(content)
        my_fast_csv_reader(content, column_inds, column_vals, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE)

    print('======after csv reader====')
    print(column_inds)
    print(column_vals)
    return column_inds, column_vals


@njit
def my_fast_csv_reader(source, column_inds, column_vals, escape_value, separator_value, newline_value):
    colcount = len(column_inds)
    maxrowcount = len(column_inds[0]) - 1  # minus extra index 0 row that created for column_inds
    print('colcount', colcount)
    print('maxrowcount', maxrowcount)
    
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

        elif c == newline_value :
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
                column_inds[col_index, row_index + 1] = cur_cell_start + cur_cell_char_count
                # print("========")
                # print(col_index, row_index + 1, column_vals.shape)
                # print(column_inds)
                # print(column_vals)
                # print("========")
            if end_line:
                row_index += 1
                col_index = 0
                # print('~~~~~~~~~~~')
                # print(col_index, row_index)
                # print('~~~~~~~~~~~')
            else:
                col_index += 1

            cur_cell_start = column_inds[col_index, row_index]
            cur_cell_char_count = 0

        index += 1

        if index == len(source):
            if col_index == colcount - 1: #and row_index == maxrowcount - 1:
                column_inds[col_index, row_index + 1] = cur_cell_start + cur_cell_char_count

            # print('source', source, 'len_source', len(source))
            # print('index', cur_cell_start + cur_cell_char_count)
            # print('break', col_index, row_index)
            break


@njit           
def my_fast_categorical_mapper(chunk, i_c, column_ids, column_vals, cat_keys, cat_index, cat_values):
    pos = 0
    for row_idx in range(len(column_ids[i_c]) - 1):
        # Finds length, which we use to lookup potential matches
        key_start = column_ids[i_c, row_idx]
        key_end = column_ids[i_c, row_idx + 1]
        key_len = key_end - key_start

        print('key_start', key_start, 'key_end', key_end)

        for i in range(len(cat_index) - 1):
            sc_key_len = cat_index[i + 1] - cat_index[i]

            if key_len != sc_key_len:
                continue

            index = i
            for j in range(key_len):
                entry_start = cat_index[i]
                if column_vals[i_c, key_start + j] != cat_keys[entry_start + j]:
                    index = -1
                    break

            if index != -1:
                chunk[row_idx] = cat_values[index]

    pos = row_idx + 1
    return pos


def get_byte_map(string_map):
    # sort by length of key first, and then sort alphabetically
    sorted_string_map = {k: v for k, v in sorted(string_map.items(), key=lambda item: (len(item[0]), item[0]))}
    sorted_string_key = [(len(k), np.frombuffer(k.encode(), dtype=np.uint8), v) for k, v in sorted_string_map.items()]
    sorted_string_values = list(sorted_string_map.values())
    
    # assign byte_map_key_lengths, byte_map_value
    byte_map_key_lengths = np.zeros(len(sorted_string_map), dtype=np.uint8)
    byte_map_value = np.zeros(len(sorted_string_map), dtype=np.uint8)

    for i, (length, _, v)  in enumerate(sorted_string_key):
        byte_map_key_lengths[i] = length
        byte_map_value[i] = v

    # assign byte_map_keys, byte_map_key_indices
    byte_map_keys = np.zeros(sum(byte_map_key_lengths), dtype=np.uint8)
    byte_map_key_indices = np.zeros(len(sorted_string_map)+1, dtype=np.uint8)
    
    idx_pointer = 0
    for i, (_, b_key, _) in enumerate(sorted_string_key):   
        for b in b_key:
            byte_map_keys[idx_pointer] = b
            idx_pointer += 1

        byte_map_key_indices[i + 1] = idx_pointer  

    byte_map = [byte_map_keys, byte_map_key_lengths, byte_map_key_indices, byte_map_value]
    return byte_map



def get_cell(row,col, column_inds, column_vals):
    start_row_index = column_inds[col][row]
    end_row_index = column_inds[col][row+1]
    return column_vals[col][start_row_index:end_row_index].tobytes()


if __name__ == "__main__":
    main()
