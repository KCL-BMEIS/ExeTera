import csv
import time
from numba import njit,jit
import numpy as np
from exetera.core import utils


ESCAPE_VALUE = np.frombuffer(b'"', dtype='S1')[0][0]
SEPARATOR_VALUE = np.frombuffer(b',', dtype='S1')[0][0]
NEWLINE_VALUE = np.frombuffer(b'\n', dtype='S1')[0][0]
WHITE_SPACE_VALUE = np.frombuffer(b' ', dtype='S1')[0][0]
#CARRIAGE_RETURN_VALUE = np.frombuffer(b'\r', dtype='S1')[0][0]



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

        # f.seek(0)
        # print(f.read())
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
    with utils.Timer("my_fast_csv_reader"):
        content = np.fromfile(source, dtype=np.uint8)
        print(content)
        my_fast_csv_reader(content, column_inds, column_vals, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE)

    print('======after csv reader====')
    print(column_inds)
    print(column_vals)
    return column_inds, column_vals


def get_file_stat(source, chunk_size):
    with open(source, 'rb') as f:
        f.seek(0,2)
        total_byte_size = f.tell()
        print('total_byte_size', total_byte_size)

    with open(source) as f:
        header = csv.DictReader(f)
        count_columns = len(header.fieldnames)
        avg_line_length = len(f.readline()) * 10
    
    count_rows = max(chunk_size // count_columns * avg_line_length, 5)
    print('count_columns', count_columns, 'count_rows', count_rows)

    val_row_count = count_rows * avg_line_length
    val_threshold = int(count_rows * avg_line_length * 0.8)
    
    return total_byte_size, count_columns, count_rows, val_row_count, val_threshold


def read_file_using_fast_csv_reader(source, chunk_size, categorical_map_list, writer_list=None):

    total_byte_size, count_columns, count_rows, val_row_count, val_threshold = get_file_stat(source, chunk_size)

    with utils.Timer("my_fast_csv_reader"):
        chunk_index = 0
        hasHeader = True

        while chunk_index < total_byte_size:
            # initialize column_inds, column_vals
            column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
            column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)
            # print('====== initialize =====')
            # print(column_inds)
            # print(column_vals)

            # reads chunk size of file content
            content = np.fromfile(source, count=chunk_size, offset=chunk_index, dtype=np.uint8)
            length_content = content.shape[0]
            if length_content == 0:
                break

            # check if there's newline at EOF in the last chunk. add one if it's missing
            if chunk_index + length_content == total_byte_size and content[-1] != NEWLINE_VALUE:
                content = np.append(content, NEWLINE_VALUE)

            offset_pos, written_row_count = my_fast_csv_reader(content, column_inds, column_vals, hasHeader, val_threshold, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)
            # print('====== after csv reader =====')
            # print('chunk_index', chunk_index)
            # print('offset_pos', offset_pos)
            # print('written_row_count', written_row_count)
            print(column_inds)
            print(column_vals)

            chunk_index += offset_pos
            hasHeader = False

            chunk = None
            for i_c in range(count_columns):
                if categorical_map_list[i_c] is not None:
                    cat_keys, _, cat_index, cat_values = categorical_map_list[i_c]
                    print(cat_keys, cat_index, cat_values)

                    chunk = np.zeros(written_row_count, dtype=np.uint8)
                    my_fast_categorical_mapper(chunk, i_c, column_inds, column_vals, cat_keys, cat_index, cat_values)

                if writer_list and writer_list[i_c] and chunk is not None: 
                    writer_list[i_c].write_part(chunk)


@njit
def my_fast_csv_reader(source, column_inds, column_vals, hasHeader, val_threshold, escape_value, separator_value, newline_value, whitespace_value ):
    colcount = column_inds.shape[0]
    maxrowcount = column_inds.shape[1] - 1  # -1: minus the first 0 that created for prefix
    print('colcount', colcount)
    print('maxrowcount', maxrowcount)
    
    index = np.int64(0)
    index_for_end_line = np.int64(0)
    
    col_index = np.int64(0)
    row_index = np.int64(-1) if hasHeader else np.int64(0)

    escaped = False
    end_cell = False
    end_line = False
    escaped_literal_candidate = False

    cur_cell_char_count = np.int32(0)
    cur_cell_start = column_inds[col_index, row_index] if row_index >= 0 else 0
    index_for_cur_cell_start = np.int64(0)

    is_column_inds_full = False
    is_column_vals_full = False

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
            # \n \n - last line may have two newline_value
            while index + 1 < len(source) and source[index] + 1 == newline_value:
                index += 1

            if not escaped:
                end_cell = True
                end_line = True
                index_for_end_line = index
            else:
                write_char = True

        elif c == escape_value:
            # ,"... - start of an escaped cell
            # ...", - end of an escaped cell
            # ...""... - literal quote character
            # otherwise error
            if not escaped:
                if index != index_for_cur_cell_start:
                    raise Exception('double quote should start at the beginning of the cell')

                escaped = True
            else:   
                if escaped_literal_candidate:
                    write_char = True
                    escaped_literal_candidate = False
                elif index + 1 < len(source) and source[index + 1] == escape_value:
                    escaped_literal_candidate = True
                elif index + 1 < len(source) and (source[index + 1] == separator_value or source[index + 1] == newline_value):
                    escaped = False
                else:
                    raise Exception('invalid double quote')
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

                if cur_cell_char_count + cur_cell_char_count > val_threshold:
                    is_column_vals_full = True

            if end_line:
                row_index += 1
                col_index = 0
                # print('~~~~~~~~~~~')
                # print(col_index, row_index)
                # print('~~~~~~~~~~~')
                if row_index == maxrowcount:
                    is_column_inds_full = True

            else:
                col_index += 1

            cur_cell_start = column_inds[col_index, row_index]
            cur_cell_char_count = 0
            while index + 1 < len(source) and source[index + 1] == whitespace_value:
                index += 1
            index_for_cur_cell_start = index + 1

        index += 1

        if index == len(source) or is_column_inds_full or is_column_vals_full:
            next_pos = index_for_end_line + 1
            written_row_count = row_index

            return next_pos, written_row_count 


@njit           
def my_fast_categorical_mapper(chunk, i_c, column_ids, column_vals, cat_keys, cat_index, cat_values):
        
    for row_idx in range(len(column_ids[i_c]) - 1):
        if row_idx >= chunk.shape[0]:
            break

        key_start = column_ids[i_c, row_idx]
        key_end = column_ids[i_c, row_idx + 1]
        key_len = key_end - key_start

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


def get_byte_map(string_map):
    # sort by length of key first, and then sort alphabetically
    sorted_string_map = {k: v for k, v in sorted(string_map.items(), key=lambda item: item[0])}
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



if __name__ == "__main__":
    main()
