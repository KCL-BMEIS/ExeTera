import csv
import time
from numba import njit,jit
import numpy as np
from exetera.core import utils
import time
from collections import Counter
from io import StringIO


def get_file_stat(source, chunk_row_size):
    total_byte_size, count_columns = 0, 0

    if isinstance(source,str):
        with open(source) as f:
            f.seek(0,2)
            total_byte_size = f.tell()

            f.seek(0)
            header = csv.DictReader(f)  
            count_columns = len(header.fieldnames)              

    elif isinstance(source, StringIO):
        source.seek(0,2)
        total_byte_size = source.tell()

        source.seek(0)
        header = csv.DictReader(source)
        count_columns = len(header.fieldnames)     

    count_rows = chunk_row_size
    chunk_byte_size = count_rows * count_columns

    return total_byte_size, count_columns, count_rows, chunk_byte_size


def read_file_using_fast_csv_reader(source, chunk_row_size, column_offsets, index_map, field_importer_list=None, stop_after_rows=None):
    ESCAPE_VALUE = np.frombuffer(b'"', dtype='S1')[0][0]
    SEPARATOR_VALUE = np.frombuffer(b',', dtype='S1')[0][0]
    NEWLINE_VALUE = np.frombuffer(b'\n', dtype='S1')[0][0]
    WHITE_SPACE_VALUE = np.frombuffer(b' ', dtype='S1')[0][0]

    chunk_row_size *= 2
    time0 = time.time()

    total_byte_size, count_columns, count_rows, chunk_byte_size = get_file_stat(source, chunk_row_size)

    column_val_total_count = column_offsets[-1]

    with utils.Timer("read_file_using_fast_csv_reader"):
        chunk_index = 0
        hasHeader = True

        accumulated_written_rows = 0

        # initialize column_inds, column_vals ouside of while-loop
        column_inds = np.zeros((count_columns, count_rows + 1), dtype=np.int64) # add one more row for initial index 0
        
        # column_vals = np.zeros((count_columns, val_row_count), dtype=np.uint8)
        column_vals = np.zeros(np.int64(column_val_total_count), dtype=np.uint8)

        # make ndarray larger factor
        larger_factor = 2
        is_indices_full, is_values_full = False, False

        content = None
        start_index = 0

        ch = 0
        while chunk_index < total_byte_size:
            if stop_after_rows and accumulated_written_rows >= stop_after_rows:
                break

            # reads chunk size of file content 
            # when indices or values is full, we need to call fast_csv_reader again, but we don't want to read same content again
            if not is_indices_full and not is_values_full:
                content = np.fromfile(source, count=chunk_byte_size, offset=chunk_index, dtype=np.uint8)
                start_index = 0

                length_content = content.shape[0]
                if length_content == 0:
                    break

                # check if there's newline at EOF in the last chunk. add one if it's missing
                if chunk_index + length_content == total_byte_size and content[-1] != NEWLINE_VALUE:
                    content = np.append(content, NEWLINE_VALUE)
            
            offset_pos, written_row_count, is_indices_full, is_values_full, val_full_col_idx = fast_csv_reader(content, start_index, column_inds, column_vals, column_offsets, hasHeader, ESCAPE_VALUE, SEPARATOR_VALUE, NEWLINE_VALUE, WHITE_SPACE_VALUE)

            # convert and write
            for ith, i_c in enumerate(index_map):     
                if field_importer_list and field_importer_list[ith]: 
                    field_importer_list[ith].transform_and_write_part(column_inds, column_vals, column_offsets, i_c, written_row_count)

            # make column_inds larger if it gets full before reach the end of chunk
            if is_indices_full:
                indices_row_count = column_inds.shape[1] - 1
                column_inds = np.zeros((count_columns, np.uint32(indices_row_count * larger_factor + 1)), dtype=np.int64)  

            # make column_values larger if it gets full before reach the end of chunk             
            if is_values_full and val_full_col_idx != -1:
                col_val_count = column_offsets[val_full_col_idx + 1] - column_offsets[val_full_col_idx]
                delta = col_val_count * (larger_factor - 1)
                column_offsets = np.concatenate((column_offsets[:val_full_col_idx + 1], column_offsets[val_full_col_idx + 1:] + np.int64(delta)))
                column_val_total_count = column_offsets[-1]
                column_vals = np.zeros(np.int64(column_val_total_count), dtype=np.uint8)
                
            # reassign
            if is_indices_full or is_values_full:
                start_index = offset_pos
            else:
                chunk_index += offset_pos

            hasHeader = False
            accumulated_written_rows += written_row_count
            ch += 1
           
            print(f"{ch} chunks, {accumulated_written_rows} accumulated_written_rows parsed in {time.time() - time0}s")

        # flush at the end
        for ith in range(len(index_map)):
            field_importer_list[ith].flush()

    print(f"Total time {time.time() - time0}s")


@njit
def fast_csv_reader(source, start_index, column_inds, column_vals, column_offsets, hasHeader, escape_value, separator_value, newline_value, whitespace_value ):
    colcount = column_inds.shape[0]
    maxrowcount = np.int64(column_inds.shape[1] - 1)  # -1: minus the first element (0) in the row that created for prefix
    
    index = np.int64(start_index)
    index_for_end_line = np.int64(0)
    
    col_index = np.int64(0)
    row_index = np.int64(-1) if hasHeader else np.int64(0)
    val_full_col_idx = np.int64(-1)

    escaped = False
    end_cell = False
    end_line = False
    escaped_literal_candidate = False

    cur_cell_char_count = np.int64(0)
    cur_cell_start = column_inds[col_index, row_index] if row_index >= 0 else np.int64(0)
    
    index_for_cur_cell_start = np.int64(0)

    is_column_inds_full = False
    is_column_vals_full = False

    col_offset = np.int64(0)
    col_val_count = column_offsets[1]

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
            # while index + 1 < len(source) and source[index + 1] == newline_value:
            #     index += 1

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
                elif index + 1 == len(source):
                    # reach the end of source, will retry in next chunk
                    pass
                else:
                    raise Exception('invalid double quote')
        else:
            write_char = True

        if write_char and row_index >= 0:
            column_vals[col_offset + cur_cell_start + cur_cell_char_count] = c
            cur_cell_char_count += 1

            if cur_cell_start + cur_cell_char_count >= col_val_count:
                is_column_vals_full = True
                val_full_col_idx = col_index
                                
        if end_cell:
            if row_index >= 0:
                column_inds[col_index, row_index + 1] = cur_cell_start + cur_cell_char_count              
            if end_line:
                row_index += 1
                col_index = 0
                col_offset = column_offsets[col_index]
                col_val_count = column_offsets[col_index + 1] - col_offset

                if row_index == maxrowcount:
                    is_column_inds_full = True
            else:
                col_index += 1
                col_offset = column_offsets[col_index]
                col_val_count = column_offsets[col_index + 1] - col_offset

            cur_cell_start = column_inds[col_index, row_index]
            cur_cell_char_count = 0
            while index + 1 < len(source) and source[index + 1] == whitespace_value:
                index += 1
            index_for_cur_cell_start = index + 1

        index += 1

        if index == len(source) or is_column_inds_full or is_column_vals_full: 
            next_pos = index_for_end_line + 1
            written_row_count = row_index
            return next_pos, written_row_count, is_column_inds_full, is_column_vals_full, val_full_col_idx
     

    
