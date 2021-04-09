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


def main():
    source = 'resources/assessment_input_small_data.csv' 
    print(source)
    # run once first
    original_csv_read(source)
    
    with Timer("Original csv reader took:"):
        original_csv_read(source)

    file_read_line_fast_csv(source)
    with Timer("FAST Open file read lines took"):
        file_read_line_fast_csv(source)
    


# original csv reader
def original_csv_read(source):
    time0 = time.time()
    with open(source) as f:
        csvf = csv.reader(f, delimiter=',', quotechar='"')

        for i_r, row in enumerate(csvf):
            pass

    # print('Original csv reader took {} s'.format(time.time() - time0))

    
# FAST open file read line
def file_read_line_fast_csv(source):

    with open(source) as f:
        header = csv.DictReader(f)
        count_columns = len(header.fieldnames)
        content = f.read()
        count_rows = content.count('\n') + 1

    content = np.fromfile(source, dtype='|S1')

    column_inds = np.zeros(count_rows * count_columns, dtype = np.int64).reshape(count_rows, count_columns)

    my_fast_csv_reader_int(content, column_inds)

    for row in column_inds:
        #print(row)
        for i, e in enumerate(row):
            pass


@njit
def my_fast_csv_reader_int(source, column_inds):
    ESCAPE_VALUE = b'"' 
    SEPARATOR_VALUE = b','
    NEWLINE_VALUE = b'\n'    

    #max_rowcount = len(column_inds) - 1
    colcount = len(column_inds[0])

    index = np.int64(0)
    line_start = np.int64(0)
    cell_start_idx = np.int64(0)
    cell_end_idx = np.int64(0)
    col_index = np.int64(0)
    row_index = np.int64(0)

    # how to parse csv
    # . " is the escape character
    # . fields that need to contain '"', ',' or '\n' must be quoted
    # . while escaped
    #   . ',' and '\n' are considered part of the field
    #   . i.e. a,"b,c","d\ne","f""g"""
    # . while not escaped
    #   . ',' ends the cell and starts a new cell
    #   . '\n' ends the cell and starts a new row
    #     . after the first row, we should check that subsequent rows have the same cell count
    escaped = False
    end_cell = False
    end_line = False
    escaped_literal_candidate = False
    while True:
        c = source[index]
        if c == SEPARATOR_VALUE:
            if not escaped: #or escaped_literal_candidate:
                # don't write this char
                end_cell = True  
                cell_end_idx = index
                # while index + 1 < len(source) and source[index + 1] == ' ':
                #     index += 1
                

            else:
                # write literal ','
                # cell_value.append(c)
                pass

        elif c == NEWLINE_VALUE:
            if not escaped: #or escaped_literal_candidate:
                # don't write this char
                end_cell = True
                end_line = True  
                cell_end_idx = index
            else:
                # write literal '\n'
                pass
                #cell_value.append(c)

        elif c == ESCAPE_VALUE:
            # ,"... - start of an escaped cell
            # ...", - end of an escaped cell
            # ...""... - literal quote character
            # otherwise error
            if not escaped:
                # this must be the first character of a cell
                if index != cell_start_idx:
                    # raise error!
                    pass
                # don't write this char
                else:
                    escaped = True
            else:

                escaped = False
                # if escaped_literal_candidate:
                #     escaped_literal_candidate = False
                #     # literal quote character confirmed, write it
                #     cell_value.append(c)
                # else:
                #     escaped_literal_candidate = True
                #     # don't write this char

        else:
            # cell_value.append(c)
            pass
        #     if escaped_literal_candidate:
        #         # error!
        #         pass
        #         # raise error return -2

        # parse c
        index += 1
          
        if end_cell:
            end_cell = False
            #column_inds[col_index][row_index+1] =\
            #    column_inds[col_index][row_index] + cell_end - cell_start
            column_inds[row_index][col_index] = cell_end_idx

            cell_start_idx = cell_end_idx + 1

            col_index += 1

    
            if col_index == colcount:
                if not end_line:
                    raise Exception('.....')
                else:
                    end_line = False
                    
                row_index += 1
                col_index = 0


        if index == len(source):
            # "erase the partially written line"
            return column_inds
            #return line_start



if __name__ == "__main__":
    main()
