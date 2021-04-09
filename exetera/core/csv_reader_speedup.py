import csv
import time
from numba import njit
import numpy as np


class Timer:
    def __init__(self, start_msg, new_line=False, end_msg='completed in'):
        print(start_msg, end=': ' if new_line is False else '\n')
        self.end_msg = end_msg

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.end_msg + f' {time.time() - self.t0} seconds')


def main():
    source = 'resources/assessment_input_small_data.csv' 

    original_csv_read(source)
    file_read_line_fast_csv(source)
    
    with Timer('Original csv reader took:'):
        original_csv_read(source)

    with Timer('FAST Open file read lines took:'):
        file_read_line_fast_csv(source)


# original csv reader
def original_csv_read(source):
    time0 = time.time()
    with open(source) as f:
        csvf = csv.reader(f, delimiter=',', quotechar='"')

        for i_r, row in enumerate(csvf):
            pass

    #print('Original csv reader took {} s'.format(time.time() - time0))

    
# FAST open file read line
def file_read_line_fast_csv(source):
    time0 = time.time()
    #input_lines = []
    with open(source) as f:
        header = csv.DictReader(f)
        content = f.read()

    index_excel = my_fast_csv_reader_int(content)

    for row in index_excel:
        for (s,e) in row:
            r = content[s:e]
            
    # print(excel)
    # print('FAST Open file read lines took {} s'.format(time.time() - time0))


@njit
def my_fast_csv_reader_int(source):
    ESCAPE_VALUE = '"' 
    SEPARATOR_VALUE = ','
    NEWLINE_VALUE = '\n'    

    index = np.int64(0)
    line_start = np.int64(0)
    cell_start_idx = np.int64(0)
    cell_end_idx = np.int64(0)
    col_index = np.int64(0)
    row_index = np.int64(0)

    fieldnames = None
    colcount = np.int64(0)
    row = [(0,0)]
    row.pop()
    excel = []

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
            row.append((cell_start_idx, cell_end_idx))

            cell_start_idx = cell_end_idx + 1

            col_index += 1

            if end_line and fieldnames is None and row is not None:
                fieldnames = row 
                colcount = len(row)  
    
            if col_index == colcount:
                if not end_line:
                    raise Exception('.....')
                else:
                    end_line = False
                    
                row_index += np.int64(1)
                col_index = np.int64(0)
                excel.append(row)
                row = [(0,0)]
                row.pop()
                #print(row)
                #print(excel)


        if index == len(source):
            # "erase the partially written line"
            return excel
            #return line_start




@njit
def my_fast_csv_reader_string(source, column_inds = None, column_vals = None):
    ESCAPE_VALUE = '"' 
    SEPARATOR_VALUE = ','
    NEWLINE_VALUE = '\n'    

    #colcount = len(column_inds)
    #max_rowcount = len(column_inds[0])-1

    index = np.int64(0)
    line_start = np.int64(0)
    cell_start = np.int64(0)
    cell_end = np.int64(0)
    col_index = np.int32(0)
    row_index = np.int32(0)

    fieldnames = None
    colcount = 0
    cell_value = ['']
    cell_value.pop()
    row = ['']
    row.pop()
    excel = []

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
                while index + 1 < len(source) and source[index + 1] == ' ':
                    index += 1
                
                cell_start = index + 1

            else:
                # write literal ','
                cell_value.append(c)

        elif c == NEWLINE_VALUE:
            if not escaped: #or escaped_literal_candidate:
                # don't write this char
                end_cell = True
                end_line = True  
                 
            else:
                # write literal '\n'
                cell_value.append(c)

        elif c == ESCAPE_VALUE:
            # ,"... - start of an escaped cell
            # ...", - end of an escaped cell
            # ...""... - literal quote character
            # otherwise error
            if not escaped:
                # this must be the first character of a cell
                if index != cell_start:
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
            cell_value.append(c)
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
            row.append(''.join(cell_value))
            cell_value = ['']
            cell_value.pop()

            col_index += 1  

            if end_line and fieldnames is None and row is not None:
                fieldnames = row 
                colcount = len(row)  


            if col_index == colcount:
                if not end_line:
                    raise Exception('.....')
                else:
                    end_line = False
                    
                row_index += 1
                col_index = 0
                excel.append(row)
                row = ['']
                row.pop()
                #if row_index == max_rowcount:
                    #return index

        if index == len(source):
            # "erase the partially written line"
            return excel
            #return line_start



if __name__ == "__main__":
    main()
