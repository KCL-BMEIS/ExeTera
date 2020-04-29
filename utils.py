import numpy as np
from numba import jit, prange


def count_flag_set(flags, flag_to_test):
    count = 0
    for f in flags:
        if f & flag_to_test:
            count += 1
    return count


def timestamp_to_day(field):
    if field == '':
        return ''
    return f'{field[0:4]}-{field[5:7]}-{field[8:10]}'
