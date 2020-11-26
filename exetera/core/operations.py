from datetime import datetime
import numpy as np
from numba import jit, njit
import numba
from numba.typed import List

from exetera.core import validation as val
from exetera.core import fields, utils

DEFAULT_CHUNKSIZE = 1 << 20
INVALID_INDEX = 1 << 62
MAX_DATETIME = datetime(year=9999, month=1, day=1) #.timestamp()


def chunks(length, chunksize=1 << 20):
    cur = 0
    while cur < length:
        next_ = min(length, cur + chunksize)
        yield cur, next_
        cur = next_


def safe_map(field, map_field, map_filter, empty_value=None):
    if isinstance(field, fields.Field):
        if isinstance(field, fields.IndexedStringField):
            return safe_map_indexed_values(
                field.indices[:], field.values[:], map_field, map_filter, empty_value)
        else:
            return safe_map_values(field.data[:], map_field, map_filter, empty_value)
    elif isinstance(field, np.ndarray):
        return safe_map_values(field, map_field, map_filter, empty_value)


@njit
def safe_map_indexed_values(data_indices, data_values, map_field, map_filter, empty_value=None):
    empty_value_len = 0 if empty_value is None else len(empty_value)
    value_length = 0
    for i in range(len(map_field)):
        if map_filter[i]:
            value_length += data_indices[map_field[i]+1] - data_indices[map_field[i]]
        else:
            value_length += empty_value_len

    i_result = np.zeros(len(map_field) + 1, dtype=data_indices.dtype)
    v_result = np.zeros(value_length, dtype=data_values.dtype)

    offset = 0
    i_result[0] = 0
    for i in range(len(map_field)):
        if map_filter[i]:
            sst = data_indices[map_field[i]]
            sse = data_indices[map_field[i]+1]
            dst = offset
            delta = sse - sst
            dse = offset + delta
            i_result[i+1] = dse
            v_result[dst:dse] = data_values[sst:sse]
            offset += delta
        else:
            dst = offset
            dse = offset + empty_value_len
            i_result[i+1] = dse
            v_result[dst:dse] = empty_value
            offset += dse - dst

    return i_result, v_result


@njit
def safe_map_values(data_field, map_field, map_filter, empty_value=None):
    result = np.zeros_like(map_field, dtype=data_field.dtype)
    empty_val = result[0] if empty_value is None else empty_value
    for i in range(len(map_field)):
        if map_filter[i]:
            result[i] = data_field[map_field[i]]
        else:
            result[i] = empty_val
    return result


@njit
def map_valid(data_field, map_field, result=None):
    if result is None:
        result = np.zeros_like(map_field, dtype=data_field.dtype)
    for i in range(len(map_field)):
        if map_field[i] < INVALID_INDEX:
            result[i] = data_field[map_field[i]]
    return result


def ordered_map_valid_stream(data_field, map_field, result_field, chunksize=DEFAULT_CHUNKSIZE):
    df_it = iter(chunks(len(data_field.data), chunksize=chunksize))
    mf_it = iter(chunks(len(map_field.data), chunksize=chunksize))
    df_range = next(df_it)
    mf_range = next(mf_it)
    dfc = data_field.data[df_range[0]:df_range[1]]
    mfc = map_field.data[mf_range[0]:mf_range[1]]

    is_field_parameter = val.is_field_parameter(result_field)
    result_dtype = result_field.data.dtype if is_field_parameter else result_field.dtype
    rslt = np.zeros(chunksize, dtype=result_dtype)

    m = 0
    d = 0
    while m < len(map_field.data):
        mm, dd = ordered_map_valid_partial(df_range[0], dfc, mfc, rslt)
        if mm > 0:
            if is_field_parameter:
                result_field.data.write(rslt[:mm])
            else:
                result_field[m:m + mm] = rslt[:mm]
            rslt[:] = 0
        m += mm
        if m == mf_range[1] and m < len(map_field.data):
            mf_range = next(mf_it)
            mfc = map_field.data[mf_range[0]:mf_range[1]]
        else:
            mfc = mfc[mm:]
        if dd >= df_range[1] and dd < len(data_field.data):
            df_range = next(df_it)
            dfc = data_field.data[df_range[0]:df_range[1]]
        # else:
        #     dfc = dfc[dd:]


# 0 2 3 4 5 7 8 9 11 12 14 15 17 18 19
# 0 0 0 0 1 1 1 1  2  2  2  2  3  3  3
@njit
def ordered_map_valid_partial(d, data_field, map_field, result):
    i = 0
    while True:
        val = map_field[i]
        if val < INVALID_INDEX:
            if val >= d + len(data_field):
                # need a new data_field chunk
                return i, val
            result[i] = data_field[val - d]
        i += 1
        if i >= len(map_field):
            return i, val


@njit
def data_iterator(data_field, chunksize=1 << 20):
    cur = np.int64(0)
    chunks_ = chunks(len(data_field.data), chunksize)
    for c in chunks_:
        start = c[0]
        data = data_field.data[start:start + chunksize * 2]
        for v in range(c[0], c[1]):
            yield data[v]


@njit
def apply_filter_to_index_values(index_filter, indices, values):
    # pass 1 - determine the destination lengths
    cur_ = indices[:-1]
    next_ = indices[1:]
    count = 0
    total = 0
    for i in range(len(index_filter)):
        if index_filter[i] == True:
            count += 1
            total += next_[i] - cur_[i]
    dest_indices = np.zeros(count+1, indices.dtype)
    dest_values = np.zeros(total, values.dtype)
    dest_indices[0] = 0
    count = 1
    total = 0
    for i in range(len(index_filter)):
        if index_filter[i] == True:
            n = next_[i]
            c = cur_[i]
            delta = n - c
            dest_values[total:total + delta] = values[c:n]
            total += delta
            dest_indices[count] = total
            count += 1
    return dest_indices, dest_values


@njit
def apply_indices_to_index_values(indices_to_apply, indices, values):
    # pass 1 - determine the destination lengths
    cur_ = indices[:-1]
    next_ = indices[1:]
    count = 0
    total = 0
    for i in indices_to_apply:
        count += 1
        total += next_[i] - cur_[i]
    dest_indices = np.zeros(count+1, indices.dtype)
    dest_values = np.zeros(total, values.dtype)
    dest_indices[0] = 0
    count = 1
    total = 0
    for i in indices_to_apply:
        n = next_[i]
        c = cur_[i]
        delta = n - c
        dest_values[total:total + delta] = values[c:n]
        total += delta
        dest_indices[count] = total
        count += 1
    return dest_indices, dest_values


@njit
def apply_spans_index_of_min(spans, src_array, dest_array):
    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]

        if next - cur == 1:
            dest_array[i] = cur
        else:
            dest_array[i] = cur + src_array[cur:next].argmin()

    return dest_array


@njit
def apply_spans_index_of_max(spans, src_array, dest_array):
    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]

        if next - cur == 1:
            dest_array[i] = cur
        else:
            dest_array[i] = cur + src_array[cur:next].argmax()

    return dest_array


@njit
def apply_spans_index_of_first(spans, dest_array):
    dest_array[:] = spans[:-1]


@njit
def apply_spans_index_of_last(spans, dest_array):
    dest_array[:] = spans[1:] - 1


@njit
def apply_spans_index_of_min_filter(spans, src_array, dest_array, filter_array):
    for i in range(len(spans) - 1):
        cur = spans[i]
        next = spans[i + 1]
        if next - cur == 0:
            filter_array[i] = False
        elif next - cur == 1:
            filter_array[i] = True
            dest_array[i] = cur
        else:
            filter_array[i] = True
            dest_array[i] = cur + src_array[cur:next].argmin()

    return dest_array, filter_array


@njit
def apply_spans_index_of_max_filter(spans, src_array, dest_array, filter_array):
    for i in range(len(spans) - 1):
        cur = spans[i]
        next = spans[i + 1]
        if next - cur == 0:
            filter_array[i] = False
        elif next - cur == 1:
            filter_array[i] = True
            dest_array[i] = cur
        else:
            filter_array[i] = True
            dest_array[i] = cur + src_array[cur:next].argmax()

    return dest_array, filter_array


@njit
def apply_spans_index_of_first_filter(spans, dest_array, filter_array):
    for i in range(len(spans) - 1):
        cur = spans[i]
        next = spans[i + 1]
        if next - cur == 0:
            filter_array[i] = False
        else:
            filter_array[i] = True
            dest_array[i] = spans[i]

    return dest_array, filter_array


@njit
def apply_spans_index_of_last_filter(spans, dest_array, filter_array):
    for i in range(len(spans) - 1):
        cur = spans[i]
        next = spans[i + 1]
        if next - cur == 0:
            filter_array[i] = False
        else:
            filter_array[i] = True
            dest_array[i] = spans[i+1]-1

    return dest_array, filter_array


@njit
def apply_spans_count(spans, dest_array):
    for i in range(len(spans)-1):
        dest_array[i] = np.int64(spans[i+1] - spans[i])


@njit
def apply_spans_first(spans, src_array, dest_array):
    dest_array[:] = src_array[spans[:-1]]


@njit
def apply_spans_last(spans, src_array, dest_array):
    spans = spans[1:]-1
    dest_array[:] = src_array[spans]


@njit
def apply_spans_max(spans, src_array, dest_array):

    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]
        if next - cur == 1:
            dest_array[i] = src_array[cur]
        else:
            dest_array[i] = src_array[cur:next].max()


@njit
def apply_spans_min(spans, src_array, dest_array):

    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]
        if next - cur == 1:
            dest_array[i] = src_array[cur]
        else:
            dest_array[i] = src_array[cur:next].min()


# def _apply_spans_concat(spans, src_field):
#     dest_values = [None] * (len(spans)-1)
#     for i in range(len(spans)-1):
#         cur = spans[i]
#         next = spans[i+1]
#         if next - cur == 1:
#             dest_values[i] = src_field[cur]
#         else:
#             src = [s for s in src_field[cur:next] if len(s) > 0]
#             if len(src) > 0:
#                 dest_values[i] = ','.join(utils.to_escaped(src))
#             else:
#                 dest_values[i] = ''
#             # if len(dest_values[i]) > 0:
#             #     print(dest_values[i])
#     return dest_values


@njit
def apply_spans_concat(spans, src_index, src_values, dest_index, dest_values,
                       max_index_i, max_value_i, s_start):
    separator = np.frombuffer(b',', dtype=np.uint8)[0]
    delimiter = np.frombuffer(b'"', dtype=np.uint8)[0]
    if s_start == 0:
        index_i = np.uint32(1)
        index_v = np.int64(0)
        dest_index[0] = spans[0]
    else:
        index_i = np.uint32(0)
        index_v = np.int64(0)

    s_end = len(spans)-1
    for s in range(s_start, s_end):
        cur = spans[s]
        next = spans[s+1]
        cur_src_i = src_index[cur]
        next_src_i = src_index[next]

        dest_index[index_i] = next_src_i
        index_i += 1

        if next_src_i - cur_src_i > 1:
            if next - cur == 1:
                # only one entry to be copied, so commas not required
                next_index_v = next_src_i - cur_src_i + np.int64(index_v)
                dest_values[index_v:next_index_v] = src_values[cur_src_i:next_src_i]
                index_v = next_index_v
            else:
                # check to see how many non-zero-length entries there are; >1 means we must
                # separate them by commas
                non_empties = 0
                for e in range(cur, next):
                   if src_index[e] < src_index[e+1]:
                       non_empties += 1
                if non_empties == 1:
                    # only one non-empty entry to be copied, so commas not required
                    next_index_v = next_src_i - cur_src_i + np.int64(index_v)
                    dest_values[index_v:next_index_v] = src_values[cur_src_i:next_src_i]
                    index_v = next_index_v
                else:
                    # the outer conditional already determines that we have a non-empty entry
                    # so there must be multiple non-empty entries and commas are required
                    for e in range(cur, next):
                        src_start = src_index[e]
                        src_end = src_index[e+1]
                        comma = False
                        quotes = False
                        for i_c in range(src_start, src_end):
                            if src_values[i_c] == separator:
                                comma = True
                            elif src_values[i_c] == delimiter:
                                quotes = True

                        d_index = np.int64(0)
                        if comma or quotes:
                            dest_values[d_index] = delimiter
                            d_index += 1
                            for i_c in range(src_start, src_end):
                                if src_values[i_c] == delimiter:
                                    dest_values[d_index] = src_values[i_c]
                                    d_index += 1
                                dest_values[d_index] = src_values[i_c]
                                d_index += 1
                            dest_values[d_index] = delimiter
                            d_index += 1
                        else:
                            s_len = np.int64(src_end - src_start)
                            dest_values[index_v:index_v + s_len] = src_values[src_start:src_end]
                            d_index += s_len
                        index_v += np.int64(d_index)

        # if either the index or values are past the threshold, write them
        if index_i >= max_index_i or index_v >= max_value_i:
            break
    return s+1, index_i, index_v


# def ordered_map_to_right_left_unique_streamed(left, right, left_to_right, chunksize=1 << 20):
#     i = 0
#     j = 0
#     lc_it = iter(chunks(len(left.data), chunksize))
#     lc_range = next(lc_it)
#     rc_it = iter(chunks(len(right.data), chunksize))
#     rc_range = next(rc_it)
#     lc = left.data[lc_range[0]:lc_range[1]]
#     rc = right.data[rc_range[0]:rc_range[1]]
#     acc_written = 0
#
#     ltri = np.zeros(chunksize, dtype=left_to_right.data.dtype)
#     unmapped = 0
#     while i < len(left.data) and j < len(right.data):
#         ii, jj, u = ordered_map_to_right_left_unique_partial(i, lc, rc, ltri)
#         unmapped += u
#         if jj > 0:
#             if val.is_field_parameter(left_to_right):
#                 left_to_right.data.write(ltri[:jj])
#             else:
#                 left_to_right[acc_written:acc_written + jj] = ltri[:jj]
#                 acc_written += jj
#         i += ii
#         j += jj
#         if i > lc_range[1]:
#             raise ValueError("'i' has got ahead of current chunk; this shouldn't happen")
#         if j > rc_range[1]:
#             raise ValueError("'j' has got ahead of current chunk; this shouldn't happen")
#         if i == lc_range[1] and i < len(left.data):
#             lc_range = next(lc_it)
#             lc = left.data[lc_range[0]:lc_range[1]]
#         else:
#             lc = lc[ii:]
#         if j == rc_range[1] and j < len(right.data):
#             rc_range = next(rc_it)
#             rc = right.data[rc_range[0]:rc_range[1]]
#         else:
#             rc = rc[jj:]
#     return unmapped > 0


# @njit
# def ordered_map_to_right_left_unique_partial(d_i, left, right, left_to_right):
#     """
#     Returns:
#     [0]: how many positions forward i moved
#     [1]: how many positions forward j moved
#     [2]: how many elements were written
#     """
#     i = 0
#     j = 0
#     unmapped = 0
#     while i < len(left) and j < len(right):
#         if left[i] < right[j]:
#             i += 1
#         elif left[i] > right[j]:
#             left_to_right[j] = INVALID_INDEX
#             j += 1
#             unmapped += 1
#         else:
#             left_to_right[j] = i + d_i
#             if j+1 < len(right) and right[j+1] != right[j]:
#                 i += 1
#             j += 1
#     return i, j, unmapped


def ordered_map_to_right_right_unique_streamed(left, right, left_to_right, chunksize=1 << 20):
    i = 0
    j = 0
    lc_it = iter(chunks(len(left.data), chunksize))
    lc_range = next(lc_it)
    rc_it = iter(chunks(len(right.data), chunksize))
    rc_range = next(rc_it)
    lc = left.data[lc_range[0]:lc_range[1]]
    rc = right.data[rc_range[0]:rc_range[1]]
    acc_written = 0

    is_field_parameter = val.is_field_parameter(left_to_right)
    result_dtype = left_to_right.data.dtype if is_field_parameter else left_to_right.dtype
    ltri = np.zeros(chunksize, dtype=result_dtype)
    unmapped = 0
    while i < len(left.data) and j < len(right.data):
        ii, jj, u = ordered_map_to_right_right_unique_partial(j, lc, rc, ltri)
        unmapped += u
        if ii > 0:
            if is_field_parameter:
                left_to_right.data.write(ltri[:ii])
            else:
                left_to_right[acc_written:acc_written + ii] = ltri[:ii]
                acc_written += ii
        i += ii
        j += jj
        if i > lc_range[1]:
            raise ValueError("'i' has got ahead of current chunk; this shouldn't happen")
        if j > rc_range[1]:
            raise ValueError("'j' has got ahead of current chunk; this shouldn't happen")
        if i == lc_range[1] and i < len(left.data):
            lc_range = next(lc_it)
            lc = left.data[lc_range[0]:lc_range[1]]
        else:
            lc = lc[ii:]
        if j == rc_range[1] and j < len(right.data):
            rc_range = next(rc_it)
            rc = right.data[rc_range[0]:rc_range[1]]
        else:
            rc = rc[jj:]
    return unmapped > 0


@njit
def ordered_map_to_right_right_unique_partial(d_j, left, right, left_to_right):
    """
    Returns:
    [0]: how many positions forward i moved
    [1]: how many positions forward j moved
    [2]: how many elements were written
    """
    i = 0
    j = 0
    unmapped = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            left_to_right[i] = INVALID_INDEX
            i += 1
            unmapped += 1
        elif left[i] > right[j]:
            j += 1
        else:
            left_to_right[i] = j + d_j
            if i+1 >= len(left) or left[i+1] != left[i]:
                j += 1
            i += 1
            # if j+1 < len(right) and right[j+1] != right[j]:
            #     i += 1
            # j += 1
    return i, j, unmapped


@njit
def ordered_map_to_right_right_unique(first, second, result):
    if len(first) != len(result):
        msg = "'first' and 'result' must be the same length"
        raise ValueError(msg)
    i = 0
    j = 0
    unmapped = 0
    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            result[i] = INVALID_INDEX
            i += 1
            unmapped += 1
        elif first[i] > second[j]:
            j += 1
        else:
            result[i] = j
            if i+1 >= len(first) or first[i+1] != first[i]:
                j += 1
            i += 1

    while j < len(second):
        result[j] = INVALID_INDEX
        j += 1

    return unmapped > 0


@njit
def ordered_map_to_right_both_unique(first, second, result):
    if len(first) != len(result):
        msg = "'second' and 'result' must be the same length"
        raise ValueError(msg)
    i = 0
    j = 0

    unmapped = 0
    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            result[i] = INVALID_INDEX
            i += 1
            unmapped += 1
        elif first[i] > second[j]:
            j += 1
        else:
            result[i] = j
            i += 1
            j += 1

    while i < len(first):
        result[i] = INVALID_INDEX
        i += 1

    return unmapped > 0


@njit
def ordered_inner_map_result_size(left, right):
    i = 0
    j = 0
    result_size = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            cur_i_count = 1
            while i+1 < len(left) and left[i + 1] == left[i]:
                cur_i_count += 1
                i += 1
            cur_j_count = 1
            while j+1 < len(right) and right[j + 1] == right[j]:
                cur_j_count += 1
                j += 1
            result_size += cur_i_count * cur_j_count
            i += 1
            j += 1
    return result_size


def ordered_outer_map_result_size_both_unique(left, right):
    i = 0
    j = 0
    result_size = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            i += 1
            j += 1
        result_size += 1
    while i < len(left):
        i += 1
        result_size += 1
    while j < len(right):
        j += 1
        result_size += 1
    return result_size


@njit
def ordered_inner_map_both_unique(left, right, left_to_inner, right_to_inner):
    i = 0
    j = 0
    cur_m = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            left_to_inner[cur_m] = i
            right_to_inner[cur_m] = j
            cur_m += 1
            i += 1
            j += 1


def ordered_inner_map_left_unique_streamed(left, right, left_to_inner, right_to_inner,
                                           chunksize=1 << 20):
    i = 0
    j = 0
    left_chunks = chunks(len(left.data), 4)
    right_chunks = chunks(len(right.data), 4)

    lc_it = iter(left_chunks)
    lc_range = next(lc_it)
    rc_it = iter(right_chunks)
    rc_range = next(rc_it)
    lc = left.data[lc_range[0]:lc_range[1]]
    rc = right.data[rc_range[0]:rc_range[1]]

    lti = np.zeros(4, dtype=left_to_inner.data.dtype)
    rti = np.zeros(4, dtype=right_to_inner.data.dtype)
    while i < len(left.data) and j < len(right.data):
        ii, jj, m = ordered_inner_map_left_unique_partial(i, j, lc, rc, lti, rti)
        if m > 0:
            left_to_inner.data.write(lti[:m])
            right_to_inner.data.write(rti[:m])
        i += ii
        j += jj
        if i > lc_range[1]:
            raise ValueError("'i' has got ahead of current chunk; this shouldn't happen")
        if j > rc_range[1]:
            raise ValueError("'j' has got ahead of current chunk; this shouldn't happen")
        if i == lc_range[1] and i < len(left.data):
            lc_range = next(lc_it)
            lc = left.data[lc_range[0]:lc_range[1]]
        else:
            lc = lc[ii:]
        if j == rc_range[1] and j < len(right.data):
            rc_range = next(rc_it)
            rc = right.data[rc_range[0]:rc_range[1]]
        else:
            rc = rc[jj:]

@njit
def ordered_inner_map_left_unique_partial(d_i, d_j, left, right,
                                          left_to_inner, right_to_inner):
    """
    Returns:
    [0]: how many positions forward i moved
    [1]: how many positions forward j moved
    [2]: how many elements were written
    """
    i = 0
    j = 0
    m = 0
    while i < len(left) and j < len(right) and m < len(left_to_inner):
        if left[i] < right[j]:
            i += 1

        elif left[i] > right[j]:
            j += 1
        else:
            left_to_inner[m] = i + d_i
            right_to_inner[m] = j + d_j
            m += 1
            if j+1 >= len(right) or right[j+1] != right[j]:
                i += 1
            j += 1
    return i, j, m


@njit
def ordered_inner_map_left_unique(left, right, left_to_inner, right_to_inner):
    i = 0
    j = 0
    cur_m = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            cur_j = j
            while cur_j + 1 < len(right) and right[cur_j + 1] == right[cur_j]:
                cur_j += 1
            for jj in range(j, cur_j+1):
                left_to_inner[cur_m] = i
                right_to_inner[cur_m] = jj
                cur_m += 1
            i += 1
            j = cur_j + 1


@njit
def ordered_inner_map(left, right, left_to_inner, right_to_inner):
    i = 0
    j = 0
    cur_m = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            cur_i = i
            while cur_i + 1 < len(left) and left[cur_i + 1] == left[cur_i]:
                cur_i += 1
            cur_j = j
            while cur_j + 1 < len(right) and right[cur_j + 1] == right[cur_j]:
                cur_j += 1
            for ii in range(i, cur_i+1):
                for jj in range(j, cur_j+1):
                    left_to_inner[cur_m] = ii
                    right_to_inner[cur_m] = jj
                    cur_m += 1
            i = cur_i + 1
            j = cur_j + 1


@njit
def ordered_get_last_as_filter(field):
    result = np.zeros(len(field), dtype=numba.types.boolean)
    for i in range(len(field)-1):
        result[i] = field[i] != field[i+1]
    result[-1] = True
    return result


@njit
def ordered_generate_journalling_indices(old, new):
    i = 0
    j = 0
    total = 0
    while i < len(old) and j < len(new):
        if old[i] < new[j]:
            while i+1 < len(old) and old[i+1] == old[i]:
                i += 1
            i += 1
            total += 1
        elif old[i] > new[j]:
            j += 1
            total += 1
        else:
            while i+1 < len(old) and old[i+1] == old[i]:
                i += 1
            i += 1
            j += 1
            total += 1
    while i < len(old):
        while i+1 < len(old) and old[i+1] == old[i]:
            i += 1
        i += 1
        total += 1
    while j < len(new):
        j += 1
        total += 1

    old_inds = np.full(total, -1, dtype=np.int64)
    new_inds = np.full(total, -1, dtype=np.int64)

    i = 0
    j = 0
    joint = 0
    while i < len(old) and j < len(new):
        if old[i] < new[j]:
            while i+1 < len(old) and old[i+1] == old[i]:
                i += 1
            old_inds[joint] = i
            new_inds[joint] = -1
            i += 1
            joint += 1
        elif old[i] > new[j]:
            old_inds[joint] = -1
            new_inds[joint] = j
            j += 1
            joint += 1
        else:
            while i+1 < len(old) and old[i+1] == old[i]:
                i += 1
            old_inds[joint] = i
            new_inds[joint] = j
            i += 1
            j += 1
            joint += 1

    while i < len(old):
        while i+1 < len(old) and old[i+1] == old[i]:
            i += 1
        old_inds[joint] = i
        new_inds[joint] = -1
        i += 1
        joint += 1
    while j < len(new):
        old_inds[joint] = -1
        new_inds[joint] = j
        j += 1
        joint += 1

    return old_inds, new_inds


@njit
def compare_rows_for_journalling(old_map, new_map, old_field, new_field, to_keep):
    for i in range(len(old_map)):
        if to_keep[i] == False:
            if old_map[i] == -1:
                # row is new so must be kept
                to_keep[i] = True
            elif new_map[i] == -1:
                # row has been removed so don't count as kept
                to_keep[i] = False
            else:
                to_keep[i] = old_field[old_map[i]] != new_field[new_map[i]]


@njit
def compare_indexed_rows_for_journalling(old_map, new_map,
                                         old_indices, old_values, new_indices, new_values,
                                         to_keep):
    assert len(old_map) == len(new_map)
    assert old_indices[-1] == len(old_values)
    assert new_indices[-1] == len(new_values)
    for i in range(len(old_map)):
        if to_keep[i] == False:
            if old_map[i] == -1:
                # row is new so must be kept
                to_keep[i] = True
            elif new_map[i] == -1:
                # row has been removed so don't count as kept
                to_keep[i] = False
            else:
                old_value = old_values[old_indices[old_map[i]]:old_indices[old_map[i]+1]]
                new_value = new_values[new_indices[new_map[i]]:new_indices[new_map[i]+1]]
                to_keep[i] = not np.array_equal(old_value, new_value)


@njit
def merge_journalled_entries(old_map, new_map, to_keep, old_src, new_src, dest):
    cur_old = 0
    cur_dest = 0
    for i in range(len(old_map)):
        # copy all rows up to the entry, if there are any
        while cur_old <= old_map[i]:
            dest[cur_dest] = old_src[cur_old]
            cur_old += 1
            cur_dest += 1
        # copy the new row if there is one
        if to_keep[i] == True:
            dest[cur_dest] = new_src[new_map[i]]
            cur_dest += 1

# def merge_journalled_entries(old_map, new_map, to_keep, old_src, new_src, dest):
#     for om, im, tk in zip(old_map, new_map, to_keep):
#         for omi in old_map:
#             dest.add_to(next(old_src))
#         if tk:
#             dest.add_to(next(new_src))


@njit
def merge_indexed_journalled_entries_count(old_map, new_map, to_keep, old_src_inds, new_src_inds):
    cur_old = 0
    acc_val = 0
    for i in range(len(old_map)):
        while cur_old <= old_map[i]:
            ind_delta = old_src_inds[cur_old+1] - old_src_inds[cur_old]
            acc_val += ind_delta
            cur_old += 1
        if to_keep[i] == True:
            ind_delta = new_src_inds[new_map[i]+1] - new_src_inds[new_map[i]]
            acc_val += ind_delta
    return acc_val


@njit
def merge_indexed_journalled_entries(old_map, new_map, to_keep,
                                    old_src_inds, old_src_vals,
                                    new_src_inds, new_src_vals,
                                    dest_inds, dest_vals):
    cur_old = 0
    cur_dest = 1
    ind_acc = 0
    dest_inds[0] = 0
    for i in range(len(old_map)):
        # copy all rows up to the entry, if there are any
        while cur_old <= old_map[i]:
            ind_delta = old_src_inds[cur_old + 1] - old_src_inds[cur_old]
            ind_acc += ind_delta
            dest_inds[cur_dest] = ind_acc
            if ind_delta > 0:
                dest_vals[ind_acc-ind_delta:ind_acc] = \
                    old_src_vals[old_src_inds[cur_old]:old_src_inds[cur_old + 1]]
            cur_old += 1
            cur_dest += 1
        # copy the new row if there is one
        if to_keep[i] == True:
            ind_delta = new_src_inds[new_map[i] + 1] - new_src_inds[new_map[i]]
            ind_acc += ind_delta
            dest_inds[cur_dest] = ind_acc
            if ind_delta > 0:
                dest_vals[ind_acc-ind_delta:ind_acc] = \
                    new_src_vals[new_src_inds[new_map[i]]:new_src_inds[new_map[i] + 1]]
            cur_dest += 1


def merge_entries_segment(i_start, cur_old_start,
                          old_map, new_map, to_keep, old_src, new_src, dest):
    """
    :param i_start: the initial value to apply to 'i'
    :param cur_old_start: the initial value to apply to 'cur_old
    :param old_map: the map (in i-space) for the existing records
    :param new_map: the map (in i-space) for the new records
    :param to_keep: the flags (in i-space) indicating whether the new record should be kept
    :param old_src: the source for the existing records
    :param new_src: the source for the new records
    :param dest: the sink for the merged sources
    :return:
    """
    cur_old = cur_old_start
    cur_dest = 0
    for i in range(i_start, len(old_map)):
        # copy all rows up to the entry, if there are any
        while cur_old <= old_map[i]:
            dest[cur_dest] = old_src[cur_old]
            cur_old += 1
            cur_dest += 1
            if cur_dest == len(dest):
                return i, cur_old
        # copy the new row if there is one
        if to_keep[i] == True:
            dest[cur_dest] = new_src[new_map[i]]
            cur_dest += 1
            if cur_dest == len(dest):
                return i, cur_old


def streaming_sort_merge(src_index_f, src_value_f, tgt_index_f, tgt_value_f,
                         segment_length, chunk_length):

    # get the number of segments
    segment_count = len(src_index_f.data) // segment_length
    if len(src_index_f.data) % segment_length != 0:
        segment_count += 1

    segment_starts = np.zeros(segment_count, dtype=np.int64)
    segment_lengths = np.zeros(segment_count, dtype=np.int64)
    for i, s in enumerate(utils.chunks(len(src_index_f.data), segment_length)):
        segment_starts[i] = s[0]
        segment_lengths[i] = s[1] - s[0]

    # original segment indices for debugging
    segment_indices = np.zeros(segment_count, dtype=np.int32)

    # the index of the chunk within a given segment
    chunk_indices = np.zeros(segment_count, dtype=np.int64)

    # the (chunk_local) index for each segment
    in_chunk_indices = np.zeros(segment_count, dtype=np.int64)

    # the (chunk_local) length for each segment
    in_chunk_lengths = np.zeros(segment_count, dtype=np.int64)

    src_value_chunks = List()
    src_index_chunks = List()

    # get the first chunk for each segment
    for i in range(segment_count):
        index_start = segment_starts[i] + chunk_indices[i] * chunk_length
        src_value_chunks.append(src_value_f.data[index_start:index_start+chunk_length])
        src_index_chunks.append(src_index_f.data[index_start:index_start+chunk_length])
        in_chunk_lengths[i] = len(src_value_chunks[i])

    dest_indices = np.zeros(segment_count * chunk_length, dtype=src_index_f.data.dtype)
    dest_values = np.zeros(segment_count * chunk_length, dtype=src_value_f.data.dtype)

    target_index = 0
    while target_index < len(src_index_f.data):
        index_delta = streaming_sort_partial(in_chunk_indices, in_chunk_lengths,
                                             src_value_chunks, src_index_chunks,
                                             dest_values, dest_indices)
        tgt_index_f.data.write(dest_indices[:index_delta])
        tgt_value_f.data.write(dest_values[:index_delta])
        target_index += index_delta

        chunk_filter = np.ones(segment_count, dtype=np.bool)
        for i in range(segment_count):
            if in_chunk_indices[i] == in_chunk_lengths[i]:
                chunk_indices[i] += 1
                index_start = segment_starts[i] + chunk_indices[i] * chunk_length
                remaining = segment_starts[i] + segment_lengths[i] - index_start
                remaining = min(remaining, chunk_length)
                if remaining > 0:
                    src_value_chunks[i] = src_value_f.data[index_start:index_start+remaining]
                    src_index_chunks[i] = src_index_f.data[index_start:index_start+remaining]
                    in_chunk_lengths[i] = len(src_value_chunks[i])
                    in_chunk_indices[i] = 0
                else:
                    # can't clear list contents because we are using numba list, but they
                    # get filtered out in the following section
                    chunk_filter[i] = 0

        if chunk_filter.sum() < len(chunk_filter):
            segment_count = chunk_filter.sum()
            segment_indices = segment_indices[chunk_filter]
            segment_starts = segment_starts[chunk_filter]
            segment_lengths = segment_lengths[chunk_filter]
            chunk_indices = chunk_indices[chunk_filter]
            in_chunk_indices = in_chunk_indices[chunk_filter]
            in_chunk_lengths = in_chunk_lengths[chunk_filter]
            filtered_value_chunks = List()
            filtered_index_chunks = List()
            for i in range(len(src_value_chunks)):
                if chunk_filter[i]:
                    filtered_value_chunks.append(src_value_chunks[i])
                    filtered_index_chunks.append(src_index_chunks[i])
            src_value_chunks = filtered_value_chunks
            src_index_chunks = filtered_index_chunks


@njit
def streaming_sort_partial(in_chunk_indices, in_chunk_lengths,
                           src_value_chunks, src_index_chunks, dest_value_chunk, dest_index_chunk):
    dest_index = 0
    max_possible = in_chunk_lengths.sum()
    while(dest_index < max_possible):
        if in_chunk_indices[0] == in_chunk_lengths[0]:
            return dest_index
        min_value = src_value_chunks[0][in_chunk_indices[0]]
        min_value_index = 0
        for i in range(1, len(in_chunk_indices)):
            if in_chunk_indices[i] == in_chunk_lengths[i]:
                return dest_index
            cur_value = src_value_chunks[i][in_chunk_indices[i]]
            if cur_value < min_value:
                min_value = cur_value
                min_value_index = i

        min_index = src_index_chunks[min_value_index][in_chunk_indices[min_value_index]]
        dest_index_chunk[dest_index] = min_index
        dest_value_chunk[dest_index] = min_value
        dest_index += 1
        in_chunk_indices[min_value_index] += 1

    return dest_index


def is_ordered(field):
    if len(field) == 1:
        return True

    if np.issubdtype(field.dtype, np.number):
        fn = np.greater
    else:
        fn = np.char.greater
    return not np.any(fn(field[:-1], field[1:]))
