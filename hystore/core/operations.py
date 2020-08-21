import numpy as np
from numba import jit, njit

INVALID_INDEX = 1 << 62


@njit
def safe_map(data_field, map_field, map_filter, empty_value):
    result = np.zeros_like(map_field, dtype=data_field.dtype)
    for i in range(len(map_field)):
        if map_filter[i]:
            result[i] = data_field[map_field[i]]
        else:
            result[i] = empty_value
    return result


@njit
def map_valid(data_field, map_field, result=None):
    if result is None:
        result = np.zeros_like(map_field, dtype=data_field.dtype)
    for i in range(len(map_field)):
        if map_field[i] < INVALID_INDEX:
            result[i] = data_field[map_field[i]]
    return result


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


@njit
def ordered_map_to_right_left_unique(first, second, result):
    if len(second) != len(result):
        msg = "'second' and 'result' must be the same length"
        raise ValueError(msg)
    i = 0
    j = 0

    unmapped = 0
    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            i += 1
        elif first[i] > second[j]:
            result[j] = INVALID_INDEX
            j += 1
            unmapped += 1
        else:
            result[j] = i
            if j+1 < len(second) and second[j+1] != second[j]:
                i += 1
            j += 1

    while j < len(second):
        result[j] = INVALID_INDEX
        j += 1

    return unmapped > 0


@njit
def ordered_map_to_right_both_unique(first, second, result):
    if len(second) != len(result):
        msg = "'second' and 'result' must be the same length"
        raise ValueError(msg)
    i = 0
    j = 0

    unmapped = 0
    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            i += 1
        elif first[i] > second[j]:
            result[j] = INVALID_INDEX
            j += 1
            unmapped += 1
        else:
            result[j] = i
            i += 1
            j += 1

    while j < len(second):
        result[j] = INVALID_INDEX
        j += 1

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
