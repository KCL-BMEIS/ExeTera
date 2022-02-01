from datetime import datetime
from typing import Optional, Union

import numpy as np
from numba import jit, njit
import numba
import numba.typed as nt

from exetera.core import validation as val
from exetera.core.abstract_types import Field
from exetera.core import fields
from exetera.core import utils

DEFAULT_CHUNKSIZE = 1 << 20
INVALID_INDEX = 1 << 62
INVALID_INDEX_64 = INVALID_INDEX
INVALID_INDEX_32 = (1 << 31) - 1
MAX_DATETIME = datetime(year=3000, month=1, day=1)


def dtype_to_str(dtype):
    if isinstance(dtype, str):
        return dtype

    if dtype == bool:
        return 'bool'
    elif dtype == np.int8:
        return 'int8'
    elif dtype == np.int16:
        return 'int16'
    elif dtype == np.int32:
        return 'int32'
    elif dtype == np.int64:
        return 'int64'
    elif dtype == np.uint8:
        return 'uint8'
    elif dtype == np.uint16:
        return 'uint16'
    elif dtype == np.uint32:
        return 'uint32'
    elif dtype == np.uint64:
        return 'uint64'
    elif dtype == np.float32:
        return 'float32'
    elif dtype == np.float64:
        return 'float64'

    raise ValueError("Unsupported dtype '{}'".format(dtype))


def str_to_dtype(str_dtype):
    if str_dtype == 'bool':
        return bool
    elif str_dtype == 'int8':
        return np.int8
    elif str_dtype == 'int16':
        return np.int16
    elif str_dtype == 'int32':
        return np.int32
    elif str_dtype == 'int64':
        return np.int64
    elif str_dtype == 'uint8':
        return np.uint8
    elif str_dtype == 'uint16':
        return np.uint16
    elif str_dtype == 'uint32':
        return np.uint32
    elif str_dtype == 'uint64':
        return np.uint64
    elif str_dtype == 'float32':
        return np.float32
    elif str_dtype == 'float64':
        return np.float64

    raise ValueError("Unsupported dtype '{}'".format(str_dtype))


@njit
def chunks(length, chunksize=1 << 20):
    cur = 0
    while cur < length:
        next_ = min(length, cur + chunksize)
        yield cur, next_
        cur = next_


# mapping functionality
# =====================


def count_back(array):
    """
    This is a helper function that provides functionality specific to streaming ordered
    merges. It takes an array in sorted order and calculates a trimmed length that excludes
    the final sequence of equal values:
    Example::

        [10, 20, 30, 40, 50] -> 4 ([10, 20, 30, 40])
        [10, 20, 30, 40, 40] -> 3 ([10, 20, 30])
        [10, 20, 30, 30, 30] -> 2 ([10, 20])
        [10, 20, 20, 20, 20] -> 1 ([10])
    """
    v = len(array)-1
    while v > 0:
        if array[v-1] != array[v]:
            return v
        v -= 1
    return 0


def next_chunk(current: int,
               length: int,
               desired: int):
    """
    This is a helper function that can be used whenever you want to access a large sequence
    of data in chunks. It simply carries out the calculation that returns the extents of the
    next chunk taking into account the ``length`` of the sequence. The sequence itself is not
    required here, only the length.
    :param current: the starting point of the chunk
    :param length: the length of the sequence being chunked
    :param desired: the requested length of the chunk
    :return: A tuple of the chunk extents. The first value is inclusive; the second is exclusive
    """
    if current + desired < length:
        return current, current + desired
    else:
        return current, length


def get_next_chunk(start: int,
                   chunk_size: int,
                   field: Field):
    """
    This is a helper function that provides functionality specific to streaming ordered
    merges. It assumes that ``field`` is in sorted order.

    This function is used to fetch chunks of memory from a field to be consumed by
    streaming merges. It first fetches the chunk of a given chunk size, or the size of
    the remaining memory, whichever is smaller. It then 'trims' that memory by removing
    the last sequence of equal values from the valid range.

    :param start: The start of the chunk to be returned
    :param chunksize: The size of the chunk to be considered. The returned chunk will always
    be shorter than this unless it is the final chunk of the ``field`` data
    :param field: The field from which data should be fetched. This field must be in sorted
    order
    :return: A tuple representing the range (inclusive, exclusive) and an numpy ndarray
    containing the data. Note, this is is typically longer than the range returned, as we
    do not trim the data for performance reasons.
    """
    next_range = next_chunk(start, len(field), chunk_size)
    next_ = field.data[next_range[0]:next_range[1]]
    if next_range[1] != len(field):
        next_range = next_range[0], next_range[0] + count_back(next_)
    return next_range, next_


def first_trimmed_chunk(field, chunk_size):
    chunk, data = get_next_chunk(0, chunk_size, field)
    max_index = chunk[1] - chunk[0]
    return chunk, data, max_index, chunk[0], 0


def next_trimmed_chunk(field, chunk, chunk_size):
    chunk, data = get_next_chunk(chunk[1], chunk_size, field)
    max_index = chunk[1] - chunk[0]
    return chunk, data, max_index, chunk[0], 0


def first_untrimmed_chunk(field, chunk_size):
    chunk = next_chunk(0, len(field), chunk_size)
    data = field.data[chunk[0]:chunk[1]]
    max_index = chunk[1] - chunk[0]
    return chunk, data, max_index, chunk[0], 0


def next_untrimmed_chunk(field, chunk, chunk_size):
    chunk = next_chunk(chunk[1], len(field), chunk_size)
    data = field.data[chunk[0]:chunk[1]]
    max_index = chunk[1] - chunk[0]
    return chunk, data, max_index, chunk[0], 0


@njit
def get_valid_value_extents(chunk, start, end, invalid=-1):
    first = invalid
    for i in range(start, end):
        if chunk[i] != invalid:
            first = chunk[i]
            break
    last = invalid

    j = end - 1
    while j >= i:
        if chunk[j] != invalid:
            last = chunk[j]
            break
        j -= 1

    return first, last


def get_map_datatype_str_based_on_lengths(left_len, right_len):
    if left_len < (2 << 30) and right_len < (2 << 30):
        index_dtype = 'int32'
    else:
        index_dtype = 'int64'
    return index_dtype


def get_map_datatype_based_on_lengths(left_len, right_len):
    dtype_str = get_map_datatype_str_based_on_lengths(left_len, right_len)
    return np.int32 if dtype_str == 'int32' else np.int64


# def safe_map(field, map_field, map_filter, empty_value=None):
#     if isinstance(field, Field):
#         if field.indexed:
#             return safe_map_indexed_values(
#                 field.indices[:], field.values[:], map_field, map_filter, empty_value)
#         else:
#             return safe_map_values(field.data[:], map_field, map_filter, empty_value)
#     elif isinstance(field, np.ndarray):
#         return safe_map_values(field, map_field, map_filter, empty_value)


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
            if empty_value is not None:
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
def map_valid(data_field, map_field, result=None, invalid=-1):
    if result is None:
        result = np.zeros_like(map_field, dtype=data_field.dtype)
    for i in range(len(map_field)):
        if map_field[i] != invalid:
            result[i] = data_field[map_field[i]]
    return result


def ordered_map_valid_stream_old(data_field, map_field, result_field,
                             invalid=-1, chunksize=DEFAULT_CHUNKSIZE):
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
        mm, dd = ordered_map_valid_partial_old(df_range[0], dfc, mfc, rslt, invalid)
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


@njit
def ordered_map_valid_partial_old(d, data_field, map_field, result, invalid):
    i = 0
    while True:
        val = map_field[i]
        if val != invalid:
            if val >= d + len(data_field):
                # need a new data_field chunk
                return i, val
            result[i] = data_field[val - d]
        i += 1
        if i >= len(map_field):
            return i, val


@njit
def next_map_subchunk(map_, sm, invalid, chunksize):

    start = -1
    while sm < len(map_) and map_[sm] == invalid:
        sm += 1

    if sm < len(map_):
        start = map_[sm]

    while sm < len(map_) and map_[sm] - start < chunksize:
        sm += 1

    return sm


def get_map_subchunks_based_on_index_lengths(map_, invalid, chunksize):
    chunks = list()
    sm = 0
    while sm < len(map_):
        next_sm = next_map_subchunk(map_, sm, -1, chunksize)
        chunks.append((sm, next_sm))
        sm = next_sm
    return chunks


def ordered_map_valid_stream(data_field, map_field, result_field,
                             invalid=-1, chunksize=DEFAULT_CHUNKSIZE):
    """
    . for each map chunk
      . calculate sub chunks based on indices
        . for each sub chunk
          . map indices for sub chunk
    """
    result_data = np.zeros(chunksize, dtype=result_field.data.dtype)

    empty_value = None
    if np.issubdtype(result_field.data.dtype, bool):
        empty_value = False
    elif np.issubdtype(result_field.data.dtype, np.number):
        empty_value = 0
    else:
        empty_value = b''
    # empty_value = 0 if np.issubdtype(result_field.data.dtype, np.number) else b''

    m_chunk, map_, m_max, m_off, m = first_untrimmed_chunk(map_field, chunksize)

    while m + m_off < len(map_field):

        # the whole map chunk might encompass too large a range for the index buffer, so
        # break it up into sub-chunks if necessary
        sub_map_chunks = get_map_subchunks_based_on_index_lengths(map_, invalid, chunksize)
        for sm_start, sm_end in sub_map_chunks:
            d_limits = get_valid_value_extents(map_, sm_start, sm_end, invalid)
            if d_limits[0] == invalid:
                # no unfiltered values in this chunk so just assign empty entries to the result field
                result_data.fill(0)
            else:
                values = data_field.data[d_limits[0]:d_limits[1]+1]
                _ = ordered_map_valid_partial(values, map_, sm_start, sm_end, d_limits[0],
                                               result_data, invalid, empty_value)


        result_field.data.write(result_data[:m_max])
        m_chunk, map_, m_max, m_off, m = next_untrimmed_chunk(map_field, m_chunk, chunksize)

        # if m > 0:
        #     result_field.data.write(result_data[:m])
        #     m_chunk, map_, m_max, m_off, m = next_untrimmed_chunk(map_field, m_chunk, chunksize)


@njit
def ordered_map_valid_partial(values,
                              map_values,
                              sm_start,
                              sm_end,
                              d_start,
                              result_data,
                              invalid,
                              invalid_value):
    sm = sm_start
    while sm < sm_end:
        if map_values[sm] == invalid:
            result_data[sm] = invalid_value
        else:
            result_data[sm] = values[map_values[sm] - d_start]
        sm += 1

    return sm


def calculate_chunk_decomposition(s_start, s_end, indices, value_chunk_size, sub_chunks):
    if indices[s_end] - indices[s_start] > value_chunk_size and s_end - s_start > 1:
        s_mid = s_start + (s_end - s_start) // 2
        calculate_chunk_decomposition(s_start, s_mid, indices, value_chunk_size, sub_chunks)
        calculate_chunk_decomposition(s_mid, s_end, indices, value_chunk_size, sub_chunks)
    else:
        sub_chunks.append((s_start, s_end))


def ordered_map_valid_indexed_stream(data_field, map_field, result_field,
                                     invalid=-1, chunksize=DEFAULT_CHUNKSIZE, value_factor=8):
    result_indices = np.zeros(chunksize, dtype=np.int64)
    result_field.indices.write(result_indices[:1])
    result_values = np.zeros(chunksize * value_factor, dtype=np.uint8)

    # iterate over the map field
    # . for each map chunk
    #   . get a chunk of src data indices
    #   . sub-divide that chunk if necessary if values blows the budget
    #   . for each sub-chunk
    #     . perform partial map
    # fetch a chunk of indices and values

    # m: the index of our current position in the overall map field - updates with each map chunk
    # sm_start: the start of the current sub-map chunk, relative to the start of the current map
    #           chunk
    # sm_end: the end of the current sub-map chunk, relative to the start of the current map chunk

    ri, rv = 0, 0
    ri_accum = 0
    m_chunk, map_, m_max, m_off, m = first_untrimmed_chunk(map_field, chunksize)

    while m + m_off < len(map_field):

        # the whole map chunk might encompass too large a range for the index buffer, so
        # break it up into sub-chunks if necessary
        sub_map_chunks = get_map_subchunks_based_on_index_lengths(map_, invalid, chunksize)
        for sm_start, sm_end in sub_map_chunks:

            i_limits = get_valid_value_extents(map_, sm_start, sm_end, invalid)
            if i_limits[0] == -1:
                # no unfiltered values in this chunk so just assign empty entries to the result field
                result_indices.fill(ri_accum)
                result_field.indices.write(result_indices[:sm_end - sm_start])
                # m += sm_end - sm_start
            else:
                # TODO: can potentially optimise here by checking if upper limit has increased
                indices_ = data_field.indices[i_limits[0]:i_limits[1]+2]
                sub_chunks = list()
                calculate_chunk_decomposition(0, i_limits[1] - i_limits[0]+1, indices_,
                                              chunksize * value_factor, sub_chunks)

                s = 0
                sc = sub_chunks[s]
                values_ = data_field.values[indices_[sc[0]]:indices_[sc[1]]]

                # iterate over the sub-chunks (there may only be one) until this map sub-chunk
                # has been fully consumed
                sm = sm_start
                while sm < sm_end:

                    sm, ri, rv, ri_accum, need_subchunk = \
                        ordered_map_valid_indexed_partial(map_, sm_start, sm_end,
                                                          indices_, sc[0], sc[1], values_,
                                                          i_limits[0],
                                                          result_indices, result_values,
                                                          invalid, sm, ri, rv, ri_accum)

                    # update the subchunk if necessary
                    if need_subchunk:
                        s += 1
                        sc = sub_chunks[s]
                        values_ = data_field.values[indices_[sc[0]]:indices_[sc[1]]]

                    # TODO: extra validation to ensure subchunks have been iterated over and we exit
                    # this while loop on the last chunk

                    # write the result buffers
                    if ri > 0:
                        result_field.indices.write_part(result_indices[:ri])
                        ri = 0

                    if rv > 0:
                        result_field.values.write_part(result_values[:rv])
                        rv = 0

            # # update m to be the end of this subchunk
            # m += sm_end - sm_start

        if m_off + m < len(map_field):
            m_chunk, map_, m_max, m_off, m = next_untrimmed_chunk(map_field, m_chunk, chunksize)
            ri, rv = 0, 0


@njit
def ordered_map_valid_indexed_partial(sm_values,
                                      sm_start,
                                      sm_end,
                                      indices,
                                      i_start,
                                      i_max,
                                      values,
                                      mv_start,
                                      result_indices,
                                      result_values,
                                      invalid,
                                      sm,
                                      ri,
                                      rv,
                                      ri_accum):

    need_values = False
    # this is the offset that must be subtracted from the value index before it is looked up
    v_offset = indices[i_start]

    while sm < sm_end: # and ri < len(result_indices):
        if sm_values[sm] == invalid:
            result_indices[ri] = ri_accum
        else:
            i = sm_values[sm] - mv_start
            if i >= i_max:
                need_values = True
                break
            v_start = indices[i] - v_offset
            v_end = indices[i+1] - v_offset
            if rv + v_end - v_start > len(result_values):
                break
            for v in range(v_start, v_end):
                result_values[rv] = values[v]
                rv += 1
            ri_accum += v_end - v_start
            result_indices[ri] = ri_accum
        sm += 1
        ri += 1

    return sm, ri, rv, ri_accum, need_values


# chunked copying functionality

def element_chunked_copy(src_elem, dest_elem, chunksize):
    i = 0
    chunk = next_chunk(i, len(src_elem), chunksize)
    while i < len(src_elem):
        dest_elem.write(src_elem[chunk[0]:chunk[1]])
        i += chunk[1] - chunk[0]
        chunk = next_chunk(i, len(src_elem), chunksize)


def chunked_copy(src_field, dest_field, chunksize=1 << 20):
    if src_field.indexed:
        element_chunked_copy(src_field.indices, dest_field.indices, chunksize)
        element_chunked_copy(src_field.values, dest_field.values, chunksize)
    else:
        element_chunked_copy(src_field.data, dest_field.data, chunksize)


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


def get_spans_for_field(ndarray):
    results = np.zeros(len(ndarray) + 1, dtype=bool)
    if np.issubdtype(ndarray.dtype, np.number):
        fn = np.not_equal
    else:
        fn = np.char.not_equal
    results[1:-1] = fn(ndarray[:-1], ndarray[1:])

    results[0] = True
    results[-1] = True
    return np.nonzero(results)[0]


@njit
def _get_spans_for_2_fields_by_spans(span0, span1):
    spans = []
    j=0
    for i in range(len(span0)):
        if j<len(span1):
            while span1[j] < span0[i]:
                spans.append(span1[j])
                j += 1
            if span1[j] == span0[i]:
                j += 1
        spans.append(span0[i])
    if j<len(span1): #if two ndarray are not equally sized
        spans.extend(span1[j:])
    return spans


@njit
def _get_spans_for_2_fields(ndarray0, ndarray1):
    count = 0
    spans = np.zeros(len(ndarray0)+1, dtype=np.uint32)
    spans[0] = 0
    for i in np.arange(1, len(ndarray0)):
        if ndarray0[i] != ndarray0[i-1] or ndarray1[i] != ndarray1[i-1]:
            count += 1
            spans[count] = i
    spans[count+1] = len(ndarray0)
    return spans[:count+2]

    
@njit
def _get_spans_for_multi_fields(fields_data):
    count = 0
    length = len(fields_data[0])
    spans = np.zeros(length + 1, dtype = np.uint32)
    spans[0] = 0

    for i in np.arange(1, length):
        not_equal = False
        for f_d in fields_data:
            if f_d[i] != f_d[i - 1]:
                not_equal = True
                break
        
        if not_equal:
            count += 1
            spans[count] = i
        
    spans[count + 1] = length
    return spans[:count + 2]


@njit
def check_if_sorted_for_multi_fields(fields_data):
    """
    Check if input fields data is sorted. Note that fields_data should be treat as a group key

    pre_row[j] < cur_row[j], means these two rows are sorted, move to next row => i + 1
    pre_row[j] = cur_row[j], means we need to check if next element is sorted => j + 1
    pre_row[j] > cur_row[j], means input data is not sorted
    """
    field_count = len(fields_data)

    total_row = len(fields_data[0])
    if total_row == 0:
        return True

    pre_row = fields_data[:, 0]
    for i in range(1, total_row):
        cur_row = fields_data[:, i]

        for j in range(field_count):
            if pre_row[j] > cur_row[j]:
                return False
            elif pre_row[j] < cur_row[j]:
                break

        pre_row = cur_row

    return True

    

@njit
def _get_spans_for_index_string_field(indices,values):
    result = []
    result.append(0)
    for i in range(1, len(indices) - 1):
        last = indices[i - 1]
        current = indices[i]
        next = indices[i + 1]
        if next - current != current - last:  # compare size first
            result.append(i)
            continue
        if not np.array_equal(values[last:current], values[current:next]):
            result.append(i)
    result.append(len(indices) - 1)  # total number of elements
    return result


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
def apply_spans_index_of_min_indexed(spans, src_indices, src_values, dest_array):
    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]

        if next - cur == 1:
            dest_array[i] = cur
        else:
            minind = cur
            minstart = src_indices[cur]
            minend = src_indices[cur+1]
            minlen = minend - minstart
            for j in range(cur+1, next):
                curstart = src_indices[j]
                curend = src_indices[j+1]
                curlen = curend - curstart
                shortlen = min(curlen, minlen)
                found = False
                for k in range(shortlen):
                    if src_values[curstart+k] < src_values[minstart+k]:
                        minind = j
                        minstart = curstart
                        minend = curend
                        found = True
                        break
                    elif src_values[curstart+k] > src_values[minstart+k]:
                        found = True
                        break
                if not found and curlen < minlen:
                    minind = j
                    minstart = curstart
                    minend = curend

            dest_array[i] = minind

    return dest_array


@njit
def apply_spans_index_of_max_indexed(spans, src_indices, src_values, dest_array):
    for i in range(len(spans)-1):
        cur = spans[i]
        next = spans[i+1]

        if next - cur == 1:
            dest_array[i] = cur
        else:
            minind = cur
            minstart = src_indices[cur]
            minend = src_indices[cur+1]
            minlen = minend - minstart
            for j in range(cur+1, next):
                curstart = src_indices[j]
                curend = src_indices[j+1]
                curlen = curend - curstart
                shortlen = min(curlen, minlen)
                found = False
                for k in range(shortlen):
                    if src_values[curstart+k] > src_values[minstart+k]:
                        minind = j
                        minstart = curstart
                        minlen = curend - curstart
                        found = True
                        break
                    elif src_values[curstart+k] < src_values[minstart+k]:
                        found = True
                        break
                if not found and curlen > minlen:
                    minind = j
                    minstart = curstart
                    minlen = curend - curstart

            dest_array[i] = minind

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


# ordered map to left functionality: streaming
# ============================================

def generate_ordered_map_to_left_streamed(left: Field,
                                          right: Field,
                                          l_result: Field,
                                          r_result: Field,
                                          invalid: Union[np.int32, np.int64],
                                          chunksize: Optional[int] = 1 << 20,
                                          rdtype=np.int32):
    """
    This function performs the most generic type of left to right mapping calculation in
    which both key fields can have repeated key values.
    At its heart, the function generates a mapping from left to right that can then be
    used to map data in the right space to data in the left space. Note that this can also
    be used to generate the inverse mapping my simply flipping left and right collections.

    As the Fields ``left`` and ``right`` can contain arbitrarily long sequences of data,
    the data is streamed through the algorithm in a series of chunks. Similarly, the resulting
    map is written to a buffer that is written to the ``result`` field in chunks.

    This streamed function makes a sequence of calls to a corresponding _partial function that
    does the heavy lifting. Inside the _partial function, a finite state machine (FSM) iterates
    over the data, performing the mapping. The _partial function call exits whenever any of the
    chunks (``left_``, ``right_`` or ``result_`` that it is passed become exhausted.

    Please take a look at the documentation for the partial function to understand the finite
    state machine parameters to understand that role that the various parameters play.

    We have to make some adjustments to the finite state machine between calls to _partial:
     * if the call used all the ``left_`` data, add the size of that data chunk to ``i_off``
     * if the call used all of the ``right_`` data, add the size of that data chunk to ``j_off``
     * write the accumulated ``result_`` data to the `result`` field, and reset ``r`` to 0
    """
    # the collection of variables that make up the finite state machine for the calls to
    # partial
    i_off, j_off, i, j, r, ii, jj, ii_max, jj_max, inner = 0, 0, 0, 0, 0, 0, 0, -1, -1, False

    l_result_ = np.zeros(chunksize, dtype=rdtype)
    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_trimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_trimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r, ii, jj, ii_max, jj_max, inner = \
            generate_ordered_map_to_left_partial(left_, i_max, right_, j_max, l_result_, r_result_,
                                                 invalid,
                                                 i_off, j_off, i, j, r,
                                                 ii, jj, ii_max, jj_max, inner)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_trimmed_chunk(left, l_chunk, chunksize)

        # update the right chunk if necessary
        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_trimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    while i + i_off < len(left):
        i, r = generate_ordered_map_to_left_remaining(i_max, l_result_, r_result_, i_off, i, r,
                                                      invalid)

        # update which part of left we are writing for; note we don't need to fetch the data
        # itself as we are mapping left on a 1:1 basis for the rest of its length
        l_chunk = next_chunk(l_chunk[1], len(left), chunksize)
        i_max = l_chunk[1] - l_chunk[0]
        i_off = l_chunk[0]
        i = 0

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    l_result.data.complete()
    r_result.data.complete()


@njit
def generate_ordered_map_to_left_remaining(i_max, l_result, r_result, i_off, i, r, invalid):
    while i < i_max and r < len(l_result):
        l_result[r] = i_off + i
        r_result[r] = invalid
        i += 1
        r += 1
    return i, r


def generate_ordered_map_to_left_left_unique_streamed(left: Field,
                                                      right: Field,
                                                      l_result: Field,
                                                      r_result: Field,
                                                      invalid: Union[np.int32, np.int64],
                                                      chunksize: Optional[int] = 1 << 20,
                                                      rdtype=np.int32):
    # the collection of variables that make up the finite state machine for the calls to
    # partial
    i_off, j_off, i, j, r = 0, 0, 0, 0, 0

    l_result_ = np.zeros(chunksize, dtype=rdtype)
    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_untrimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_trimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r = \
            generate_ordered_map_to_left_left_unique_partial(left_, right_, j_max,
                                                             l_result_, r_result_,
                                                             invalid, i_off, j_off, i, j, r)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_untrimmed_chunk(left, l_chunk, chunksize)

        # update the right chunk if necessary
        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_trimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    while i + i_off < len(left):
        i, r = generate_ordered_map_to_left_remaining(i_max, l_result_, r_result_, i_off, i, r,
                                                      invalid)

        # update which part of left we are writing for; note we don't need to fetch the data
        # itself as we are mapping left on a 1:1 basis for the rest of its length
        l_chunk = next_chunk(l_chunk[1], len(left), chunksize)
        i_max = l_chunk[1] - l_chunk[0]
        i_off = l_chunk[0]
        i = 0

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    l_result.data.complete()
    r_result.data.complete()


def generate_ordered_map_to_left_right_unique_streamed(left: Field,
                                                       right: Field,
                                                       r_result: Field,
                                                       invalid: Union[np.int32, np.int64],
                                                       chunksize: Optional[int] = 1 << 20,
                                                       rdtype=np.int32):
    i_off, j_off, i, j, r = 0, 0, 0, 0, 0

    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_trimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_untrimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r = \
            generate_ordered_map_to_left_right_unique_partial(left_, i_max, right_, r_result_,
                                                              invalid, j_off, i, j, r)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_trimmed_chunk(left, l_chunk, chunksize)

        # update the right chunk if necessary
        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_untrimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            r_result.data.write_part(r_result_[:r])
            r = 0

    while i + i_off < len(left):
        i, r = generate_ordered_map_to_left_right_unique_remaining(i_max, r_result_, i, r, invalid)

        # update the left chunk if necessary
        l_chunk = next_chunk(l_chunk[1], len(left), chunksize)
        i_max = l_chunk[1] - l_chunk[0]
        i_off = l_chunk[0]
        i = 0

        # write the result buffer
        if r > 0:
            r_result.data.write_part(r_result_[:r])
            r = 0

    r_result.data.complete()


def generate_ordered_map_to_left_both_unique_streamed(left: Field,
                                                      right: Field,
                                                      r_result: Field,
                                                      invalid: Union[np.int32, np.int64],
                                                      chunksize: Optional[int] = 1 << 20,
                                                      rdtype=np.int32):
    i_off, j_off, i, j, r = 0, 0, 0, 0, 0

    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_untrimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_untrimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r = \
            generate_ordered_map_to_left_both_unique_partial(left_, right_, r_result_,
                                                             invalid, j_off, i, j, r)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_untrimmed_chunk(left, l_chunk, chunksize)

        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_untrimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            r_result.data.write_part(r_result_[:r])
            r = 0

    while i + i_off < len(left):
        i, r = generate_ordered_map_to_left_right_unique_remaining(i_max, r_result_, i, r, invalid)

        # update the left chunk if necessary
        l_chunk = next_chunk(l_chunk[1], len(left), chunksize)
        i_max = l_chunk[1] - l_chunk[0]
        i_off = l_chunk[0]
        i = 0

        # write the result buffer
        if r > 0:
            r_result.data.write_part(r_result_[:r])
            r = 0

    r_result.data.complete()


@njit
def generate_ordered_map_to_left_partial(left,
                                         i_max,
                                         right,
                                         j_max,
                                         l_result,
                                         r_result,
                                         invalid,
                                         i_off,
                                         j_off,
                                         i,
                                         j,
                                         r,
                                         ii,
                                         jj,
                                         ii_max,
                                         jj_max,
                                         inner):
    """
    This function performs generates a mapping from a subset of a left key to a subset of a
    a right key, writing the resulting mapping to a buffer, where both keys can contain repeated
    entries.

    Example::

      left = [10, 20, 30, 40, 40, 50, 50]
      right = [20, 30, 30, 40, 40, 40, 60, 70]

      i  j op r lres rres
      0  0 <  0  0   INV
      1  0 =  1  1   0
      2  1 =  2  2   1
      2  2    3  2   2
      3  3    4  3   3
      3  4    5  3   4
      3  5    6  3   5
      4  3    7  4   3
      4  4    8  4   4
      4  5    9  4   5
      5  6   10  5   INV
      6  6   11  6   INV


      left_map = [0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
      right_map = [INV, 1, 2, 2, 3, 3, 3, 4, 4, 4, INV, INV]

    Everything about this function is optimised for performance under njit. It is effectively
    a finite state machine that iterates through left, right, and result arrays. The various...

    i and i_max are used to track the index of the left source
    j and j_max are used to track the index of the right source
    """

    while i < i_max and j < j_max and r < len(l_result):
        if inner is False:
            if left[i] < right[j]:
                l_result[r] = i + i_off
                r_result[r] = invalid
                i += 1
                r += 1
            elif left[i] > right[j]:
                j += 1
            else:
                # freeze i for the duration of the loop; i_ tracks
                i_ = i
                cur_i_count = 1
                while i_ + 1 < i_max and left[i_+1] == left[i_]:
                    cur_i_count += 1
                    i_ += 1

                j_ = j
                cur_j_count = 1
                while j_ + 1 < j_max and right[j_+1] == right[j_]:
                    cur_j_count += 1
                    j_ += 1

                ii = 0
                jj = 0
                ii_max = cur_i_count
                jj_max = cur_j_count
                inner = True
        else:
            # TODO: if ii_max * jj_max is > a passed in threshold, raise
            # and error saying the merge is unperformable (say 10,000,000,000)
            l_result[r] = i_off + i + ii
            r_result[r] = j_off + j + jj
            r += 1
            jj += 1
            if jj == jj_max:
                jj = 0
                ii += 1
                if ii == ii_max:
                    i += ii_max
                    j += jj_max
                    inner = False
                    ii = 0
                    jj = 0
                    ii_max = -1
                    jj_max = -1
    return i, j, r, ii, jj, ii_max, jj_max, inner


@njit
def generate_ordered_map_to_left_left_unique_partial(left,
                                                     right,
                                                     j_max,
                                                     l_result,
                                                     r_result,
                                                     invalid,
                                                     i_off,
                                                     j_off,
                                                     i,
                                                     j,
                                                     r):
    while i < len(left) and j < j_max:
        if left[i] < right[j]:
            l_result[r] = i + i_off
            r_result[r] = invalid
            i += 1
            r += 1
        elif left[i] > right[j]:
            j += 1
        else:
            l_result[r] = i + i_off
            r_result[r] = j + j_off
            r += 1
            if j+1 >= j_max or right[j+1] != right[j]:
                i += 1
            j += 1
    return i, j, r


@njit
def generate_ordered_map_to_left_right_unique_partial(left,
                                                      i_max,
                                                      right,
                                                      r_result,
                                                      invalid,
                                                      j_off,
                                                      i,
                                                      j,
                                                      r):
    while i < i_max and j < len(right):
        if left[i] < right[j]:
            r_result[r] = invalid
            i += 1
            r += 1
        elif left[i] > right[j]:
            j += 1
        else:
            r_result[r] = j + j_off
            r += 1
            if i+1 >= i_max or left[i+1] != left[i]:
                j += 1
            i += 1
    return i, j, r


@njit
def generate_ordered_map_to_left_both_unique_partial(left,
                                                     right,
                                                     r_result,
                                                     invalid,
                                                     j_off,
                                                     i,
                                                     j,
                                                     r):
    i_max = len(left)
    j_max = len(right)
    r_max = len(r_result)
    while i < i_max and j < j_max and r < r_max:
        if left[i] < right[j]:
            r_result[r] = invalid
            i += 1
            r += 1
        elif left[i] > right[j]:
            j += 1
        else:
            r_result[r] = j + j_off
            i += 1
            j += 1
            r += 1
    return i, j, r


@njit
def generate_ordered_map_to_left_right_unique_remaining(i_max, r_result, i, r, invalid):
    while i < i_max and r < len(r_result):
        r_result[r] = invalid
        i += 1
        r += 1
    return i, r


def generate_ordered_map_to_left_right_unique_streamed_old(left,
                                                           right,
                                                           left_to_right,
                                                           invalid=-1,
                                                           chunksize=1 << 20):
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
        ii, jj, u = generate_ordered_map_to_left_right_unique_partial_old(j, lc, rc, ltri, invalid)
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
def generate_ordered_map_to_left_right_unique_partial_old(d_j, left, right, left_to_right, invalid):
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
            left_to_right[i] = invalid
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


# ordered map to inner functionality: streaming
# =============================================

def generate_ordered_map_to_inner_streamed(left: Field,
                                           right: Field,
                                           l_result: Field,
                                           r_result: Field,
                                           chunksize: Optional[int] = 1 << 20,
                                           rdtype=np.int32):
    """
    This function performs the most generic type of left to right mapping calculation in
    which both key fields can have repeated key values.
    At its heart, the function generates a mapping from left to right that can then be
    used to map data in the right space to data in the left space. Note that this can also
    be used to generate the inverse mapping my simply flipping left and right collections.

    As the Fields ``left`` and ``right`` can contain arbitrarily long sequences of data,
    the data is streamed through the algorithm in a series of chunks. Similarly, the resulting
    map is written to a buffer that is written to the ``result`` field in chunks.

    This streamed function makes a sequence of calls to a corresponding _partial function that
    does the heavy lifting. Inside the _partial function, a finite state machine (FSM) iterates
    over the data, performing the mapping. The _partial function call exits whenever any of the
    chunks (``left_``, ``right_`` or ``result_`` that it is passed become exhausted.

    Please take a look at the documentation for the partial function to understand the finite
    state machine parameters to understand that role that the various parameters play.

    We have to make some adjustments to the finite state machine between calls to _partial:
     * if the call used all the ``left_`` data, add the size of that data chunk to ``i_off``
     * if the call used all of the ``right_`` data, add the size of that data chunk to ``j_off``
     * write the accumulated ``result_`` data to the `result`` field, and reset ``r`` to 0
    """
    # the collection of variables that make up the finite state machine for the calls to
    # partial
    i_off, j_off, i, j, r, ii, jj, ii_max, jj_max, inner = 0, 0, 0, 0, 0, 0, 0, -1, -1, False

    l_result_ = np.zeros(chunksize, dtype=rdtype)
    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_trimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_trimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r, ii, jj, ii_max, jj_max, inner = \
            generate_ordered_map_to_inner_partial(left_, i_max, right_, j_max, l_result_, r_result_,
                                                  i_off, j_off, i, j, r,
                                                  ii, jj, ii_max, jj_max, inner)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_trimmed_chunk(left, l_chunk, chunksize)

        # update the right chunk if necessary
        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_trimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    l_result.data.complete()
    r_result.data.complete()


def generate_ordered_map_to_inner_left_unique_streamed(left: Field,
                                                       right: Field,
                                                       l_result: Field,
                                                       r_result: Field,
                                                       invalid: Union[np.int32, np.int64],
                                                       chunksize: Optional[int] = 1 << 20,
                                                       rdtype=np.int32):
    # the collection of variables that make up the finite state machine for the calls to
    # partial
    i_off, j_off, i, j, r = 0, 0, 0, 0, 0

    l_result_ = np.zeros(chunksize, dtype=rdtype)
    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_untrimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_trimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r = \
            generate_ordered_map_to_inner_left_unique_partial(left_, i_max, right_, j_max,
                                                              l_result_, r_result_,
                                                              i_off, j_off, i, j, r)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_untrimmed_chunk(left, l_chunk, chunksize)

        # update the right chunk if necessary
        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_trimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    l_result.data.complete()
    r_result.data.complete()


def generate_ordered_map_to_inner_right_unique_streamed(left: Field,
                                                        right: Field,
                                                        l_result: Field,
                                                        r_result: Field,
                                                        invalid: Union[np.int32, np.int64],
                                                        chunksize: Optional[int] = 1 << 20,
                                                        rdtype=np.int32):
    # the collection of variables that make up the finite state machine for the calls to
    # partial
    i_off, j_off, i, j, r = 0, 0, 0, 0, 0

    l_result_ = np.zeros(chunksize, dtype=rdtype)
    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_trimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_untrimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r = \
            generate_ordered_map_to_inner_right_unique_partial(left_, i_max, right_, j_max,
                                                               l_result_, r_result_,
                                                               i_off, j_off, i, j, r)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_trimmed_chunk(left, l_chunk, chunksize)

        # update the right chunk if necessary
        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_untrimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    l_result.data.complete()
    r_result.data.complete()


def generate_ordered_map_to_inner_both_unique_streamed(left: Field,
                                                       right: Field,
                                                       l_result: Field,
                                                       r_result: Field,
                                                       invalid: Union[np.int32, np.int64],
                                                       chunksize: Optional[int] = 1 << 20,
                                                       rdtype=np.int32):
    # the collection of variables that make up the finite state machine for the calls to
    # partial
    i_off, j_off, i, j, r = 0, 0, 0, 0, 0

    l_result_ = np.zeros(chunksize, dtype=rdtype)
    r_result_ = np.zeros(chunksize, dtype=rdtype)

    l_chunk, left_, i_max, i_off, i = first_untrimmed_chunk(left, chunksize)
    r_chunk, right_, j_max, j_off, j = first_untrimmed_chunk(right, chunksize)

    while i + i_off < len(left) and j + j_off < len(right):
        i, j, r = \
            generate_ordered_map_to_inner_both_unique_partial(left_, i_max, right_, j_max,
                                                              l_result_, r_result_,
                                                              i_off, j_off, i, j, r)

        # update the left chunk if necessary
        if i_off + i < len(left) and i >= l_chunk[1] - l_chunk[0]:
            l_chunk, left_, i_max, i_off, i = next_untrimmed_chunk(left, l_chunk, chunksize)

        # update the right chunk if necessary
        if j_off + j < len(right) and j >= r_chunk[1] - r_chunk[0]:
            r_chunk, right_, j_max, j_off, j = next_untrimmed_chunk(right, r_chunk, chunksize)

        # write the result buffer
        if r > 0:
            l_result.data.write_part(l_result_[:r])
            r_result.data.write_part(r_result_[:r])
            r = 0

    l_result.data.complete()
    r_result.data.complete()


@njit
def generate_ordered_map_to_inner_partial(left,
                                          i_max,
                                          right,
                                          j_max,
                                          l_result,
                                          r_result,
                                          i_off,
                                          j_off,
                                          i,
                                          j,
                                          r,
                                          ii,
                                          jj,
                                          ii_max,
                                          jj_max,
                                          inner):
    """
    This function performs generates a mapping from a subset of a left key to a subset of a
    a right key, writing the resulting mapping to a buffer, where both keys can contain repeated
    entries.

    Example::

      left = [10, 20, 30, 40, 40, 50, 50]
      right = [20, 30, 30, 40, 40, 40, 60, 70]

      i  j op r lres rres
      0  0 <  0  0   INV
      1  0 =  1  1   0
      2  1 =  2  2   1
      2  2    3  2   2
      3  3    4  3   3
      3  4    5  3   4
      3  5    6  3   5
      4  3    7  4   3
      4  4    8  4   4
      4  5    9  4   5
      5  6   10  5   INV
      6  6   11  6   INV


      left_map = [0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
      right_map = [INV, 1, 2, 2, 3, 3, 3, 4, 4, 4, INV, INV]

    Everything about this function is optimised for performance under njit. It is effectively
    a finite state machine that iterates through left, right, and result arrays. The various...

    i and i_max are used to track the index of the left source
    j and j_max are used to track the index of the right source
    """

    while i < i_max and j < j_max and r < len(l_result):
        if inner is False:
            if left[i] < right[j]:
                i += 1
            elif left[i] > right[j]:
                j += 1
            else:
                # freeze i for the duration of the loop; i_ tracks
                i_ = i
                cur_i_count = 1
                while i_ + 1 < i_max and left[i_+1] == left[i_]:
                    cur_i_count += 1
                    i_ += 1

                j_ = j
                cur_j_count = 1
                while j_ + 1 < j_max and right[j_+1] == right[j_]:
                    cur_j_count += 1
                    j_ += 1

                ii = 0
                jj = 0
                ii_max = cur_i_count
                jj_max = cur_j_count
                inner = True
        else:
            # TODO: if ii_max * jj_max is > a passed in threshold, raise
            # and error saying the merge is unperformable (say 10,000,000,000)
            l_result[r] = i_off + i + ii
            r_result[r] = j_off + j + jj
            r += 1
            jj += 1
            if jj == jj_max:
                jj = 0
                ii += 1
                if ii == ii_max:
                    i += ii_max
                    j += jj_max
                    inner = False
                    ii = 0
                    jj = 0
                    ii_max = -1
                    jj_max = -1
    return i, j, r, ii, jj, ii_max, jj_max, inner


@njit
def generate_ordered_map_to_inner_left_unique_partial(left,
                                                      i_max,
                                                      right,
                                                      j_max,
                                                      l_result,
                                                      r_result,
                                                      i_off,
                                                      j_off,
                                                      i,
                                                      j,
                                                      r):
    while i < i_max and j < j_max:
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            l_result[r] = i + i_off
            r_result[r] = j + j_off
            r += 1
            if j+1 >= j_max or right[j+1] != right[j]:
                i += 1
            j += 1
    return i, j, r


@njit
def generate_ordered_map_to_inner_right_unique_partial(left,
                                                       i_max,
                                                       right,
                                                       j_max,
                                                       l_result,
                                                       r_result,
                                                       i_off,
                                                       j_off,
                                                       i,
                                                       j,
                                                       r):
    while i < i_max and j < j_max:
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            l_result[r] = i + i_off
            r_result[r] = j + j_off
            r += 1
            if i+1 >= i_max or left[i+1] != left[i]:
                j += 1
            i += 1
    return i, j, r


@njit
def generate_ordered_map_to_inner_both_unique_partial(left,
                                                      i_max,
                                                      right,
                                                      j_max,
                                                      l_result,
                                                      r_result,
                                                      i_off,
                                                      j_off,
                                                      i,
                                                      j,
                                                      r):
    while i < i_max and j < j_max:
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            l_result[r] = i + i_off
            r_result[r] = j + j_off
            r += 1
            i += 1
            j += 1
    return i, j, r


# ordered map to left functionality: non-streaming
# ================================================


@njit
def generate_ordered_map_to_left_right_unique(first, second, result, invalid):
    if len(first) != len(result):
        msg = "'first' and 'result' must be the same length"
        raise ValueError(msg)
    i = 0
    j = 0
    unmapped = 0
    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            result[i] = invalid
            i += 1
            unmapped += 1
        elif first[i] > second[j]:
            j += 1
        else:
            result[i] = j
            if i+1 >= len(first) or first[i+1] != first[i]:
                j += 1
            i += 1

    while i < len(first):
        result[i] = invalid
        i += 1

    return unmapped > 0


@njit
def generate_ordered_map_to_left_both_unique(first, second, result, invalid):
    if len(first) != len(result):
        msg = "'second' and 'result' must be the same length"
        raise ValueError(msg)
    i = 0
    j = 0

    unmapped = 0
    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            result[i] = invalid
            i += 1
            unmapped += 1
        elif first[i] > second[j]:
            j += 1
        else:
            result[i] = j
            i += 1
            j += 1

    while i < len(first):
        result[i] = invalid
        i += 1

    return unmapped > 0


@njit
def ordered_left_map_result_size(left, right):
    i = 0
    j = 0
    result_size = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result_size += 1
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            cur_i_count = 1
            while i + 1 < len(left) and left[i + 1] == left[i]:
                cur_i_count += 1
                i += 1
            cur_j_count = 1
            while j + 1 < len(right) and right[j + 1] == right[j]:
                cur_j_count += 1
                j += 1
            result_size += cur_i_count * cur_j_count
            i += 1
            j += 1
        return result_size

    if i < len(left):
        result_size += left - i


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
            while i + 1 < len(left) and left[i + 1] == left[i]:
                cur_i_count += 1
                i += 1
            cur_j_count = 1
            while j + 1 < len(right) and right[j + 1] == right[j]:
                cur_j_count += 1
                j += 1
            result_size += cur_i_count * cur_j_count
            i += 1
            j += 1
    return result_size


@njit
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

    src_value_chunks = nt.List()
    src_index_chunks = nt.List()

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

        chunk_filter = np.ones(segment_count, dtype=bool)
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
            filtered_value_chunks = nt.List()
            filtered_index_chunks = nt.List()
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


#======== method for transform functions that called in readerwriter.py ==========#

def get_byte_map(string_map):
    """
    Getting byte indices and byte values from categorical key-value pair
    """
    # sort by length of key first, and then sort alphabetically
    sorted_string_map = {k: v for k, v in sorted(string_map.items(), key=lambda item: item[0])}
    sorted_string_key = [(len(k), np.frombuffer(k.encode(), dtype=np.uint8), v) for k, v in sorted_string_map.items()]
    sorted_string_values = list(sorted_string_map.values())
    
    # assign byte_map_key_lengths, byte_map_value
    total_bytes_keys = 0
    byte_map_value = np.zeros(len(sorted_string_map), dtype=np.uint8)

    for i, (length, _, v)  in enumerate(sorted_string_key):
        total_bytes_keys += length
        byte_map_value[i] = v

    # assign byte_map_keys, byte_map_key_indices
    byte_map_keys = np.zeros(total_bytes_keys, dtype=np.uint8)
    byte_map_key_indices = np.zeros(len(sorted_string_map)+1, dtype=np.uint8)
    
    idx_pointer = 0
    for i, (_, b_key, _) in enumerate(sorted_string_key):   
        for b in b_key:
            byte_map_keys[idx_pointer] = b
            idx_pointer += 1

        byte_map_key_indices[i + 1] = idx_pointer  

    byte_map = [byte_map_keys, byte_map_key_indices, byte_map_value]
    return byte_map


@njit           
def categorical_transform(chunk, i_c, column_inds, column_vals, column_offsets, cat_keys, cat_index, cat_values):
    """
    Tranform method for categorical importer in readerwriter.py
    """   
    col_offset = column_offsets[i_c]

    for row_idx in range(len(column_inds[i_c]) - 1):
        if row_idx >= chunk.shape[0]:
            break

        key_start = column_inds[i_c, row_idx]
        key_end = column_inds[i_c, row_idx + 1]
        key_len = key_end - key_start

        for i in range(len(cat_index) - 1):
            sc_key_len = cat_index[i + 1] - cat_index[i]
            if key_len != sc_key_len:
                continue

            index = i
            for j in range(key_len):
                entry_start = cat_index[i]
                if column_vals[col_offset + key_start + j] != cat_keys[entry_start + j]:
                    index = -1
                    break

            if index != -1:
                chunk[row_idx] = cat_values[index]
                

@njit           
def leaky_categorical_transform(chunk, freetext_indices, freetext_values, i_c, column_inds, column_vals, column_offsets, cat_keys, cat_index, cat_values):
    """
    Tranform method for categorical importer in readerwriter.py
    """   
    col_offset = column_offsets[i_c] 

    for row_idx in range(len(column_inds[i_c]) - 1):
        if row_idx >= chunk.shape[0]:   # reach the end of chunk
            break

        key_start = column_inds[i_c, row_idx]
        key_end = column_inds[i_c, row_idx + 1]
        key_len = key_end - key_start

        is_found = False
        for i in range(len(cat_index) - 1):
            sc_key_len = cat_index[i + 1] - cat_index[i]
            if key_len != sc_key_len:
                continue

            index = i
            for j in range(key_len):
                entry_start = cat_index[i]
                if column_vals[col_offset + key_start + j] != cat_keys[entry_start + j]:
                    index = -1
                    break

            if index != -1:
                is_found = True
                chunk[row_idx] = cat_values[index]
                freetext_indices[row_idx + 1] = freetext_indices[row_idx]

        if not is_found:
            chunk[row_idx] = -1 
            freetext_indices[row_idx + 1] = freetext_indices[row_idx] + key_len
            freetext_values[freetext_indices[row_idx]: freetext_indices[row_idx + 1]] = column_vals[col_offset + key_start: col_offset + key_end]


@njit
def numeric_bool_transform(elements, validity, column_inds, column_vals, column_offsets, col_idx, written_row_count,
                           invalid_value, validation_mode, field_name):
    """
    Transform method for numeric importer (bool) in readerwriter.py
    """  
    col_offset = column_offsets[col_idx]  
    exception_message, exception_args = 0, [field_name]     

    for row_idx in range(written_row_count):
        
        empty = False  
        valid_input = True # Start by assuming number is valid
        value = -1 # start by assuming value is -1, the valid result will be 1 or 0 for bool

        row_start_idx = column_inds[col_idx, row_idx]
        row_end_idx = column_inds[col_idx, row_idx + 1]

        length = row_end_idx - row_start_idx

        byte_start_idx, byte_end_idx = 0, length - 1
        # ignore heading whitespace
        while byte_start_idx < length and column_vals[col_offset + row_start_idx + byte_start_idx] == 32:
            byte_start_idx += 1
        # ignore tailing whitespace 
        while byte_end_idx >= 0 and column_vals[col_offset + row_start_idx + byte_end_idx] == 32:
            byte_end_idx -= 1
        
        # actual length after removing heading and trailing whitespace
        actual_length = byte_end_idx - byte_start_idx + 1

        if actual_length <= 0:
            empty = True
            valid_input = False
        else:
        
            val = column_vals[col_offset + row_start_idx + byte_start_idx: col_offset + row_start_idx + byte_start_idx + actual_length]
            if actual_length == 1:
                if val in (49, 89, 121, 84, 116): # '1', 'Y', 'y', 'T', 't'
                    value = 1
                elif val in (48, 78, 110, 70, 102):   # '0', 'N', 'n', 'F', 'f'
                    value = 0
                else:
                    valid_input = False

            elif actual_length == 2:
                if val[0] in (79, 111) and val[1] in (78, 110): # val.lower() == 'on': val[0] in ('O', 'o'), val[1] in ('N', 'n')
                    value = 1
                elif val[0] in (78, 110) and val[1] in (79, 111): # val.lower() == 'no'
                    value = 0
                else:
                    valid_input = False

            elif actual_length == 3:
                if val[0] in (89, 121) and val[1] in (69, 101) and val[2] in (83, 115): # 'yes'
                    value = 1
                elif val[0] in (79, 111) and val[1] in (70, 102) and val[2] in (70, 102): # 'off'
                    value = 0
                else:
                    valid_input = False

            elif actual_length == 4: 
                if val[0] in (84, 116) and val[1] in (82, 114) and val[2] in (85, 117) and val[3] in (69, 101): # 'true'
                    value = 1
                else:
                    valid_input = False

            elif actual_length == 5:
                if val[0] in (70, 102) and val[1] in (65, 97) and val[2] in (76, 108) and val[3] in (83, 115) and val[4] in (69, 101): # 'false'
                    value = 0
                else:
                    valid_input = False
            else:
                valid_input = False


        elements[row_idx] = value if valid_input else invalid_value
        validity[row_idx] = valid_input

        # Optimized exception handling to avoid creating strings inside loop in Numba
        if not valid_input:
            if validation_mode == 'strict':
                if empty:
                    exception_message = 1
                    exception_args = [field_name]
                    break
                else:
                    exception_message = 2
                    non_parsable = column_vals[col_offset + row_start_idx : col_offset + row_end_idx]
                    exception_args = [field_name, non_parsable]
                    break
            if validation_mode == 'allow_empty':
                if not empty:
                    exception_message = 2
                    non_parsable = column_vals[col_offset + row_start_idx : col_offset + row_end_idx]
                    exception_args = [field_name, non_parsable]
                    break
    return exception_message, exception_args      


def raiseNumericException(exception_message, exception_args):
    exceptions = {
        1: "Numeric value in the field '{0}' can not be empty in strict mode",
        2: "The following numeric value in the field '{0}' can not be parsed: {1}"
    }

    raise Exception(exceptions[exception_message].format(
        *[x.tobytes().decode('utf-8').strip() for x in exception_args]
    ))


def transform_int(column_inds, column_vals, column_offsets, col_idx,
                    written_row_count, invalid_value, validation_mode, data_type, field_name):

    widths = column_inds[col_idx, 1:written_row_count + 1] - column_inds[col_idx, :written_row_count]
    width = widths.max()
    elements = np.zeros(written_row_count, 'S{}'.format(width))
    fixed_string_transform(column_inds, column_vals, column_offsets, col_idx,
                           written_row_count, width, elements.data.cast('b'))

    if validation_mode == 'strict':
        try:
          results = elements.astype(data_type)
        except ValueError as e:
            msg = ("Field '{}' contains values that cannot "
                   "be converted to float in '{}' mode").format(field_name, validation_mode)
            raise ValueError(msg) from e
        valids = None
    elif validation_mode == 'allow_empty':
        str_invalid_value = str(invalid_value).encode()
        valids = np.char.not_equal(elements, b'')
        results = np.where(valids, elements, str_invalid_value)
        try:
            results = results.astype(data_type)
        except ValueError as e:
            msg = ("Field '{}' contains values that cannot "
                   "be converted to float in '{}' mode").format(field_name, validation_mode)
            raise ValueError(msg) from e
    elif validation_mode == 'relaxed':
        results = np.zeros(written_row_count, dtype=data_type)
        valids = np.ones(written_row_count, dtype=bool)
        for i in range(written_row_count):
            try:
                value, valid = int(elements[i]), True
            except:
                value, valid = invalid_value, False
            results[i] = value
            valids[i] = valid
    else:
        raise ValueError("'{}' is not a valid value for 'validation_mode'")

    return results, valids


def transform_float(column_inds, column_vals, column_offsets, col_idx,
                      written_row_count, invalid_value, validation_mode, data_type, field_name):

    widths = column_inds[col_idx, 1:written_row_count + 1] - column_inds[col_idx, :written_row_count]
    width = widths.max()
    elements = np.zeros(written_row_count, 'S{}'.format(width))
    fixed_string_transform(column_inds, column_vals, column_offsets, col_idx,
                           written_row_count, width, elements.data.cast('b'))

    if validation_mode == 'strict':
        try:
          results = elements.astype(data_type)
        except ValueError as e:
            msg = ("Field '{}' contains values that cannot "
                   "be converted to float in '{}' mode").format(field_name, validation_mode)
            raise ValueError(msg) from e
        valids = None
    elif validation_mode == 'allow_empty':
        str_invalid_value = str(invalid_value).encode()
        valids = np.char.not_equal(elements, b'')
        results = np.where(valids, elements, str_invalid_value)
        try:
            results = results.astype(data_type)
        except ValueError as e:
            msg = ("Field '{}' contains values that cannot "
                   "be converted to float in '{}' mode").format(field_name, validation_mode)
            raise ValueError(msg) from e
    elif validation_mode == 'relaxed':
        results = np.zeros(written_row_count, dtype=data_type)
        valids = np.ones(written_row_count, dtype=bool)
        for i in range(written_row_count):
            try:
                value, valid = float(elements[i]), True
            except:
                value, valid = invalid_value, False
            results[i] = value
            valids[i] = valid
    else:
        raise ValueError("'{}' is not a valid value for 'validation_mode'")

    return results, valids


@njit
def transform_to_values(column_inds, column_vals, column_offsets, col_idx, written_row_count):
    """
    Trasnform method for byte data from np.int to np.bytes_
    """
    data = []
    col_offset = column_offsets[col_idx]
    for row_idx in range(written_row_count):
        val = column_vals[col_offset + column_inds[col_idx, row_idx]: col_offset + column_inds[col_idx, row_idx + 1]]
        data.append(val)
    return data



@njit
def fixed_string_transform(column_inds, column_vals, column_offsets, col_idx, written_row_count,
                           strlen, memory):
    col_offset = column_offsets[col_idx]
    for i in range(written_row_count):
        a = i * strlen
        start_idx = column_inds[col_idx, i] + col_offset
        end_idx = min(column_inds[col_idx, i+1] + col_offset, start_idx + strlen)
        for c in range(start_idx, end_idx):
            memory[a] = column_vals[c]
            a += 1


@njit
def unique_indexed_string(indices, values):
    unique_result = nt.List([values[indices[0]:indices[1]]])
    lengths_seen = {indices[1] - indices[0]}

    for i in range(1, len(indices)-1):
        length = indices[i+1] - indices[i]
        v = values[indices[i]:indices[i+1]]

        # If we have not seen length of value, we can add it directly
        if length not in lengths_seen:
            lengths_seen.add(length)
            unique_result.append(v)
            continue

        # If we have seen same length before, then compare to existing unique values
        # Can probably be further optimized by only comparing to those with same length
        is_unique = True
        for unique_v in unique_result:
            if np.array_equal(v, unique_v):
                is_unique = False
                break

        if is_unique:
            unique_result.append(v)

    return unique_result
