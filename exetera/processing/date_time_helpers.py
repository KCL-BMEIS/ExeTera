from typing import Optional, Sequence, Tuple, Union
from datetime import datetime, timedelta

import numpy as np

from exetera.core.utils import SECONDS_PER_DAY



def get_periods(start_date: datetime,
                end_date: datetime,
                period: str,
                delta: int = 1
                ) -> Sequence[datetime]:
    """
    Generate a set of periods into which timestamped data can be grouped.
    Delta controls whether the sequence of periods is generated from an start point
    or an end point. When delta is positive, the sequence is generated forwards in time.
    When delta is negative, the sequence is generate backwards in time.
    
    :param start_date: a ``datetime.datetime`` object for the starting period
    :param end_date: a ``datetime.datetime`` object for tne ending period, exclusive
    :param period: a string representing the unit in which the delta is calculated ('day', 'days', 'week', 'weeks')
    :param delta: an integer representing the delta.
    :return: a list of dates
    """

    period_map = {
        'day': lambda x: timedelta(days=x),
        'days': lambda x: timedelta(days=x),
        'week': lambda x: timedelta(weeks=x),
        'weeks': lambda x: timedelta(weeks=x)
    }
    if not isinstance(period, str):
        raise ValueError("'period' must be of type str but is {}".format(type(period)))
    if period not in period_map.keys():
        raise ValueError("'period': must be one of {} but is {}".format(period_map.keys(), period))
    if not isinstance(delta, int):
        raise ValueError("'delta': element 1 must of type int but is {}".format(type(delta)))
    if delta == 0:
        raise ValueError("'delta' cannot be 0")
    if delta < 0:
        if start_date < end_date:
            raise ValueError("'start_date' must be greater than 'end_date' if 'delta' is negative")
    else:
        if start_date > end_date:
            raise ValueError("'start_date' must be less than 'end_date' if 'delta' is positive")

    tdelta = period_map[period](delta)
    if delta > 0:
        dates = [start_date]
        cur_date = start_date + tdelta
        while cur_date <= end_date:
            dates.append(cur_date)
            cur_date += tdelta
    else:
        dates = [start_date]
        cur_date = start_date + tdelta
        while cur_date >= end_date:
            dates.append(cur_date)
            cur_date += tdelta
    return dates


def get_days(date_field: np.ndarray,
             date_filter: Optional[np.ndarray] = None,
             start_date: Optional[np.float64] = None,
             end_date: Optional[np.float64] = None
             ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    This converts a field of timestamps into a field of relative elapsed days.
    The precise behaviour depends on the optional parameters but essentially, the lowest
    valid day is taken as day 0, and all other timestamps are converted to whole numbers
    of days elapsed since this timestamp:
    
    * If ``start_date`` is set, the start_date is used as the zero-date
    * If ``start_date`` is not set:
    
      * If ``date_filter`` is not set, the lowest timestamp is used as the zero-date
      * If ``date_filter`` is set, the lowest unfiltered timestamp is used as the zero-date

    As well as returning the elapsed days, this method can also return a filter for which
    elapsed dates are valid. This is determined as follows:
    
    * If ``date_filter``, ``start_date`` and ``end_date`` are None, None is returned
    * otherwise:
    
      * If ``date_filter`` is not provided, the filter represents all dates that are out
        of range with respect to the start_date and end_date parameters
      * If ``date_filter`` is provided, the filter is all dates out of range with respect to
        the start_date and end_date parameters unioned with the date_filter that was
        passed in

    """
    if not isinstance(date_field, np.ndarray) or date_field.dtype != np.float64:
        raise ValueError("'date_field' must be a numpy array of type np.float64")
    if date_filter is not None:
        if not isinstance(date_filter, np.ndarray) or date_filter.dtype not in (bool, np.int8):
            raise ValueError("'date_filter' must be a numpy array of type bool or np.int8")

    if start_date is None and end_date is None and date_filter is None:
        min_date = date_field.min()
        days = np.floor((date_field - min_date) / SECONDS_PER_DAY).astype(np.int32)
        return days, None
    else:
        in_range = np.ones(len(date_field), dtype=bool) if date_filter is None else date_filter
        if start_date is not None:
            min_date = start_date
            in_range = in_range & (date_field >= start_date)
        else:
            min_date = np.min(date_field if date_filter is None else date_field[date_filter])
        if end_date is not None:
            in_range = in_range & (date_field < end_date)
        days = np.floor((date_field - min_date) / SECONDS_PER_DAY).astype(np.int32)
        return days, in_range


def generate_period_offset_map(periods: Sequence[datetime]
                               ) -> np.ndarray:
    """
    Given a list of ordered datetimes relating to period boundaries, generate a numpy
    array of days that map each day to a period.

    Example::

        [datetime(2020,1,5), datetime(2020,1,12), datatime(2020,1,19), datetime(2020,1,26)]


    generates the following output::
    
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]

    In the above example, each period spans a week, and periods cover a total of 3 weeks.
    As a result, the output is 21 entries long, one for each day covered by the period, and
    matches each day to the corresponding period.
    """
    period_deltas = [(p - periods[0]).days for p in periods]
    period_per_day = np.zeros(period_deltas[-1], dtype=np.int32)
    for i_p in range(len(period_deltas)-1):
        period_per_day[period_deltas[i_p]:period_deltas[i_p + 1]] = i_p
    return period_per_day


def get_period_offsets(periods_by_day: np.ndarray,
                       days: np.ndarray,
                       in_range: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """
    Given a ``periods_by_day``, a numpy array of days mapping to periods and ``days``, a numpy array of days to be mapped to
    periods, perform the mapping to generate a numpy array indicating which period a day is
    in for each element.

    Example::

        periods_by_day: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        days: [3, 18, 4, 7, 10, 0, 0, 2, 19, 20, 16, 17, 19, 4, 5, 9, 8, 15]

    generates the following output::

        [0, 2, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2]

    This function should generally be used in concert with generate_period_offset_map, as follows::

        start_date = # a start date
        end_date = # an end date
        periods = get_periods(start_date, end_date, 'week', 1)
        
        days = get_days(session.get(src['my_table']['my_timestamps']).data[:])
        result = get_period_offsets(generate_period_offset_map(periods), days)

    """
    if not isinstance(periods_by_day, np.ndarray) and\
            periods_by_day.dtype not in (np.int8, np.int16, np.int32, np.int64):
        raise ValueError("'periods_by_day' must be a numpy array of a signed integer type")
    if not isinstance(days, np.ndarray) and\
            days.dtype not in (np.int8, np.int16, np.int32, np.int64):
        raise ValueError("'days' must be a numpy array of a signed integer type")

    if in_range is None:
        periods = periods_by_day[days]
    else:
        periods = np.where(in_range, days, 0)
        periods = periods_by_day[periods]
        periods = np.where(in_range, periods, -1)

    return periods
