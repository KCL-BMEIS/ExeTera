from typing import Sequence, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike

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
    :param period: a string representing the unit in which the delta is calculated
    ('day', 'days', 'week', 'weeks')
    :param delta: an integer representing the delta.
    :return:
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


def get_days(date_field: ArrayLike[np.float64],
             date_filter: ArrayLike[bool] = None,
             start_date: np.float64 = None,
             end_date: np.float64 = None
             ) -> Tuple[ArrayLike[np.int32], Union[ArrayLike[bool], None]]:
    """
    get_days converts a field of timestamps into a field of relative elapsed days.
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
                               ) -> ArrayLike[np.int32]:
    """
    Given a list of ordered datetimes relating to period boundaries, generate a numpy
    array of days that map each day to a period.

    Example:

    .. code-block:: python

      [datetime(2020,1,5), datetime(2020,1,12), datatime(2020,1,19), datetime(2020,1,26)]


    generates the following output

    .. code-block:: python

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


def get_period_offsets(periods_by_day: ArrayLike[np.int32],
                       days: ArrayLike[np.int32]
                       ) -> ArrayLike[np.int32]:
    """
    Given a ``periods_by_day``, a numpy array of days mapping to periods and ``days``, a numpy array of days to be mapped to
    periods, perform the mapping to generate a numpy array indicating which period a day is
    in for each element.

    Example:

    .. code-block:: python

      periods_by_day: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
      days: [3, 18, 4, 7, 10, 0, 0, 2, 19, 20, 16, 17, 19, 4, 5, 9, 8, 15]

    generates the following output:

    .. code-block:: python
      [0, 2, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2]

    This function should generally be used in concert with generate_period_offset_map, as follows:

    .. code-block:: python

      start_date = # a start date
      end_date = # an end date
      periods = get_periods(start_date, end_date, 'week', 1)

      days = get_days(session.get(src['my_table']['my_timestamps']).data[:])
      result = get_period_offsets(generate_period_offset_map(periods), days)

    """
    periods = periods_by_day[days]
    return periods
