from typing import Tuple, Union
from datetime import datetime, timedelta

import numpy as np

from exetera.core.utils import SECONDS_PER_DAY



def get_periods(
        start_date: datetime,
        end_date: datetime,
        period: str,
        delta: int = 1):
    """
    Generate a set of periods into which timestamped data can be grouped.
    Delta controls whether the sequence of periods is generated from an start point
    or an end point. When delta is positive, the sequence is generated forwards in time.
    When delta is negative, the sequence is generate backwards in time.
    :param start_date: a datetime.datetime object for the starting period
    :param end_date: a datetime.datetime object for tne ending period, exclusive
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


def get_days(date_field: np.array,
             date_filter: np.array = None,
             start_date: np.float64 = None,
             end_date: np.float64 = None)\
        -> Union[Tuple[np.array, None], Tuple[np.array, np.array]]:
    """
    get_days converts a field of timestamps into a field of relative elapsed days.
    The precise behaviour depends on the optional parameters but essentially, the lowest
    valid day is taken as day 0, and all other timestamps are converted to whole numbers
    of days elapsed since this timestamp:
    * If start_date is set, the start_date is used as the zero-date
    * If start_date is not set:
      * If date_filter is not set, the lowest timestamp is used as the zero-date
      * If date_filter is set, the lowest unfiltered timestamp is used as the zero-date

    As well as returning the elapsed days, this method can also return a filter for which
    elapsed dates are valid. This is determined as follows:
    * If date_filter, start_date and end_date are None, None is returned
    * otherwise:
      * If date_filter is not provided, the filter represents all dates that are out
        of range with respect to the start_date and end_date parameters
      * If date_filter is provided, the filter is all dates out of range with respect to
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


def get_period_offsets_by_day(periods):
    period_deltas = [(p - periods[0]).days for p in periods]
    period_per_day = np.zeros(period_deltas[-1], dtype=np.int32)
    for i_p in range(len(period_deltas)-1):
        period_per_day[period_deltas[i_p]:period_deltas[i_p + 1]] = i_p
    return period_per_day


def get_period_offsets_by_day(periods_by_day, days):
    periods = periods_by_day[days]
    return periods
