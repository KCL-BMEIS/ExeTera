from datetime import datetime, timedelta

import numpy as np

from exetera.core.utils import SECONDS_PER_DAY



def get_periods(start_date, end_date, period, delta=1):
    """
    Generate a set of periods by which data can be quantised.
    :param start_date: a datetime.datetime object for the starting period
    :param end_date:
    :param period:
    :param delta:
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
        # dates.reverse()
    return dates


def get_days(date_field, date_filter=None, start_date=None, end_date=None):
    if start_date is None and end_date is None and date_filter is None:
        min_date = np.min(date_field // SECONDS_PER_DAY * SECONDS_PER_DAY)
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
        days = np.where(in_range, np.floor((date_field - min_date) / 86400).astype(np.int32), 0)
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
