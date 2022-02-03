import unittest

from datetime import datetime as D
from datetime import timedelta as T

import numpy as np

from exetera.processing import date_time_helpers as dth


class S:
    def __init__(self, start, end, expected=None, message=None):
        self.start = start
        self.end = end
        self.expected = expected
        self.message = message


class TestDateTimeHelpers(unittest.TestCase):

    def _do_scenario_test(self, scenarios, period, count):
        for s in scenarios:
            if isinstance(s.expected, (list, tuple)):
                actual = dth.get_periods(s.start, s.end, period, count)
                err = "{}, {}, {}, {}".format(s.start, s.end, period, count)
                self.assertListEqual(s.expected, actual, err)
            elif issubclass(type(s.expected), BaseException):
                if s.message is not None:
                    with self.assertRaises(s.expected, msg=s.message):
                        dth.get_periods(s.start, s.end, period, count)
                with self.assertRaises(s.expected):
                    dth.get_periods(s.start, s.end, period, count)

    def test_periods_zero_count(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 11), ValueError, "'delta' cannot be 0")
        ]
        self._do_scenario_test(scenarios, 'day', 0)

    def test_periods_day_positive_delta(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 9), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), [D(2020, 5, 10), D(2020, 5, 11)])
        ]
        self._do_scenario_test(scenarios, 'day', 1)
        self._do_scenario_test(scenarios, 'days', 1)

    def test_periods_day_negative_delta(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 9), [D(2020, 5, 10), D(2020, 5, 9)]),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), ValueError)
        ]
        self._do_scenario_test(scenarios, 'day', -1)
        self._do_scenario_test(scenarios, 'days', -1)

    def test_periods_multi_day_positive_delta(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 9), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 12), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 13), [D(2020, 5, 10), D(2020, 5, 13)]),
            S(D(2020, 5, 25), D(2020, 6, 3),
              [D(2020, 5, 25), D(2020, 5, 28), D(2020, 5, 31), D(2020, 6, 3)])
        ]
        self._do_scenario_test(scenarios, 'day', 3)

    def test_periods_multi_day_negative_delta(self):
        scenarios = [
            S(D(2020, 5, 5), D(2020, 4, 25),
              [D(2020, 5, 5), D(2020, 5, 2), D(2020, 4, 29), D(2020, 4, 26)]),
            S(D(2020, 5, 10), D(2020, 5, 7), [D(2020, 5, 10), D(2020, 5, 7)]),
            S(D(2020, 5, 10), D(2020, 5, 8), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 9), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), ValueError),
        ]
        self._do_scenario_test(scenarios, 'day', -3)

    def test_periods_week_positive_delta(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 3), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 9), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 16), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 17), [D(2020, 5, 10), D(2020, 5, 17)]),
            S(D(2020, 5, 10), D(2020, 5, 18), [D(2020, 5, 10), D(2020, 5, 17)])
        ]
        self._do_scenario_test(scenarios, 'day', 7)
        self._do_scenario_test(scenarios, 'days', 7)
        self._do_scenario_test(scenarios, 'week', 1)
        self._do_scenario_test(scenarios, 'weeks', 1)

    def test_periods_week_negative_delta(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 2), [D(2020, 5, 10), D(2020, 5, 3)]),
            S(D(2020, 5, 10), D(2020, 5, 3), [D(2020, 5, 10), D(2020, 5, 3)]),
            S(D(2020, 5, 10), D(2020, 5, 4), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 9), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 17), ValueError)
        ]
        self._do_scenario_test(scenarios, 'day', -7)
        self._do_scenario_test(scenarios, 'days', -7)
        self._do_scenario_test(scenarios, 'week', -1)
        self._do_scenario_test(scenarios, 'weeks', -1)


    def test_periods_multi_week_positive_delta(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 3), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 9), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 6, 6), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 6, 7), [D(2020, 5, 10), D(2020, 6, 7)]),
            S(D(2020, 5, 10), D(2020, 6, 8), [D(2020, 5, 10), D(2020, 6, 7)])
        ]
        self._do_scenario_test(scenarios, 'day', 28)
        self._do_scenario_test(scenarios, 'days', 28)
        self._do_scenario_test(scenarios, 'week', 4)
        self._do_scenario_test(scenarios, 'weeks', 4)

    def test_periods_multi_week_negative_delta(self):
        scenarios = [
            S(D(2020, 5, 10), D(2020, 4, 11), [D(2020, 5, 10), D(2020, 4, 12)]),
            S(D(2020, 5, 10), D(2020, 4, 12), [D(2020, 5, 10), D(2020, 4, 12)]),
            S(D(2020, 5, 10), D(2020, 4, 13), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 9), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), ValueError),
            S(D(2020, 5, 10), D(2020, 6, 7), ValueError)
        ]
        self._do_scenario_test(scenarios, 'day', -28)
        self._do_scenario_test(scenarios, 'days', -28)
        self._do_scenario_test(scenarios, 'week', -4)
        self._do_scenario_test(scenarios, 'weeks', -4)


class TestGetDays(unittest.TestCase):

    def _setup_timestamps(self, start, delta, count):

        return np.asarray([(start + T(seconds=delta * i)).timestamp()
                         for i in range(count)], dtype=np.float64)

    def _get_expected(self, timestamps, filter_field=None, start_date=None, end_date=None):
        if start_date is not None:
            timestamps = ((timestamps - start_date) // 86400).astype(np.int32)
        else:
            if filter_field is not None:
                timestamps =\
                    ((timestamps - timestamps[np.argmax(filter_field)]) // 86400)\
                    .astype(np.int32)
                # timestamps = np.where(filter_field != 0, timestamps, 0)
            else:
                timestamps = ((timestamps - timestamps.min()) // 86400).astype(np.int32)
        return timestamps

    def test_get_days(self):
        tss = self._setup_timestamps(D(2020, 5, 10, 23, 55, 0, 123500), 30000, 11)
        actual = dth.get_days(tss)
        expected = self._get_expected(tss).tolist()
        self.assertListEqual(expected, actual[0].tolist())

    def test_get_days_filtered(self):
        tss = self._setup_timestamps(D(2020, 5, 10, 23, 55, 0, 123500), 30000, 11)
        filter_field = np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        actual = dth.get_days(tss, filter_field)
        expected = self._get_expected(tss, filter_field).tolist()
        self.assertListEqual(expected, actual[0].tolist())

    def test_get_days_delimiting_dates(self):
        tss = self._setup_timestamps(D(2020, 5, 10, 23, 55, 0, 123500), 30000, 11)
        start_date = D(2020, 5, 11).timestamp()
        actual = dth.get_days(tss, start_date=start_date)
        expected = self._get_expected(tss, start_date=start_date)
        self.assertListEqual(expected.tolist(), actual[0].tolist())


class TestGeneratePeriodOffsetMap(unittest.TestCase):

    def test_generate_period_offset_map(self):
        start_dt = D(2020, 3, 1)
        end_dt = D(2021, 3, 1)
        periods = dth.get_periods(end_dt, start_dt, 'week', -1)
        periods.reverse()
        #print(periods)


class TestGetPeriodOffsets(unittest.TestCase):

    def test_get_period_offsets_with_out_of_range(self):
        periods_by_day = np.asarray([0, 0, 0, 1, 1, 1, 2], dtype=np.int32)
        in_range = np.asarray([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        days = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        with self.assertRaises(IndexError):
            dth.get_period_offsets(periods_by_day, days)
        actual = dth.get_period_offsets(periods_by_day, days, in_range)
        expected = np.asarray([0, 0, 0, 1, 1, 1, 2, -1])
        self.assertListEqual(actual.tolist(), expected.tolist())
