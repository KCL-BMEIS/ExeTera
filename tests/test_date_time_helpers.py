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


    def test_periods_day_periods(self):

        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 9), ValueError),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), [D(2020, 5, 10), D(2020, 5, 11)])
        ]
        self._do_scenario_test(scenarios, 'day', 1)
        self._do_scenario_test(scenarios, 'days', 1)

        scenarios = [
            S(D(2020, 5, 10), D(2020, 5, 9), [D(2020, 5, 10), D(2020, 5, 9)]),
            S(D(2020, 5, 10), D(2020, 5, 10), [D(2020, 5, 10)]),
            S(D(2020, 5, 10), D(2020, 5, 11), ValueError)
        ]
        self._do_scenario_test(scenarios, 'day', -1)
        self._do_scenario_test(scenarios, 'days', -1)

    def test_periods_multi_day_periods(self):

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

    def test_periods_week_periods(self):

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


    def test_periods_multi_week_periods(self):

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

    def test_get_days(self):
        dt = D(2020, 5, 10, 23, 55, 0, 123400)
        print(dt, dt.timestamp())
        tss =\
            np.asarray([(dt + T(seconds=30000 * i)).timestamp() for i in range(11)],
                       dtype=np.float64)
        expected = (tss // 86400).tolist()
        actual = dth.get_days(tss)
        self.assertListEqual(expected, actual[0].tolist())
