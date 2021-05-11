import unittest
from exetera.core.load_schema import NewDataSchema 

class TestLoadSchema(unittest.TestCase):
    def test__get_min_max_for_permitted_types(self):
        permitted_numeric_types = ('float32', 'float64', 'int8', 'uint8', 'int16', 'uint16', 
                                    'int32', 'uint32', 'int64')
        expected_min_max_values = {
            'float32': (-2147483648, 2147483647),
            'float64': (-9223372036854775808, 9223372036854775807),
            'int8': (-128, 127),
            'uint8': (0, 255),
            'int16': (-32768, 32767),
            'uint16': (0, 65535),
            'int32': (-2147483648, 2147483647),
            'uint32': (0, 4294967295),
            'int64': (-9223372036854775808, 9223372036854775807)
        }
        for value_type in permitted_numeric_types:
            (min_value, max_value) = NewDataSchema._get_min_max(value_type)
            self.assertEqual(min_value, expected_min_max_values[value_type][0])
            self.assertEqual(max_value, expected_min_max_values[value_type][1])