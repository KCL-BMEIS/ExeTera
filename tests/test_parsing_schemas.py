import unittest


import parsing_schemas

class MockDataset:
    def __init__(self):
        self.fields_ = None
        self.index_ = None
        self.names_ = None

    def field_to_index(self, field):
        return self.names_.index(field)

class TestParsingSchemas(unittest.TestCase):

    def test_validate_covid_test_results_version_2_no_to_yes(self):
        dataset = MockDataset()
        dataset.fields_ = [['a', 'x', 'no'],
                           ['b', 'x', 'yes']]
        dataset.index_ = [x for x in range(len(dataset.fields_))]
        dataset.names_ = ['id', 'patient_id', 'tested_covid_positive']

        filter_status = [0] * len(dataset.index_)
        results = [0] * len(dataset.index_)
        fn = parsing_schemas.ValidateCovidTestResultsFacVersion2(dataset, filter_status, None, results, 0x1)
        fn(dataset.fields_, filter_status, 0, len(dataset.fields_) - 1)

    def test_validate_covid_test_results_version_2_na_waiting_no_waiting(self):
        dataset = MockDataset()
        dataset.fields_ = [['a', 'x', ''],
                           ['b', 'x', 'waiting'],
                           ['c', 'x', 'no'],
                           ['d', 'x', 'waiting']]
        dataset.index_ = [x for x in range(len(dataset.fields_))]
        dataset.names_ = ['id', 'patient_id', 'tested_covid_positive']

        filter_status = [0] * len(dataset.index_)
        results = [0] * len(dataset.index_)
        fn = parsing_schemas.ValidateCovidTestResultsFacVersion2(dataset, filter_status, None, results, 0x1)
        fn(dataset.fields_, filter_status, 0, len(dataset.fields_) - 1)