import unittest
import io
import dataset

small_dataset = ('id,patient_id,foo\n'
                 '0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,11111111111111111111111111111111,,\n'
                 '07777777777777777777777777777777,33333333333333333333333333333333,True,\n'
                 '02222222222222222222222222222222,11111111111111111111111111111111,False,\n')

class TestDataset(unittest.TestCase):

    def test_construction(self):
        s = io.StringIO(small_dataset)
        ds = dataset.Dataset(s)

        # field names and fields must match in length
        self.assertEqual(len(ds.names_), len(ds.fields_))

        self.assertEqual(ds.row_count(), 3)

        self.assertEqual(ds.names_, ['id', 'patient_id', 'foo'])

        expected_values = [['0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', '11111111111111111111111111111111', ''],
                           ['07777777777777777777777777777777', '33333333333333333333333333333333', 'True'],
                           ['02222222222222222222222222222222', '11111111111111111111111111111111', 'False']]

        # value works as expected
        for row in range(len(expected_values)):
            for col in range(len(expected_values[0])):
                self.assertEqual(ds.value(row, col), expected_values[row][col])

        # value_from_fieldname works as expected
        sorted_names = sorted(ds.names_)
        for n in sorted_names:
            index = ds.names_.index(n)
            for row in range(len(expected_values)):
                self.assertEqual(ds.value_from_fieldname(row, n), expected_values[row][index])
