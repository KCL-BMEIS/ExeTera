import unittest
from exetera.core import dataset
from exetera.core import session
from io import BytesIO

class TestDataSet(unittest.TestCase):

    def test_dataset_init(self):
        bio=BytesIO()
        with session.Session() as s:
            dst=s.open_dataset(bio,'r+','dst')
            df=dst.create_dataframe('df')
            num=s.create_numeric(df,'num','int32')
            num.data.write([1,2,3,4])
            self.assertEqual([1,2,3,4],num.data[:].tolist())

            num2=s.create_numeric(df,'num2','int32')
            num2 = s.get(df['num2'])

    def test_dataset_ops(self):
        pass
