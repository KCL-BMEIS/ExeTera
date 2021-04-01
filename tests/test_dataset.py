import unittest
from exetera.core import dataset
from exetera.core import session
from io import BytesIO

class TestDataSet(unittest.TestCase):
    def TestDataSet_init(self):
        bio=BytesIO()
        with session.Session() as s:
            dst=s.open_dataset(bio,'r+','dst')
            




