
from parameterized import parameterized

from exetera.core import fields

import numpy as np

from .utils import SessionTestCase, shuffle_randstate, allow_slow_tests, DEFAULT_FIELD_DATA

NUMERIC_ONLY=[d for d in DEFAULT_FIELD_DATA if d[0]=="create_numeric"]

class TestDefaultData(SessionTestCase):
    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_fields(self,creator,name,kwargs,data):
        f=self.setup_field(self.df,creator,name,(),kwargs,data)        
        self.assertFieldEqual(data,f)


# replaces TestFieldArray in test_fields.py
class TestFieldArray(SessionTestCase):

    @parameterized.expand(NUMERIC_ONLY)
    def test_write_part(self,creator,name,kwargs,data):
        f = self.s.create_numeric(self.df, name, **kwargs)
        f.data.write_part(data)
        self.assertFieldEqual(data,f)
        
    @parameterized.expand(NUMERIC_ONLY)
    def test_clear(self,creator,name,kwargs,data):
        f = self.s.create_numeric(self.df, name, **kwargs)
        f.data.write_part(data)
        f.data.clear()
        self.assertFieldEqual([],f)
        
        
REALLY_LARGE_LIST = list(range(1_000_000))
NUMERIC_ISIN_TESTS=[
    ("int16", [1,2,3,4,5],[],[False,False,False, False, False]), 
    ("int16", [1,2,3,4,5],[6,7],[False,False,False, False, False]),
    ("int16", [1,2,3,4,5],[1,2,3],[True, True, True, False, False]),
    ("int16", [1,2,3,4,5],[1,2,3,6,7],[True, True, True, False, False]),
    ("int16", [1,2,3,4,5],[4,1,3],[True, False, True, True, False]),
    ("int16", [3,1,5,4,2],[4,1,3],[True, True, False, True, False]),
    ("int16", [1,2,3,4,5],3,[False, False, True, False, False]),
    ("int16", [3,1,5,4,2],4,[False, False, False, True, False]),
    ("int32", REALLY_LARGE_LIST,REALLY_LARGE_LIST,[True]*len(REALLY_LARGE_LIST)),
    ("int32", REALLY_LARGE_LIST,shuffle_randstate(REALLY_LARGE_LIST),[True]*len(REALLY_LARGE_LIST)),
]

INDEX_STR_DATA=['a', '', 'apple','app', 'APPLE', 'APP', "aaaa",'app/', 'apple12', 'ip']
INDEX_STR_ISIN_TESTS=[
    # (INDEX_STR_DATA,[],[False, False, False, False, False, False, False, False, False, False]), # ERROR: raises exception rather than return all Falses
    # (INDEX_STR_DATA,None,[False, False, False, False, False, False, False, False, False, False]), # ERROR: raises exception from too far down call stack
    # (INDEX_STR_DATA,[None],[False, False, False, False, False, False, False, False, False, False]), # ERROR: raises exception from too far down call stack
    (INDEX_STR_DATA,["None"],[False, False, False, False, False, False, False, False, False, False]),
    (INDEX_STR_DATA,["a", "APPLE"],[True, False, False, False, True, False, False, False, False, False]),
    (INDEX_STR_DATA,["app","APP"],[False, False, False, True, False, True, False, False, False, False]),
    (INDEX_STR_DATA,["app/","app//"],[False, False, False, False, False, False, False, True, False, False]),
    (INDEX_STR_DATA,["apple12","APPLE12", "apple13"],[False, False, False, False, False, False, False, False, True, False]),
    (INDEX_STR_DATA,["ip","ipd", "id"],[False, False, False, False, False, False, False, False, False, True]),
    (INDEX_STR_DATA,[""],[False, True, False, False, False, False, False, False, False, False]),
    (INDEX_STR_DATA,INDEX_STR_DATA,[True, True, True, True, True, True, True, True, True, True]),
]

class TestFieldIsIn(SessionTestCase):
    @parameterized.expand(DEFAULT_FIELD_DATA)
    def test_isin_default_fields(self,creator,name,kwargs,data):
        f=self.setup_field(self.df,creator,name,(),kwargs,data) 
        
        with self.subTest("Test empty isin parameter"):
            expected=[False]*len(data)
            result=f.isin([])
            np.testing.assert_array_equal(expected,result)
            
        with self.subTest("Test 1 isin value"):
            for idx in range(len(data)):
                expected=[i==idx for i in range(len(data))]
                result=f.isin([data[idx]])
                np.testing.assert_array_equal(expected,result)
                
#     @parameterized.expand(NUMERIC_ISIN_TESTS)
#     def test_module_field_isin(self, dtype,data, isin_data, expected):
#         f=self.setup_field(self.df,"create_numeric","f",(dtype,),{},data)
        
#         with self.subTest("Test module function"):
#             result=fields.isin(f,isin_data)

#             self.assertIsInstance(result, fields.NumericMemField)
#             self.assertFieldEqual(expected,result)

#         with self.subTest("Test field method"):
#             result=f.isin(isin_data)
        
#             self.assertIsInstance(result, np.ndarray)
#             self.assertIsInstance(expected, list)
#             self.assertFieldEqual(expected,result)
        
#     @parameterized.expand(INDEX_STR_ISIN_TESTS)
#     def test_indexed_string_isin(self, data, isin_data, expected):
#         f=self.setup_field(self.df,"create_indexed_string","f",(),{},data)
        
#         with self.subTest("Test with given data"):
#             result=f.isin(isin_data)

#             self.assertIsInstance(result, list)
#             self.assertEqual(expected,result)
            
#         with self.subTest("Test with duplicate data"):
#             isin_data=shuffle_randstate(isin_data*2)  # duplicate the search items and shuffle using a fixed seed
        
#             result=f.isin(isin_data)

#             self.assertIsInstance(result, list)
#             self.assertEqual(expected,result)
        
    

    
    
    
    
    
    
#     def test_isin_on_numeric_field(self):
#         bio = BytesIO()
#         with session.Session() as s:
#             src = s.open_dataset(bio, 'w', 'src')
#             df = src.create_dataframe('df')
#             df.create_numeric('f', 'int16').data.write([1, 2, 3, 4, 5])

#             # test_element is list
#             self.assertEqual(df['f'].isin([1,2,3]).tolist(), [True, True, True, False, False])
#             # single test_element
#             self.assertEqual(df['f'].isin(3).tolist(), [False, False, True, False, False])
#             self.assertEqual(df['f'].isin(8).tolist(), [False, False, False, False, False])


#     def test_isin_on_indexed_string_field_with_testelements_all_unique(self):
#         bio = BytesIO()
#         with session.Session() as s:
#             src = s.open_dataset(bio, 'w', 'src')
#             df = src.create_dataframe('df')
#             df.create_indexed_string('foo').data.write(['a', '', 'apple','app', 'APPLE', 'APP', 'app/', 'apple12', 'ip'])

#             self.assertEqual(df['foo'].isin(['APPLE', '']), [False, True, False, False, True, False, False, False, False])
#             self.assertEqual(df['foo'].isin(['app','APP']), [False, False, False, True, False, True, False, False, False])
#             self.assertEqual(df['foo'].isin(['app/','app//']), [False, False, False, False, False, False, True, False, False])
#             self.assertEqual(df['foo'].isin(['apple12','APPLE12', 'apple13']), [False, False, False, False, False, False, False, True, False])
#             self.assertEqual(df['foo'].isin(['ip','ipd']), [False, False, False, False, False, False, False, False, True])


#     def test_isin_on_indexed_string_field_with_duplicate_in_testelements(self):
#         bio = BytesIO()
#         with session.Session() as s:
#             src = s.open_dataset(bio, 'w', 'src')
#             df = src.create_dataframe('df')
#             df.create_indexed_string('foo').data.write(['a', '', 'apple','app', 'APPLE', 'APP', 'app/', 'apple12', 'ip'])

#             self.assertEqual(df['foo'].isin(['APPLE', '', '', 'APPLE']), [False, True, False, False, True, False, False, False, False])
#             self.assertEqual(df['foo'].isin(['app','APP', 'app', 'APP']), [False, False, False, True, False, True, False, False, False])
#             self.assertEqual(df['foo'].isin(['app/','app//', 'app//']), [False, False, False, False, False, False, True, False, False])
#             self.assertEqual(df['foo'].isin(['APPLE12', 'apple12', 'apple12', 'APPLE12', 'apple13']), [False, False, False, False, False, False, False, True, False])
#             self.assertEqual(df['foo'].isin(['ip','ipd', 'id']), [False, False, False, False, False, False, False, False, True])


#     def test_isin_on_fixed_string_field(self):
#         bio = BytesIO()
#         with session.Session() as s:
#             src = s.open_dataset(bio, 'w', 'src')
#             df = src.create_dataframe('df')
#             df.create_fixed_string('foo', 2).data.write([b'aa', b'bb', b'cc'])

#             self.assertEqual(df['foo'].isin([b'aa', b'zz']).tolist(), [True, False, False])

#     def test_isin_on_timestamp_field(self):
#         bio = BytesIO()
#         with session.Session() as s:
#             src = s.open_dataset(bio, 'w', 'src')
#             df = src.create_dataframe('df')

#             ts1 = datetime(2021, 12, 1).timestamp()
#             ts2 = datetime(2022, 1, 1).timestamp()
#             ts3 = datetime(2022, 2, 1).timestamp()
#             df.create_timestamp('ts').data.write([ts2, ts3, ts1])
#             self.assertEqual(df['ts'].isin({ts1, ts2}).tolist(), [True, False, True])