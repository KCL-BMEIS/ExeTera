import sys

from numpy.lib import delete
sys.path.append("/Users/lc21/Documents/KCL_BMEIS/ExeTera")   

import exetera
from exetera.core.session import Session
import numpy as np
import time
import pandas

source = 'resources/new_vaccine_doses.hdf5'
schema_key = 'vaccine_doses'

to_csv_file= 'resources/vaccine_doses_to_csv.csv'
pandas_read_csv_file = 'resources/vacc_doses_export_20210825040021.csv'

# source = 'resources/new_vaccine_symptoms.hdf5'
# schema_key = 'vaccine_symptoms'

# source = 'resources/new_vaccine_hesitancy.hdf5'
# schema_key = 'vaccine_hesitancy'

# source = 'resources/new_mental_health.hdf5'
# schema_key = 'mental_health'

# source = 'resources/new_patients.hdf5'
# to_csv_file= 'resources/patients_to_csv.csv'
# schema_key = 'patients'

# source = 'resources/new_diet.hdf5'
# schema_key = 'diet'

# source = 'resources/new_tests.hdf5'
# schema_key = 'tests'

# source = 'resources/new_assessments.hdf5'
# schema_key = 'assessments'


def to_csv_writerow_benchmark():

    with Session() as s:
        print("==========")
        print(source)

        ds = s.open_dataset(source, 'r', 'ds')
        df = ds[schema_key]

        length = len(df[list(df.keys())[0]])
        print('length', length)

        for i in range(10, 100, 3):
            chunk_row_size = 1 << i
            if chunk_row_size > length:
                break

            time0 = time.time()
            df.to_csv(to_csv_file, chunk_row_size = chunk_row_size)

            print('chunk_row_size', chunk_row_size, 'used time', time.time() - time0)

    
def to_csv_byte_array_benchmark():
    csv_source = 'resources/patients_export_geocodes_20210825040021.csv'
    with open(csv_source) as f:
        f.seek(0,2)
        total_byte_size = f.tell()

        f.seek(0)

    chunk_byte_size = 1 << 20

    print('total_byte_size', total_byte_size)

    i = 0
    total_time = 0
    with open('resources/to_csv_byte_array.csv', 'wb') as f:
        while i < total_byte_size:
            time0 = time.time()

            x  = np.fromfile(csv_source, count=chunk_byte_size, offset=i, dtype=np.uint8)
    
            f.write(x)

            total_time += time.time() - time0

            i += chunk_byte_size

    print('used time', total_time)



def distinct_benchmark():
    hdf5_df_distinct()
    pandas_drop_duplicate()


def hdf5_df_distinct():
    with Session() as s:
        ddf_1_name = f'destination_{schema_key}_1'
        ddf_2_name = f'destination_{schema_key}_2'
        ddf_3_name = f'destination_{schema_key}_3'
        print("=====hdf5 df distinct=====")
        print(source)

        ds = s.open_dataset(source, 'r+', 'ds')
        print('dataset keys -> dataframes: ', ds.keys())
        for i in range(1,4):
            if f'destination_{schema_key}_{i}' in ds.keys():
                ddf = ds[f'destination_{schema_key}_{i}']
                ds.delete_dataframe(ddf)

        if f'destination_{schema_key}' in ds.keys():
            ddf = ds[f'destination_{schema_key}']
            ds.delete_dataframe(ddf)

        
        
        print('dataset keys -> dataframes: ', ds.keys())

        df = ds[schema_key] 

        # print(df.keys())

        # # 1
        # print('@@@@  1  @@@@')
        # ddf_1 = ds.require_dataframe(ddf_1_name) 

        # time0 = time.time()
        # df.distinct(by = 'country_code', ddf = ddf_1)
        # print('distinct directly used time', time.time() - time0)

        # print('destination dataframe keys', ddf_1.keys())
        # result = ddf_1['country_code'].data[:]
        # print(result)

        # # 2
        # print('@@@@  2  @@@@')
        # ddf_2 = ds.require_dataframe(ddf_2_name) 

        # time0 = time.time()
        # df.groupby(by = 'country_code').distinct(ddf = ddf_2)
        # print('group by first then distinct, used time', time.time() - time0)

        # print('destination dataframe keys', ddf_2.keys())
        # result = ddf_2['country_code'].data[:]
        # print(result)

        # 3
        print('@@@@  3  @@@@')
        sorted_df = ds.require_dataframe('sorted_df') 
        df.sort_values(by = 'country_code', ddf = sorted_df)

        ddf_3 = ds.require_dataframe(ddf_3_name) 

        time0 = time.time()
        sorted_df.groupby(by = 'country_code').distinct(ddf = ddf_3)
        print('sort first, then group by, then distinct, used time', time.time() - time0)

        print('destination dataframe keys', ddf_3.keys())
        result = ddf_3['country_code'].data[:]
        print(result)




def pandas_drop_duplicate():
    print("==========pandas========")
    
    df = pandas.read_csv(pandas_read_csv_file, delimiter=',')
    time0 = time.time()
    result = df.drop_duplicates(subset=['country_code'])
    print('used time: ', time.time() - time0)
    print(result)




if __name__ == "__main__":
    # to_csv_writerow_benchmark() 
    # to_csv_byte_array_benchmark()   
    distinct_benchmark()
    # pandas_drop_duplicate()