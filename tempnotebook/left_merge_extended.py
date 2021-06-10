import numpy as np
import pandas as pd

from exetera.core.utils import Timer
from exetera.core.session import Session
from exetera.core.dataframe import DataFrame, merge


def get_from_pandas(series, invalid_value, destination_dtype):
    return series.fillna(invalid_value).to_numpy(destination_dtype)

l_length = 2000000
r_length = 200000000
rng = np.random.RandomState(12345678)

l_key_ = np.arange(1, l_length + 1) * 10
l_a_ = rng.randint(1, l_length, l_length)

r_key_ = np.arange(1, r_length + 1) * 10
lf_key_ = rng.randint(0, l_length, r_length)
lf_key_ = l_key_[lf_key_]
r_a_ = rng.randint(1, r_length, r_length)

# give each patient an "activity level"
p = rng.randint(0, 9, l_length)
p = p / 10
p = p / np.sum(p)

# generate a set of patient activities based on activity level
lf_key_ = rng.choice(l_key_, p=p, size=r_length)
# print(lf_key_[:200])

sorted_order = np.argsort(lf_key_)
r_key_ = r_key_[sorted_order]
lf_key_ = lf_key_[sorted_order]
r_a_ = r_a_[sorted_order]
# print(lf_key_[:100])

for how in ('left', 'right', 'inner'):
    with Session() as s:
        print("how = {}".format(how))
        # merge on pandas dataframe

        l_df = pd.DataFrame({'l_key': l_key_, 'l_a': l_a_})
        r_df = pd.DataFrame({'r_key': r_key_, 'lf_key': lf_key_, 'r_a': r_a_})
        with Timer("pandas {} merge:".format(how)):
            m_df = pd.merge(l_df, r_df, left_on='l_key', right_on='lf_key', how=how)

        # merge in exetera dataframe

        ds = s.open_dataset('/home/ben/covid/benchmarking.hdf5', 'w', 'ds')
        l_df2 = ds.create_dataframe('l_df')
        l_df2.create_numeric('l_key', 'int32').data.write(l_key_)
        l_df2.create_numeric('l_a', 'int32').data.write(l_a_)

        r_df2 = ds.create_dataframe('r_df')
        r_df2.create_numeric('r_key', 'int32').data.write(r_key_)
        r_df2.create_numeric('lf_key', 'int32').data.write(lf_key_)
        r_df2.create_numeric('r_a', 'int32').data.write(r_a_)

        m_df2 = ds.create_dataframe('m_df_2')

        # for f in [l_df2['l_key'], l_df2['l_a'], r_df2['r_key'], r_df2['lf_key'], r_df2['r_a']]:
        #     print(f.name, f.data[:20])

        with Timer("exetera unordered {} merge:".format(how)):
            merge(l_df2, r_df2, m_df2, left_on='l_key', right_on='lf_key',
                  left_fields=['l_a'], right_fields=['r_a'], how=how)

        # print(m_df['r_a'].to_numpy(dtype='int32')[:100])
        print("pd/m 'l_a':", np.array_equal(m_df['l_a'].to_numpy(), m_df2['l_a'].data[:]))
        print("pd/m 'r_a':", np.array_equal(get_from_pandas(m_df['r_a'], 0, 'int32'),
                                            m_df2['r_a'].data[:]))

        m_df3 = ds.create_dataframe('m_df_3')

        with Timer("exetera ordered {} merge:".format(how)):
            merge(l_df2, r_df2, m_df3, left_on='l_key', right_on='lf_key',
                  left_fields=['l_a'], right_fields=['r_a'], how=how,
                  hint_left_keys_ordered=True, hint_right_keys_ordered=True)

        print("pd/om 'l_a':", np.array_equal(m_df['l_a'].to_numpy(), m_df3['l_a'].data[:]))
        print("pd/om 'r_a':", np.array_equal(get_from_pandas(m_df['r_a'], 0, 'int32'),
                                             m_df3['r_a'].data[:]))
