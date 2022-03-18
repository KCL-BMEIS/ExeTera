# Use ExeTera via R
The analytics API is written in python, but you can also use python module via R through [reticulate](https://rstudio.github.io/reticulate/). We have created a wrapper using reticulate for people to easily call ExeTera from R interface. Please refer to [ExeTeraR](https://github.com/KCL-BMEIS/ExeTeraR) repo for more detail and examples.

## Known limitations

It is possible to access ExeTera DataFrames and Fields through R but as it stands, it is not possible to write to Fields or change the contents of DataFrames. This is due to Reticulate requiring that all interaction with Reticulate-wrapped python objects being manipulated only on the thread on which it runs. We are looking at techniques to work around this.

## Accessing ExeTera through ExeTera-R-Wrapper
### Environment Setup
Before start, please make sure you have 'reticulate' and 'devtools' installed in your R environment.

Under the development stage, the ExeTera-R-Wrapper is loaded using devtools:

`library(devtools) # load devtools`

`load_all('/home/abc/codes/exetera') # path to the source code of ExeTera-R-Wrapper`

Once loaded, you can call ExeTera objects from R (note the '$' sign instead of '.'), for example

`exetera$core$dataframe$copy()`

In the future, we may consider build and install so that you can install ExeTera-R-Wrapper as a independent library.

### Session
Once you loaded the ExeTera-R-Wrapper, you should be able to create a ExeTera session instance.

`session = Session()`

Note the proper way of opening a session is through the python 'with' statement. Without a R-equivalent, please do remember to close the session manually in the end of your code, to make sure the HDFS file is properly close.

`session$close()`

### ExeTera DataSet
Once the session instance is created, you can use session to open or create a dataset object. If open an existing hdf5 file, the data inside the file is automatically loaded into dataframes.

```
source = session$open_dataset('abc.h5', 'r+', 'source') # 'r' to open an existing hdf5 file
output = session$open_dataset('playground.hdf5', 'w', 'output') # 'w' to create a new hdf5 file
ds.keys(output)  # show existing dataframes
```

### ExeTera DataFrame
The dataframe is the store unit of multiple fields, equivalent to a Pandas DataFrame or CSV file.

You can check the fields available in a dataframe: `df.keys(df)`

Get an existing field: `df['num']`

Create a new field: `df$create_numeric('num','uint32')`

Apply filter to all the fields: `df$apply_filter(filter)`

Re-index all the fields: `df$apply_index(index)`

### Fields
The field is one column of data.

Once you have a field, you can get the whole data: `fld.data(df['num'])`

Or a subset: `fld.data(df['num'], 1:2)`

Write data to field: `df['num']$data$write(c(1,2,3))`

Apply filter: `df['num']$apply_filter(filter)`

Re-index: `df['num']$apply_index()`

### Other Utilities
```
util.uniq_c(field), equivalent to the numpy's unique(return_counts=True)
df.write_csv(dataframe, 'csv_file_name.csv'), write a dataframe to csv file
```

### Putting Together
This example shows how you can access data from a given field (WORK IN PROGRESS!).
```
library(devtools) load_all('/home/jd21/codes/exetera') # path of the R wrapper
exetera('/home/jd21/miniconda3/bin/python') #init the R wrapper by point the python location
s = Session() # the exetera journey starts
src = s$open_dataset('abc.h5', 'r+', 'src') # exetera api while changing . to $
ds = src$create_dataframe('df')
num = s$create_numeric(df, 'num', 'uint32')
num$data$write(c(1,2,3,4,5))
fld.get(num)
fld.get(num,1:3)
```
