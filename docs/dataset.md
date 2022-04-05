# Datasets

ExeTera works with HDF5 datasets under the hood, and the `Dataset` class is the means why which you interact with it at the top level. Each `Dataset` instance corresponds to a physical dataset that has been created or opened through a call to `session.open_dataset`.

Datasets are in turn used to create, access and delete [`DataFrame`](https://github.com/KCL-BMEIS/ExeTera/wiki/DataFrame-API)s. Each `DataFrame` is a top-level HDF5 group that is intended to be very much like and familiar to the [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

## Dataset usage examples

### Create a new dataframe

```
ds = # get a dataset from somewhere
df = ds.create_dataframe('foo')
```

### Delete an existing dataframe

```
ds = # get a dataset from somewhere
ds.delete_dataframe('foo')
```

### Rename a dataframe

```
ds = # get a dataset from somewhere
ds['bar'] = ds['foo'] # internally performs a rename
dataset.move(ds['bar'], ds, 'foo')
```

### Copy a dataframe within a dataset

```
dataset.copy(ds['foo'], ds, 'bar')
```

### Copy a dataframe between datasets

```
ds1 = # get a dataset from somewhere
ds2 = # get another dataset from somewhere

ds2['foo'] = ds1['foo']
ds2['foobar'] = ds1['bar']
```

