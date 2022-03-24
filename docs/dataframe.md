# DataFrames

The ExeTera `DataFrame` object is intended to be familiar to users of Pandas, albeit not identical.

## Differences
ExeTera works with [`Datasets`](https://github.com/KCL-BMEIS/ExeTera/wiki/Dataset-API), which are backed up by physical key-value HDF5 datastores on drives, and, as such, there are necessarily some differences between the Pandas `DataFrame`:
 * Pandas DataFrames enforce that all Series ([`Fields`](https://github.com/KCL-BMEIS/ExeTera/wiki/Field-API) in ExeTera terms) are the same length. ExeTera doesn't require this, but there are then operations that do not make sense unless all fields are of the same length. ExeTera allows DataFrames to have fields of different lengths because the operation to apply filters and so for to a DataFrame would run out of memory on large DataFrames
 * Types always matter in ExeTera. When creating new Fields (Pandas Series) you need to specify the type of the field that you would like to create. Fortunately, Fields have convenience methods to construct empty copies of themselves for when you need to create a field of a compatible type

ExeTera DataFrames are new with the 0.5 release of ExeTera and do not yet support all of the operations that Panda DataFrames support. This functionality will be augmented in future releases.

## DataFrame usage examples

### Create a new field

```
df = # get a DataFrame from somewhere
i_f = df.create_indexed_string('i_foo')
f_f = df.create_fixed_string('f_foo', 8)
n_f = df.create_numeric('n_foo', 'int32')
c_f = df.create_categorical('c_foo', 'int8', {b'a': 0, b'b': 1})
t_f = df.create_timestamp('t_foo')
```

### Copy a field from another dataframe

```
df1 = # get a DataFrame from somewhere
df2 = # get another DataFrame from somewhere
df2['foo'] = df1['foo']
df2['foobar'] = df2['bar']
```

### Apply a filter to all fields in a dataframe

```
df1 = # get a DataFrame from somewhere
filt = # get a filter from somewhere
df2 = df1.apply_filter(filt) # creates a new dataframe from the filtered dataframe
df1.apply_filter(filt, in_place=True) # destructively filters the dataframe
```

### Re-index all fields in a dataframe

```
df1 = # get a DataFrame from somewhere
inds = # get a set of indices from somewhere
df2 = df1.apply_index(inds) # creates a new dataframe from the re-indexed dataframe
df1.apply_index(inds, in_place=True) # destructively re-indexes the dataframe
```


