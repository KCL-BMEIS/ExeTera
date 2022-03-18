# Intermediate Examples (0.5 onwards)

## Thinking in numpy

ExeTera provides a rich and growing set of operations that can be carried out directly on Fields rather than having to fetch the underlying data. There are still some times that you may need to fetch the underlying data directly. If you need to do so, it is important to understand that the underlying arrays returned by `data` are numpy arrays (excepting for indexed string fields which returns a list of strings).
The most important thing to understand when working with the underlying data is that it is in a numpy ndarray and all of the good practices of working with numpy arrays applies. In particular, performance is massively affected by dropping out of numpy for any reason (iterating loops explicitly, for example).

## Combining filters - not recommended
```
df # a dataframe containing the fields of interest

foo = df['foo']
bar = df['bar']


# terrible way - very slow as we are are reading each element from storage in turn

result = np.zeros(len(foo), dtype=bool)
for i in len(foo):
  result = foo.data[i] == 1 and bar.data[i] is False
df.create_numeric('result', 'bool').data.write(result)


# better but still slow as explicit iteration in numpy is discouraged

result = np.zeros(len(foo), dtype=bool)
foo_ = foo.data[:]
bar_ = bar.data[:]
for i in len(foo):
  result[i] = foo[r] == 1 and bar[r] is False
df.create_numeric('result', 'bool').data.write(result)
```

## Combining filters - recommended
```
# we can make use of fields directly rather than fetching the underlying numpy arrays
# we recommend this approach in general

df # a dataframe containing the fields of interest

foo = df['foo']
bar = df['bar']
df['result'] = (foo == 1) & (bar == False)

# or just

df['result'] = (df['foo'] == 1) & (df['bar'] == False)


# fetching numpy arrays

foo_ = df['foo'].data[:]
bar_ = df['bar'].data[:]
df.create_numeric('result', 'bool').data.write((foo_ == 1) & (bar_ == False))
```

There are still circumstances in which it may be better to fetch the underlying numpy arrays.
One such example is if you are checking the same value multiple times:

```
df # a dataframe containing the fields of interest

# one read of foo from storage

foo_ = df['foo'].data[:]
result = np.where(np.logical_or(foo_ == 4, foo_ == 3), True, False)
df.create_numeric('result', 'bool').data.write(result)


# two reads of foo from storage

df['result'] = (df['foo'] == 4) | (df['foo'] == 3)
```

## Filtering

Filtering is performed through the use of the `apply_filter` function. This can be performed on
individual fields or at a dataframe level. `apply_filter` applies the filter on data rows.

## Filter a dataframe
Note, this operation is destructive. It will overwrite the contents of the existing dataframe in storage.
```
df = # get a dataframe from somewhere

# apply a filter to the dataframe

filt = df['foo'] > 4
df.apply_filter(filt)

```

## Filter a dataframe, preserving the source dataframe

This operation creates a new dataframe and writes the filtered fields to it.

```
ds = # get a dataset from somewhere
df_foo = ds['foo']
df_bar = ds.create_dataframe('bar')
df_foo.apply_filter(df_foo['foobar'] > 4, df_bar)
```


## Sorting

Putting your data into the most appropriate order is very important for scaling of complex operations.
Certain operations in ExeTera require that the data is presented in sorted order in order to be able
to run correctly:
 * ordered_merges
 * generation and application of spans

Changing sorted order can be done in one of two ways:
 1. session.sort_on
 2. session.dataset_sort_index followed by DataFrame.apply_index or Field.apply_index

Either way, you must specify how you want the fields to be sorted. This is done through selecting the fields
to be sorted on. You can select one or more fields and the fields will be applied in order. For both methods,
you can specify one or more fields on which the data should be sorted (for example, 'user_id' and 'entry_date').

## Sorting with `sort_on`

Session.sort_on is provided for when you want to sort all of the fields in a dataframe. You can sort in-place
or you can sort and add the resulted sorted fields to a destination dataframe

### Sorting in place

Note: sorting in place is a destructive operation, as each dataframe is backed up by a dataset and this gets
changed when the sorted order changes. You may prefer to write the sorted data to a new dataframe instead
```
    # sort in place
    ds = # a dataset from somewhere
    session.sort_on(ds['foo'], ds['foo'], ('a_key_name',))
```

### Sorting to another dataframe

```
    # source to a destination group
    ds = # a dataset from somewhere
    ds.create_dataframe('bar')
    session.sort_on(ds['foo'], ds['bar'], ('a_key_name'))
```

## Sorting with `dataset_sort_index`

When sorting with `dataset_sort_index` we first get the permutation of the current indices to the sorted order.
We can then apply this to each of the fields that we want to reorder, as follows

```
    ds = # a dataset from somewhere

    index = session.dataset_sort_index((ds['foo']['a'],))

    # apply indices in place

    ds['foo'].apply_index(index)


    # apply indices to a destination dataframe

    ds.create_dataframe('bar')
    ds['foo'].apply_index(index, ds['bar'])
```

## Joining / merging

ExeTera provides functions that provide pandas-like merge functionality on `DataFrame` instances.
We have made this operation as familiar as possible to Pandas users, but there are a couple of
differences that we should highlight:
 * `merge` is provided as a function in the `dataframe` unit, rather than as a member function on `DataFrame` instances
 * `merge` takes three dataframe arguments, `left`, `right` and `dest`. This is due to the fact that DataFrames are always
   backed up by a datastore and so rather than create an in-memory destination dataframe, the resulting merged fields must
   be written to a dataframe of your choosing.
   * Note, this can either be a separate dataframe or it can be the dataframe that you are merging to (typically `left` in the case of a "left" merge and `right` in the case of a "right" merge
 * `merge` takes a number of optional hint fields that can save time when working with large datasets. These specify whether the keys are unique or ordered and allow the merge to occur without first checking this
   * `merge` has a number of highly scalable algorithms that can be used when the key data is sorted and / or unique.


```
ds = # a dataset fetched from somewhere
left = # a dataframe
right = # another dataframe
dest = ds.create_dataframe('merged')
merge(left, right, dest, left_on='a_key_in_left', right_on='a_key_in_right', how='left')
```

# Update the code from pre-0.5
ExeTera 0.5 is fully backward compatible, so you do not have to change your scripts. However, here are some new ways of doing things in 0.5 so that you can entirely focus on the analysis.

## 1, Creating fields using dataframe:

before:
```
with Session() as s:
    dst = s.open_dataset('/path/to/file','dst','r+')
    field = s.create_numeric('dst','a_numeric_field', 'int32')
```
Current version 0.5:
```
with Session() as s:
    dst = s.open_dataset('/path/to/file','dst','r+')
    df = dst.create_dataframe('df')
    field = df.create_numeric('a_numeric_field', 'int32')
```

## 2, Move / copy fields without careing about underlying data
before:
```
field_1 = ...
field_2 = session.create(dst,'foo','int32')
field_2.data.write(field_1.data[:])
```
Current version 0.5:
```
df2['foo'] = df['foo']
```

## 3, Grouped operations
before:
```
fielda = session.create_numeric('field_a', 'int32')
fieldb = session.create_numeric('field_b', 'int32')
fieldc = session.create_numeric('field_c', 'int32')
fielda_filtered = session.apply_filter(filter, fielda)
fieldb_filtered = session.apply_filter(filter, fieldb)
fieldc_filtered = session.apply_filter(filter, fieldc)
```

Current version 0.5:
```
df.create_numeric('field_a', 'int32')
df.create_numeric('field_b', 'int32')
df.create_numeric('field_c', 'int32')
df.apply_filter(filter)
```

## 4, Operations on field directly
before:
```
result = df['a'].data[:] + df['b'].data[:]
field_c = session.create_numeric(dst,'c', 'int32')
field_c.data.write(result)
```

Current version 0.5:
```
df['c'] = df['a'] + df['b']
```
