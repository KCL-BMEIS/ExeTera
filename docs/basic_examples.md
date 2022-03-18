# Basic Examples

As of ExeTera version 0.5.0, the API and its usage has changed dramatically, with usability and familiarity to users of Pandas being our primary goal. These updated examples are focussed on getting you started with simple operations such as creating, opening and interacting with `DataSets`, `DataFrames` and `Fields`.

## Sessions

### Creating a session object
Creating a `Session` object can be done multiple ways, but we recommend that you wrap the session in a [context manager (`with` statement)](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement). This allows the Session object to automatically manage the datasets that you have opened, closing them all once the `with` statement is exited.
Opening and closing datasets is very fast. When working in jupyter notebooks or jupyter lab, please feel free to create a new `Session` object for each cell.

```
from exetera.core.session import Session

# recommended
with Session() as s:
  ...

# not recommended
s = Session()
```


### Loading dataset(s)

Once you have a session, the next step is typically to open a dataset.
Datasets can be opened in one of three modes:
 * read - the dataset can be read from but not written to
 * append - the dataset can be read from and written to
 * write - a new dataset is created (and will overwrite an existing dataset with the same name)

```
with Session() as s:
  ds1 = s.open_dataset('/path/to/my/first/dataset/a_dataset.hdf5', 'r', 'ds1')
  ds2 = s.open_dataset('/path/to/my/second/dataset/another_dataset.hdf5', 'r+', 'ds2')
```

### Closing a dataset

Closing a dataset is done through Session.close_dataset, as follows

```
with Session() as s:
  ds1 = s.open_dataset('/path/to/dataset.hdf5', 'r', 'ds1')

  # do some work
  ...

  s.close_dataset('ds1')
```

## DataSet

Datasets are the object that maps to a given ExeTera datastore. When you open a dataset, it is a DataSet object that you get back. This can in turn be used to create, modify and delete DataFrames in the datastore.
```
with Session() as s:
  ds = s.open_dataset('/path/to/dataset.hdf5', 'r', 'ds')

  # list the tables present in this dataset
  for k in ds.keys():
    print(k)

```

## DataFrame

DataFrames are designed with Pandas users in mind and have been created to explicitly be as close to Pandas as possible, given the differences between Pandas and ExeTera and how they represent data under the hood.
DataFrames have a rich API that allows you to add, access and remove fields, as well as operations that can be carried out across all of the fields in the data frame.

### Basic DataFrame manipulation

```
ds = # get a dataset from somewhere
df = ds['a_dataframe']

df2 = s.create_dataframe('another_dataframe') # create an empty dataframe

ds['a_copy'] = df # copy a dataframe
```

### Adding and removing fields
```
ds = # get a dataset from somewhere
df = ds.create_dataframe('a_dataframe')

# create a set of (empty) fields
i_f = df.create_indexed_string('an_indexed_string_field')
f_f = df.create_fixed_string('a_fixed_string_field', 10)
n_f = df.create_numeric('a_numeric_field', 'int32')
c_f = df.create_categorical('a_categorical_field', 'int8', {0: b'a', 1: b'b'})
t_f = df.create_timestamp('a_timestamp_field')

# move / copy fields for assignment
df2 = ds['another_dataframe']
df3 = ds['yet_another_dataframe']
df2['b'] = df2['a'] # rename a field by assigning it
print('a' in df2) # -> False
df2['c'] = df3['c'] # copy a field between datasets
```

## Fields

### Get a field
The field being loaded must represent a valid field (see [Concepts](Concepts))

Getting fields has now been simplified. You can still call session.get, but it is simpler to fetch it directly from a DataFrame.

```
df = # get a dataframe from somewhere
f = df['a_field']
```

### Getting the length of a field
This can be done one of two ways. You can either ask the field for its length directly or get it from the fields 'data' property:
```
f = # get a field from somewhere
print(len(f))
print(len(f.data))
```

### Load all of the data for a field
When you need to access the underlying data directly, you can do so through the `data` property:
```
f = # get a field from somewhere
values = f.data[:]
```
Note that indexed string fields have two properties that allow you to access the underlying indices and values. Indexed string fields can still have their data accessed through the `data` property, but this is a very slow and expensive operation when perform on a large field.
```
f = # get a field from somewhere
indices, values = f.indices[:], f.values[:]
```
Note that indices and values are not the same length as the length reported through `len(f)` or `len(f.data)`.

### Performing operations on fields
New to ExeTera 0.5 is the ability to perform many operations directly on fields that previously required you to fetch the underlying data.
```
df['c'] = df['a'] + df['b']

z = df['x'] / df['y']
df['z'] = z * 2
```

### Create a field
```
patients = dataset['patients']
timestamp = datetime.now(timezone.utc)
isf = session.create_indexed_string(patients, 'foo')
fsf = session.create_fixed_string(patients, 'bar', 10)
csf = session.create_categorical(patients, 'boo', {'no':0, 'maybe':2, 'yes':1})
nsf = session.create_numeric(patients, 'far', 'uint32')
tsf = session.create_timestamp(patients, 'foobar')
```

### Write to a field in chunks
```
for c in chunks_from_somewhere:
    field.data.write_part(c)
field.flush()
```

### Write to a field in a single go
```
field.data.write(generate_data_from_somewhere())
```

### Referring to fields
Most of the session functions accept various representations of fields. The ones that are a bit more restrictive will be made more flexible in future releases. The following calls to apply_index are equivalent.

```
index = index_from_somewhere()
raw_foo = a_numpy_array_from_somewhere()
result = session.apply_index(index, src['foo']) # refer to the hdf5 Group that represents the field
result = session.apply_index(index, session.get(src['foo']) # refer to the field
result = session.apply_index(index, session.get(src['foo'].data[:]) # refer to the data in the field
result = session.apply_index(index, raw_foo) # refer to a numpy array
```


