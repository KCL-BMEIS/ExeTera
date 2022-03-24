# Fields

The `Field` object is the analogy of the Pandas DataFrame `Series` or Numpy `ndarray` in ExeTera.
Fields contain (often very large) arrays of a given data type, with an API that allows intuitive manipulations of the data.

## Fields correspond to one (or more) arrays of data

In order to store very large data arrays as efficiently as possible, Fields store their data in ways that may not be intuitive to people familiar with Pandas or Numpy. Numpy makes certain design decisions that reduce the flexibility of lists in order to gain speed and memory efficiency, and ExeTera does the same to further improve on speed and memory. The [`IndexedStringField`](https://github.com/KCL-BMEIS/ExeTera/wiki/Datatypes#indexedstringfield), for example, uses two arrays, one containing a concatinated array of bytevalues from all of the strings in the field, and another array of indices indicating where each field starts and end. This is much faster and more memory efficient to iterate over than a Numpy string array when the variability of string lengths is very high. This kind of change however, creates a great deal of complexity when exposed to the user, and `Field` does its best to hide that away and act like a single array of string values.

## Field operations
Operations on fields can be divided into the following groups:

 - accessing underlying data
 - constructing compatible empty fields
 - arithmetic operations
 - logical operations
 - comparison operations
 - application of filters, indices and spans

### Accessing underlying data

Underlying data can be accessed as follows:

 - All fields have a `data` property that provides access to the underlying data that they contain. For most field types, it is very efficient to read from and write to this property, provided it is done using slice syntax
 - Indexed string fields provide `data` as a convenience method, but this should only be used when performance is not a consideration
 - Indexed string fields provide `indices` and `values` properties should you need to interact with their underlying data efficiently and directly. For the most part, we discourage this and have tried to provide you with all of the methods that you need under the hood

### Constructing compatible empty fields

Fields have a `create_like` method that can be used to construct an empty field of a compatible type
 - when called with no arguments, this creates an in-memory field that can be further manipulated before eventually being assigned to a DataFrame (or not)
 - when called with a DataFrame and a name, it will create an empty field on that DataFrame of the given name that can subsequently be written to
See below for examples

### Arithmetic operations

Numeric and timestamp fields have the standard set of arithmetic operations that can be applied to them:
 - These are `+`, `-`, `*`, `/`, `//`, `%`, and `divmod`

### Element-wise logical operators

Numeric fields can have logical operations performed on them on an element-wise basis
 - These are `&`, `|`, `^`

### Comparison operators

Numeric, categorical and timestamp fields have comparison operations that can be applied to them:
 - These are `<`, `<=`, `==`, `|=`, `>=`, `>`


## Field usage examples

## Create a field from another field

```
f = # get a field from somewhere
g = f.create_like() # creates an empty field
```

## Create a compatible field in a dataframe
```
f = # get a field from somewhere
df = # get a dataframe from somewhere
f.create_like(df, 'foo')
new_field = df['foo'] # get the new, empty field
```

## Field arithmetic

The following snippet involves the creation of 'memory-based' fields. These are intermediate fields that are the result of performing operations on
fields that are part of a DataFrame.

```
df = # get a dataframe from somewhere
df['c'] = df['a'] + df['b']
```
If we were to modify this snippet so that we wrote the result of the addition operation to a separate variable, then assigned the variable to the dataframe, it would essentially be doing the same thing under the hood

```
df = # get a dataframe from somewhere
c = df['a'] + df['b']
df['c'] = c
```

## Applying filters to a field

```
inds = # get indices from somewhere
f = # get a field from somewhere
df # get a dataframe from somewhere

g = f.apply_filter(filt) # in-memory field 'g' produced by performing the filter: 'f' is unchanged

# create a field on dataframe 'df' and then assign to it - the slightly awkward way
h = f.create_like(df, 'h')
h = f.apply_filter(filt, h)

# the one-line way
df['h'] = f.apply_filter(filt)

# destructive, in-field filter
f.apply_filter(filt, in_place=True)
```

## Applying indices to a field

```
inds = # get a filter from somewhere
f = # get a field from somewhere
df # get a dataframe from somewhere

g = f.apply_indices(inds) # in-memory field 'g' produced by performing the index: 'f' is unchanged

# create a field on dataframe 'df' and then assign to it - the slightly awkward way
h = f.create_like(df, 'h')
h = f.apply_inds(inds, h)

# the one-line way
df['h'] = f.apply_indices(inds)

# destructive, in-field reindexing
f.apply_indices(inds, in_place=True)
```

## Applying spans to a field

```
session = # the Session object
s = # field from which we will obtain spans (typically a primary key)
f = # get a field from somewhere
df # get a dataframe from somewhere

spans = session.get_spans(s)
g = f.apply_spans_max(spans) # in-memory field 'g' produced by performing the span appliation: 'f' is unchanged

# create a field on dataframe 'df' and then assign to it - the slightly awkward way
h = f.create_like(df, 'h')
h = f.apply_spans_max(spans, h)

# the one-line way
df['h'] = f.apply_spans_max(spans)

# destructive, in-field span application
f.apply_spans_max(spans, in_place=True)
```



In order to maximise performance, both in terms of processing speed and memory footprint, ExeTera goes to a great deal of effort to use data representations that are as efficient as possible. All of the data in a CSV file is represented as strings, and must be converted to the appropriate datatypes in order to be processed quickly once in an ExeTera datastore.

# ExeTera Datatypes

ExeTera makes the following datatypes available for use:
- Variable length (indexed) strings as `IndexedStringField` objects
- Fixed-length strings as `FixedStringField` objects
- Numerical values as `NumericField` objects
- Categorical values as `CategoricalField` objects
- DateTime / Date values `TimestampField` objects

## `IndexedStringField`

Indexed strings exist to provide a compact format for storing variable length strings in HDF5. Python / HDF5 through `h5py` doesn't support efficient string storage and so we convert python strings to indexed strings before storing them, resulting in orders of magnitude smaller representation in some cases.
Indexed strings are composed to two elements, a `uint8` 'value' array containing the byte data of all the strings concatenated together, and an index array indicating where a given entry starts and ends in the 'value' array.

Example:
Take the following string list
```
['The','quick','brown','fox','jumps','over','the','','lazy','','dog']
```
This is serialised as follows:
```
values = [Thequickbrownfoxjumpsoverthelazydog]
index = [0,3,8,13,16,21,25,28,28,32,32,35]
```
Note that empty strings are stored very efficiently, as they don't require any space in the 'values' array.

### UTF8
UTF8 strings are encoded into byte arrays before being stored. They are decoded back to UTF8 when reconstituted back into strings when read. This is very expensive in both time and memory when applied to many millions of entries and so should be avoided whenever possible

## `FixedStringField`

Fixed string fields store each entry as a fixed length byte array. Entries cannot be longer than the number of bytes specified.
As with indexed string fields, data is stored as byte arrays in fixed length string fields. This means that you need to calculated the fixed size based on the maximum expected bytearray length, rather than string length.

```
över           -> b'\xc3\xb6ver'
# 4 characters -> 5 bytes
```

### Converting strings to byte arrays

You will typically write data to FixedStringField via a python list or numpy array:

#### List encode

```
strings = ['a', 'bb', 'ccc', 'dddd', 'eeeee']
field.data.write([s.encode() for s in strings])
```

#### Numpy encode

```
strings = np.asarray(['a', 'bb', 'ccc', 'dddd', 'eeeee'])
field.data.write(np.char.encode(strings))
```

### Converting byte arrays from FixedStringField back to strings

Again, this is an operation that you should only do when necessary, as for large arrays it is very expensive.

```
strings = np.char.decode(field.data[:])
```

## `NumericalField`

Numeric fields are just that, arrays of a given numeric value. Any primitive numeric value is supported, although use of `uint64` is discouraged, as this library is heavily reliant on `numpy` and `numpy` does unexpected things with `uint64` values
```
a = uint64(1)
b = a + 1
print(type(b))
# float64
```

## `CategoricalField`

Categorical fields are fields where only a certain set of values is permitted. The values are stored as an array of `uint8` values, and mapped to human readable values through the 'key' field.

## `TimestampField`

Timestamp fields are arrays of float64 posix timestamp values. These can be mapped to and from datetime fields when performing complex operations. The decision to store dates and datetimes this way is primarily one of performance. It is very quick to check whether millions of timestamps are before or after a given point in time by converting that point in time to a posix timestamp and peforming a fast floating point comparison.
