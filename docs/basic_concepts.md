This page covers the basic concepts behind the pipeline and the design decisions that have gone into it.

# Basic concepts

## Keys and Values

ExeTera is a piece of software that works with key / value pairs.

Key / value pairs are a common idea in datastores; values are typically anything from a scalar value to many millions of values in an array. Keys are a way to identify one set of values from another. The reason data is organised as key / value pairs is that a dataset composed of many such values is that it is the quickest way to access all or some of the values associated with a small set of keys. This is in contrast with CSV files (and SQL-based relational databases) that organise data so that each element of a collection is stored next to another in memory.

CSV files are what is known as row-oriented datasets. Each value for a given row is contiguous in memory. This creates many problems when processing large datasets because a typical analysis may only want to make use of a small number of the available fields. The ExeTera datastore stores data in a column-oriented fashion, by contrast, which means that all the data for a given field is together in memory. This is typically far more efficient for the vast majority of processing tasks.

```
CSV / SQL layout
----------------
 a     b     c     d
 1 ->  2 ->  3 ->  4
 5 ->  6 ->  7 ->  8
 9 -> 10 -> 11 -> 12
13 -> 14 -> 15 -> 16
17 -> 18 -> 19 -> 20
21 -> 22 -> 23 -> 24
```

```
Key-Value Store (e.g. ExeTera layout)
-------------------------------------
 a     b     c     d
 1 |   7    13    19
 2 |   8    14    20
 3 |   9    15    21
 4 |  10    16    22
 5 |  11    17    23
 6 v  12    18    24
```


## Metadata vs. data

ExeTera distinguishes between metadata and data. Metadata is the information about a field except for the actual data values themselves. It is typically much smaller than the actual data and thus is loaded up front when a dataset is opened.

### Examples of Metadata

Consider a field for `Vaccination Status`. `Vaccination Status` has three distinct values that it can take:
 * 'not_vaccinated'
 * 'partially_vaccinated'
 * 'fully_vaccinated'

Such a field may contain data on millions of individuals' vaccinations. As such, the data is stored as a [categorical field](https://github.com/KCL-BMEIS/ExeTera/wiki/Data-Schema#categorical-field). Strings are very expensive values to store when there are many millions of them, and so we prefer to store the data as a more efficient type (typically 8 bit integers). The array of millions of 8 bit integers is the ***data***. We maintain a mapping between the strings and the numbers for the user's convenience

```
0: not_vaccinated
1: partially_vaccinated
2: fully_vaccinated
```

Data on the other hand, is only loaded when requested by the user or for processing. This, along with the details of the field's type and the timestamp for when it was written, are the ***metadata***.

The metadata is typically loaded for the whole dataset up front, whereas the data is only loaded on demand. This is because the metadata even for many thousands of fields is typically trivially small, whereas the data for many thousands of fields may be many times larger than a typical computers random access memory (RAM).

## Strongly typed fields

ExeTera encourages the use of strongly-typed fields for fields that represent a specific underlying type, such as floating point numbers, integers or categorical variables. These are typically far more efficient to process than string fields and should be used whenever possible.

The following datatypes are provided:

 * Numeric fields
 * Categorical fields
 * DateTime / Date fields
 * Fixed string fields
 * Indexed string fields

### Numeric fields

Numeric fields can hold any of the following values:

 * bool
 * int8, int16, int32, int64
 * uint8, uint16, uint32, uint64

Please note, uint64 usage is discouraged. ExeTera makes heavy use of `numpy` under the hood and `numpy` has some odd conventions when it comes to `uint64` processing. In particular:
```
a = uint64(1)
b = a + 1
print(type(b))
# float64
```

## HDF5
HDF5 is a hierarchic key/value store. This means that it stores pairs of keys and data associated with that key. This is important because a dataset can be very large and the data that you want to perform analysis on can be a very small fraction of that dataset. HDF5 allows you to explore the hierarchic collection of fields without having to load them, and it allows you to load specific fields or even part of a field.

## Fields
Although we can load part of a field, which allows us to perform some types of processing on arbitrarily large fields, the native performance of HDF5 field iteration is very poor, and so much of the functionality of the pipeline is dedicated towards providing scalability without sacrificing performance.

Fields have another purpose, which is to support useful metadata along with the field data itself, and also to hide the complexity behind storing certain datatypes efficiently

## Datatypes

The pipeline has the following datatypes that can be interacted with through Fields
### Indexed string

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

#### UTF8
UTF8 strings are encoded into byte arrays before being stored. They are decoded back to UTF8 when reconstituted back into strings when read.

## Fixed string
Fixed string fields store each entry as a fixed length byte array. Entries cannot be longer than the number of bytes specified.
<TODO:> encoding / decoding and UTF8

## Numeric
Numeric fields are just that, arrays of a given numeric value. Any primitive numeric value is supported, although use of `uint64` is discouraged, as this library is heavily reliant on `numpy` and `numpy` does unexpected things with `uint64` values
```
a = uint64(1)
b = a + 1
print(type(b))
# float64
```

## Categorical
Categorical fields are fields where only a certain set of values is permitted. The values are stored as an array of `uint8` values, and mapped to human readable values through the 'key' field.

## Timestamp
Timestamp fields are arrays of float64 posix timestamp values. These can be mapped to and from datetime fields when performing complex operations. The decision to store dates and datetimes this way is primarily one of performance. It is very quick to check whether millions of timestamps are before or after a given point in time by converting that point in time to a posix timestamp and peforming a fast floating point comparison.

## Operations

## Reading from Fields
Fields don't read any of the field data from storage until the user explicitly requests it. The user does this by performing array dereference on a field's `data` property:
```
r = session.get(dataset['foo'])
rvalues = r.data[:]
```
This reads the whole of a given field from the dataset.

## Writing to fields
Fields are written to in one of three ways:

 * one or more calls to `write_part`, followed by `flush`
 * a single call to `write`
 * writing to the data member, if overwriting existing contents but maintaining the field length

```
w = session.create_numeric(dataset, 'foo', 'int32')
for p in parts_from_somewhere:
    w.write_part(p)
w.flush()
```
When using `write`
```
w = session.create_numeric(dataset, 'foo', 'int32')
w.write(data_from_somewhere)
```
Fields are marked completed upon `flush` or `write`. This is the last action that is taken when writing, and indicates that the operation was successfully completed.
