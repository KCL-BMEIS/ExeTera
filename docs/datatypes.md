In order to maximise performance, both in terms of processing speed and memory footprint, ExeTera goes to a great deal of effort to use data representations that are as efficient as possible. All of the data in a CSV file is represented as strings, and must be converted to the appropriate datatypes in order to be processed quickly once in an ExeTera datastore.

# ExeTera Datatypes

ExeTera makes the following datatypes available for use:
* Variable length (indexed) strings as `IndexedStringField` objects
* Fixed-length strings as `FixedStringField` objects
* Numerical values as `NumericField` objects
* Categorical values as `CategoricalField` objects
* DateTime / Date values `TimestampField` objects

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
