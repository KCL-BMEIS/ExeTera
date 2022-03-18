# Sessions

`Session` instances are the top-level ExeTera class. They serve two main purposes:
 1. Functionality for creating / opening / closing [`Dataset`](https://github.com/KCL-BMEIS/ExeTera/wiki/Dataset-API) objects, as well as managing the lifetime of open datasets
 2. Methods that operate on Fields


## Opening and closing datasets

### Creating a session object
Creating a `Session` object can be done multiple ways, but we recommend that you wrap the session in a [context manager (`with` statement)](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement). This allows the Session object to automatically manage the datasets that you have opened, closing them all once the `with` statement is exited.
Opening and closing datasets is very fast. When working in jupyter notebooks or jupyter lab, please feel free to create a new `Session` object for each cell.

```
from hystore.core.session import Session

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
## Field Operations

`Session` has a number of operations that can be carried out on [`Field`](https://github.com/KCL-BMEIS/ExeTera/wiki/Field-API) objects. These operations fall into two main categories:
1. Operations that don't clearly 'belong' to a given Field, such as merging
2. Operations that are now supported by `Field` and [`DataFrame`](https://github.com/KCL-BMEIS/ExeTera/wiki/DataFrame-API) but were not in legacy versions, and are maintained for backward compatibility
