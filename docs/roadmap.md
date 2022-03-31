# The ExeTera roadmap

ExeTera is under active development. Our primary goals for the software are as follows:
1. Provide a full, rich API that is familiar to users of Pandas
1. Provide an engine that can execute in environments from laptops to clusters
1. Provide a powerful set of data curation features, such as journalling and versioning of algorithms for reproducibility

These correspond to the three pillars of functionality that we believe can create an innovative, world beating software package combining data curation and analytics.

# Upcoming releases

Please note, these releases do not currently have a regular release schedule, but can be expected roughly every four to eight weeks.
Their contents may change to respond to the needs of our userbase. This page is updated quite regularly.

### v0.6
* [R wrapper for Exetera](#R-wrapper-for-ExeTera)
* [Import API](#Import-API)
* [DataFrame API: Pandas first tier](#Dataframe-api-pandas-first-tier)

### v0.7
* [DataFrame API: Pandas second tier](#Dataframe-api-pandas-second-tier)
* [DataFrame views](#Dataframe-views)

### v0.8
* [DataFrame API: Pandas third tier](#Dataframe-api-pandas-third-tier)
* [Filesystem-based serialized datastore](#Filesystem-based-serialized-datastore)

### v0.9
* [DataFrame API: Pandas fourth tier](#Dataframe-api-pandas-fourth-tier)

### v0.10
* [Graph-based compilation and execution](#Graph-based-compilation-and-execution)

## Roadmap items

The rest of the Roadmap page is divided into subsections, each dealing with an aspect of the software:
1. [R wrapper for ExeTera](#R-wrapper-for-ExeTera)
1. [API](#API)
1. [Technical refactors](#Technical-refactors)
1. [Serialization](#Serialization)
1. [Performance and scale](#Performance-and-scale)
1. [Data curation](#Data-curation)

## R wrapper for ExeTera

ExeTera is currently a Python-only library. R users can access ExeTera through Reticulate, an R library written to interop with Python packages, but this is onerous for two of reasons:
1. Syntactic sugar in Python such as slices does not work in R, and so data fetching involves the creation of explicit slice objects
2. ExeTera (like most Python libraries) uses 0-based indices, whereas R uses 1-based indices, so the user must perform this conversion correctly every time

We propose to write an R library that wraps Sessions, Datasets, DataFrames and Fields with their R equivalents. This should enable R users to write using syntax and conventions that they are used to while using ExeTera.


## API

### Import API

Importing is mainly done through the use of schema files and the `exetera import` command, but it should also be readily accessible through the API. We have importer objects that are used internally when the import is performed, but with a little bit of work, this can be made part of the API, along with an accompanying `read_csv` function in the DataFrame module.

### DataFrame API

Pandas' DataFrame API (itself based on the R data.frame) has a rich set of functionality that is not yet supported by ExeTera. This missing functionality ranges from 'implement as soon as humanly possible' to 'implement when we have time to do so'. We should be guided by our users in this matter, but we have an initial set of priorities that should be refined upon.

As ExeTera has some critical differences to Pandas and so there are areas in which we must necessarily deviate from the API. The largest difference is that ExeTera doesn't keep the DataFrame field values in memory but reads them from drive when they are required. This changes `DataFrame.merge` for example, as it is necessary to specify a DataFrame instance to which merged values should be written.

We should tackle this work in a series of stages so that for each new release we have broadened the API.

The work is divided into four tiers:
1. Tier 1 is critical API that we have identified as core-functionality that ExeTera DataFrames need as soon as possible
1. Tiers 2 and 3 are necessary functionality that we want to add to ExeTera DataFrames but ordered by priority so that they can be released regularly and often
1. Tier 4 and onwards is functionality that aren't considered necessary at present but will be added to the roadmap if our users indicate a need

#### DataFrame API: Pandas first tier
* one of `groupby` / `aggregate`: currently DataFrame has apply_spans functions but not aggregate; these must be used from Session
* `sort`: this is currently done by passing DataFrames to Session
* `describe`: this is a functionality that is beloved of data scientists at KCL and we should provide it
* various statistical measures and math functions that are available for the underlying numeric Numpy arrays

#### DataFrame API: Pandas second tier
* the other of  `groupby` / `aggregate`
* TODO: further grouping of Pandas functionality

#### DataFrame API: Pandas third tier
* TODO: further grouping of Pandas functionality

#### DataFrame API: Pandas fourth tier
* TODO: further grouping of Pandas functionality

### DataFrame Views

As all ExeTera data frames are backed up by the drive, filtering operations, either row or column-based, and sorting operations can quickly consume large amounts of memory, as well as being unnecessarily slow to complete. As such, ExeTera would benefit from the ability to present filtered views based on existing dataframes.

This is covered in more detail in DataFrame Views, but to summarise:
 * Filtering columns should always be able to be a view onto an existing DataFrame
 * Filtering rows should usually be a view onto an existing dataframe, with a corresponding filter fields
 * Sorting rows can potentially be a view onto an existing dataframe, but we may want some API by which the user prefers that to a hard copy of the DataFrame

 There is also the question as to whether a dataframe view is an explicit user requested thing or whether we also have them as optimisations that last until the user writes to a field or dataframe.



### Cleaner Syntax

We still have a number of areas where ExeTera's API can be argued to be unnecessarily complicated:
1. `.data` on Field
2. Use highly-scalable sorts by default
3. Move journalling up into DataFrame API
4. Moving functionality out of Session to the appropriate classes (Dataset / DataFrame / Field) and retiring old API

### Highly-scalable sorts

Highly scalable sorts are implemented in the ops level but not called by default in the DataFrame / Session API as yet. They should be incorporated into the DataFrame sort, although they are not currently required for datasets below around a billion rows.

### Move journalling up into DataFrame API

Journalling functionality should be incorporated into the DataFrame namespace, so that a journalled DataFrame can be generated by compatible dataframes. At the moment is must be accessed through the ops layer. For now this is just limited to journalling that functions on compatible tables. More advanced
journalling is detailed in the [data curation](#Data-curation) section.

## Serialization

### Filesystem-based serialized datastore

HDF5 has worked to provide us a good initial implementation for the serialized dataset, but it has a number of serious issues that require workarounds:
1. Data cannot be properly deleted from a HDF5 dataset
2. The format is fragile and interrupts at the wrong point can irretrievably corrupt a dataset
3. It has some restrictions on read-based concurrency
4. The `h5py` python library that wraps the underlying C library is very inefficient, particularly for iteration

These points are described in more detail in Roadmap: Move away from HDF5.


## Performance and scale

Performance and scale improvements can be delivered through several mechanisms:
1. Selection of the appropriate fast code generation technology (Numba, Cython or native C/C++)
2. Increase of the field API footprint, so that all operations can be performed scalably under the hood
3. Adoption of a multi-core/multi-node graph compiler / scheduler / executor to carry out ExeTera processing (Dask / Arrow / Spark)

### Graph-based compilation and execution

We need to move to some kind of graph-based compilation and execution engine in order to scale ExeTera to true multi-core / multi-node computation.

We could have an intermediate step in which we perform operations across fields using multiple cores, but this doesn't help memory pressure, and also adds read / write pressure due to increased reading / writing to drive.

Graph compilation and execution are very powerful mechanisms for multi-core execution in the general case, and the right implmentations scale to multiple nodes also. Graph compilation and execution can eliminate intermediate writes to drive where the user otherwise has to do them to manage working memory.

We have looked at Dask as a solution for this. Dask has great array primitives with the same rich set of operations as Numpy arrays have. For operations that are not directly supported by Dask arrays, we can write custom operators that perform the tasks that are currently performed under the hood on Numpy arrays.

In Dask, one typically sets up a series of statements such as:
```
c = a + b
d = c * e
f = 1 / d
```
Nothing is calculated until f is evaluated. This works well with ExeTera fields. Fields encapsulate the underlying arrays and give two natural points at which evaluation can be performed:
1. Writes of fields to DataFrames (provided that dataframes are always backed to the drive)
2. Evaluations of in memory fields e.g. `foo.data[:]`

Other graph computation and execution APIs exist. We can also consider Apache Arrows graph computation library, although it is a recent addition, or Spark, although it requires Java and is a more heavyweight proposition for a Python user who doesn't necessarily require libraries from the Java ecosystem.

## Data curation

This section covers the following data curation features:
* Schema discovery
* Full Journalling
* Algorithmic versioning

## Schema discovery
At present, ExeTera does not have schema discovery. This is because schema discovery cannot be completely performed, and has performance implications. Partial schema discovery is possible, however, and it would be good to have a schema discovery feature that allows the schema to be partially figured out and for the user to then be able to tweak fields afterwards where necessary. The complications we see for schema discovery are:
* ordering for categorical values, so that arithmetic can be carried out on them where the labels have some kind of ordered semantic
* 'leaky' categorical values where there are a set of standard entries and then a set of free text entries
* numeric fields where some values are empty
* numeric fields where some values are non-numeric

A good schema discovery workflow would allow data to be imported with as much schema as can be applied applied, with the ability to then check fields for partial / complete schema application subsequently

## Full Journalling
Journalling at present is limited only to compatible tables, and does not scale to handling a billion rows. Full journalling needs work on its scalability, and also needs to handle more complex scenarios in which the set of fields can also change. Furthermore, it needs the ability to set soft equality on some fields to account for slight rounding errors in sources for which a less than robust process is in place for data export.

## Algorithmic Versioning
Algorithms versioning should become a first-class concept in ExeTera. This allows algorithms to be effectively immutable once registered, so that a given script plus data always produces the same result, hardware notwithstanding. There are a number of ways that this could be done, but we have to be aware of the following confounding factors:
* random number generation
* concurrency

One potential mechanism is the ability to register python files as assets containing algorithms. Once registered, that algorithm can be copied in some fashion for subsequent dynamic loading. The user can register the same file subsequently; if changed, it will register a new version of the algorithm.
