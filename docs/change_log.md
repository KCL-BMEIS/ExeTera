# Change log

### v0.5.1 -> 0.5.2
 * BugFix: DataStore.get_spans returning None when passed Readers in legacy scripts. Functionality has been restored from v0.4

### v0.5.0 -> 0.5.1
 * Feature: DataFrame.rename function added; allows renaming of one or more fields within a dataframe

### v0.4 -> v0.5

 * Major changes to API
   * Datasets & DataFrames introduced
   * Rich API on Fields introduced
   * Much functionality previously accessed through Session can now be accessed through Datasets, DataFrames and Fields
   * See ([Basic Examples](https://github.com/KCL-BMEIS/ExeTera/wiki/Basic-Examples) and [Intermediate Examples](https://github.com/KCL-BMEIS/ExeTera/wiki/Intermediate-Examples) for more details
 * Import improvements
   * You can now specify include and exclude lists for fields in a table during import
     * This allows you to improve import performance and dataset size by excluding or only including the fields that you are interested in

### v0.3.2 -> v0.4

* Separation of all covid-specific functionality out to [https://github.com/KCL-BMEIS/ExeTeraCovid.git](https://github.com/KCL-BMEIS/ExeTeraCovid.git)
* Removal of legacy csv pipeline code
* Renaming of some of the `ordered_merge_*` functionality parameters for clarity
* Addition of `open/close/list/get_dataset` functionality to `Session`
* Made `Session` 'withable'
* Improved performance of `Session.get_spans`
* Bug fixes for Session API
  * apply_spans / aggregation issues
* Bug fixes for Field API
  * provided `__bool__` so that `if field:` works as expected
  * provided single element read for `IndexedStringField`


### v0.3.1 -> v0.3.2
* Fixing issues with use of test_type_from_mechanism_v1
* Adding ability to optionally import lsoa-based fields through add_imd script
* Import now appends by default; to overwrite an existing dataset use `-w` \ `--overwrite`
* Moved schema files to resources
* Adding separate lsoa schema for import

### v0.3.0 -> v0.3.1
* Major performance improvement to Session.get_spans

### v0.2.7 -> v0.3.0
* Renaming of hystore to ExeTera, the project's new name!
* Renaming of the `hystorex` command to `exetera`
* Removal of scripts that now belong in https://github.com/KCL-BMEIS/ExeTeraCovid.git
* Addition of snapshot journaling and extremely large sort functionality
* Removal of the legacy csv script functionality

### v0.2.7 -> v0.2.7.3
* Fix to covid_schema.json for numeric diet fields marked 'float' instead of 'float32'
* Addition of --daily flag to enable / disable generation of daily assessments
* Addition of

### v0.2.6 -> v0.2.7
* Addition of diet questionnaire schema
* Reworking of arguments for hystorex import to support arbitrary numbers and names of csvs
* Provision of highly-scalable merge functionality through ordered merge functions
  * Fix for filtering of indexed string fields

### v0.2.5 -> v0.2.6
* Moving from DataSet to Session class offering cleaner syntax
* Moving from Readers/Writers to Fields for cleaner syntax
* Introduction of schema for import command
* Consolidating commands
  * h5import -> hystorex import
  * h5process -> hystorex process

### v0.2.3 -> v0.2.5
* Please note: there was no version v0.2.4; due to a numbering error when updating the version number
* Simplifications to the API

### v0.2.2 -> v0.2.3
* Data schema updated for 1.5.1

### v0.2.1 -> v0.2.2
* Fix: Split functionality had not been moved to bin/csvsplit as documented
* Fix: Missing license headers added

### v0.2.0 -> v0.2.1 - tag
* Refactor: Created the `DataStore` class and moved `processor` api methods onto it as member
  functions
* Refactor: Simplified the creation of Writers. This can now be done through `get_writer` on
  a `DataStore` instance
* Fix: Writes to a hdf5 store can no longer be interrupted by interrupts, resulting in more
  stable hdf5 files
* Fix: Fixed critical bug in process method that resulted in exceptions when running on fields
  with a length that isn't an exact multiple of the chunksize

### v0.1.9 -> v0.2.0
* Added hdf5 import and process functionality

### v0.1.8 -> v0.1.9
* Feature: provision of the `split.py` script to split the dataset up into subsets of patients
  and their associated assessments
* Fix: added `treatments` and `other_symptoms` to cleaned assessment file. These fields are
  concatenated during the merge step using using csv-style delimiters and escapes

### v0.1.7 -> v0.1.8
* Fix: `had_covid_test` was not being patched up along with `tested_covid_positive`'
* Breaking change: output fields renamed
  * Fixed up `had_covid_test` is output as `had_covid_test_clean`
  * Fixed up `tested_covid_positive` is output as `tested_covid_positive_clean`
  * `had_covid_test` and `tested_covid_positive` contain the un-fixed-up data (although rows may
    still be modified as a result of quantising assessments by day)

### v0.1.6 -> v0.1.7
* Fix: `height_clean` contains weight data and `weight_clean` contains height data.
  This has been the case since they were introduced in v0.1.5

### v0.1.5 -> v0.1.6
* Performance: reduced memory usage
* Addition: provision of `-ps` flag for setting parsing schema

### v0.1.4 -> v0.1.5
* Fix: `health_status` was not being accumulated during the assessment compression phase of cleanup

### v0.1.3 -> v0.1.4
* Fix: added missing value `rarely_left_the_house_but_visit_lots` to `level_of_isolation`
* Fix: added missing fields `weight_clean`, `height_clean` and `bmi_clean`

### v0.1.2 -> v0.1.3
* Fix: `-po` and `-ao` options now properly export patient and assessment csvs respectively

### v0.1.1 -> v0.1.2
* Fix: `day` no longer overwriting `tested_covid_positive` on assessment export
* Fix: `tested_covid_positive` output as a label instead of a number

### v0.1 -> v0.1.1
* Change: Converted `'NA'` to `''` for csv export
