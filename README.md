The KCL covid19 joinzoe data preparation pipeline.

# Cleaning scripts

Current release version: v0.2.7.2

---
# Usage
The Zoe covid data preparation pipeline has two modes.

The first, and recommended, is to import CSV data into a HDF5 file, perform standard post import
processing on it, and then use the API to carry out further analysis.

The second is a legacy version of the pipeline that runs a cleaning script directly on the csv file
and outputs a modified csv file. This is provided for existing cleaning infrastructure but will not
be updated to match the hdf5 functionality.
 
## HDF5
The HDF5 analytics tools allow you to import data from CSV sources into HDF5, a columnar data
format more suited to performing analytics. This is done in two stages:
1. Import the data using `hystorex import`
2. Process the data using `hystorex postprocess` to create a set of useful additional data from the base data

### Why two stages?
Importing from CSV is a lengthy process that you may only want to do once. Splitting the work
between importing and processing means that the import can be done only once, and the imported file
used as a source for processing even if the processing functionality significantly changes.

### `hystorex import`
```
hystorex import
-s covid/covid_schema.json \
-i "patients:covid/patient_data.csv, assessments:covid/assessmentdata.csv, tests:covid/covid_test_data.csv, diet:covid/diet_study_data.csv" \
-o /home/ben/covid/ds_20200901_base.hdf5
```


#### Arguments
 * `-s/--schema`: The location and name of the schema file
 * `-te/--territories`: If set, this only imports the listed territories. If left unset, all
   territories are imported
 * `-i/--inputs` : A comma separated list of 'name:file' pairs. This should be put in parentheses if it contains any
  whitespace. See the example above.
 * `-o/--output_hdf5`: The path and name to where the resulting hdf5 dataset should be written
 * `-ts/--timestamp`: An override for the timestamp to be written
   (defaults to `datetime.now(timezone.utc)`)

Expect this script to take about an hour or so to execute.

### `hystorex process`
```
hystorex process -i <input_hdf5> -d -o <output_hdf5>
```
#### Arguments
 * `-i/--input`: The path and name of the import hdf5 file
 * `-o/--output`: The path and name of the processed hdf5 file
 * `-d/--daily`: A flag to indicate whether to generate daily assessments

## How do I work on the resulting dataset?
See the wiki for detailed examples of how to interact with the hdf5 datastore.

---
## Legacy csv
## `csvclean`

### Running the pipeline
```
csvclean -t <territory> -p <input patient csv> -a <input assessor csv> -po <output patient csv -ao <output assessor csv>
```

#### Arguments
 * `-t` / `--territory`: the territory to filter the dataset on (runs on all territories if not set)
 * `-p` / `--patient_data`: the location and name of the patient data csv file
 * `-a` / `--assessment_data`: the location and name of the assessment data csv file
 * `-po` / `--patient_data_out`: the location and name of the output patient data csv file
 * `-ao` / `--assessment_data_out`: the location and name of the output assessment data csv file
 * `-ps` / `--parsing_schema`: the schema number to use for parsing and cleaning data

### Pipeline help
```
python pipeline.py --help
```

### Pipeline version
```
python pipeline.py --version
```

#### parsing schema
There are currently 2 parsing schema versions:
* 1: The baseline cleaning behaviour, intented to match the R script
* 2 (in progress): Improved cleaning for height/weight/BMI and covid symptom progression

#### Including in a script

Use the function `pipeline()` from `pipeline.py`.

Proper documentation and packaging to follow

## `csvsplit`

### Running the split script
```
csvsplit -t <territory> -p <input patient csv> -a <input assessor csv> -b <bucket size>
```

The output of the split script is a series of patient and assessment files with the following structure:
```
<filename>.csv -> <filename>_<index>.csv
```
where index is the padded index of the subset (0000, 0001, 0002...).

#### options
 * `-p` / `--patient_data`: the location and name of the patient data csv file
 * `-a` / `--assessment_data`: the location and name of the assessment data csv file
 * `-b` / `--bucket_size`: the maximum number of patients to include in a subset

### Split script help
```
python split.py --help
```

### Split script version
```
python split.py --version
```

---
## Changes

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


