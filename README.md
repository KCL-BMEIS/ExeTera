# ExeTera

Welcome to the ExeTera Readme!
This page and the accompanying github wiki show you how to make use of ExeTera to create reproducible
analysis pipelines for large tabular datasets.

Please take a moment to read this page, and also take a look at the [Wiki](https://github.com/KCL-BMEIS/ExeTera/wiki), which contains in-depth documentation on the concepts behind this software, usage examples, and developer resources such as the roadmap for future releases.

# Current release and requirements

[![Documentation Status](https://readthedocs.org/projects/exetera/badge/?version=latest)](https://exetera.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/exetera?label=PyPI%20version&logo=python&logoColor=white)](https://pypi.org/project/exetera/)
[![Testing](https://github.com/KCL-BMEIS/ExeTera/workflows/Unittests/badge.svg)](https://github.com/KCL-BMEIS/ExeTera/actions)
[![codecov](https://codecov.io/gh/KCL-BMEIS/ExeTera/branch/master/graph/badge.svg)](https://codecov.io/gh/KCL-BMEIS/ExeTera)

Requires python 3.7+

---
# Usage

The ExeTera allows you to import data from CSV sources into HDF5, a columnar data
format more suited to performing analytics. This is done through `exetera import`.


### `exetera import`
```
exetera import
-s path/to/covid_schema.json \
-i "patients:path/to/patient_data.csv, assessments:path/to/assessmentdata.csv, tests:path/to/covid_test_data.csv, diet:path/to/diet_study_data.csv" \
-o /path/to/output_dataset_name.hdf5
--include "patients:(id,country_code,blood_group), assessments:(id,patient_id,chest_pain)"
--exclude "tests:(country_code)"
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
 * `-w/--overwrite`: If set, overwrite any existing dataset with the same name; appends to existing dataset otherwise
 * `-n/--include`: If set, filters out all fields apart from those in the list.
 * `-x/--exclude`: If set, filters out the fields in this list.
 
Expect this script to take about an hour or more to execute on very large datasets.


## How do I work on the resulting dataset?
This is done through the python `exetera` API.

```python
from exetera.core.session import Session

with Session() as s:
    src = s.open_dataset('/path/to/my/source/dataset', 'r', 'src')
    dest = s.open_dataset('/path/to/my/result/dataset', 'w', 'dest')

    # code...
```


See the wiki for detailed examples of how to interact with the hdf5 datastore.


## Changes

The ChangeLog can now be found on the ExeTera [wiki](https://github.com/KCL-BMEIS/ExeTera/wiki/ChangeLog)
