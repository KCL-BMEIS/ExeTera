# Import CSV files

The ExeTera allows you to import data from CSV sources into HDF5, a columnar data
format more suited to performing analytics.

## Importing via the exetera import command
Example:
```
exetera import
-s path/to/covid_schema.json \
-i "patients:path/to/patient_data.csv, assessments:path/to/assessmentdata.csv, tests:path/to/covid_test_data.csv, diet:path/to/diet_study_data.csv" \
-o /path/to/output_dataset_name.hdf5
--include "patients:(id,country_code,blood_group), assessments:(id,patient_id,chest_pain)"
--exclude "tests:(country_code)"
```

## #Arguments
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


## Importing through code

Example:
```python

importer.import_with_schema(timestamp, output_hdf5_name, schema, tokens, args.overwrite, include_fields, exclude_fields)
```

See the wiki for detailed examples of how to interact with the hdf5 datastore.
