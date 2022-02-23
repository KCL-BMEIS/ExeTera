## Examples of how to use ExeTera

This folder contains a few examples on how to use ExeTera in different scenarios.

#### Names dataset
This example shows how to generate ExeTera HDF5 datafile through 'importer.import_with_schema' function, and a few basic commands to print the dataset content.

#### import_dataset
This example shows how to import multiple CSV files into a ExeTera HDF5 datafile. The example datafile has a similar structure to Covid Symptom Study (CSS) dataset, including a user table and a assessments table.


#### basic_concept
This example shows how to use ExeTera, through the major components: dataset, dataframe and fields. Please note this example is based on assessments.hdf5 file, hence please go through the simple_linked_dataset example and generate the hdf5 file first.


#### advanced_operations
This example shows the intermediate functions of ExeTera, such as filtering, group by, sorting, performance boosting using numba, and output the dataframe to csv file. Please note this example is based on assessments.hdf5 file, hence please go through the simple_linked_dataset example and generate the hdf5 file first.
