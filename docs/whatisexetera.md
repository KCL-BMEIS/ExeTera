# What is ExeTera

ExeTera is a data curation and analysis software that enables users to work effectively with large, related tables of data. It focuses on three features which we consider essential to being able to effective curate and analyse data:
1. Scale
1. Journaling of snapshots
1. Curation features

## Scale

The python software ecosystem has provided amazing tools such as `numpy` and `pandas` that have made complex analysis of data available to a wide audience. These tools are used by countless individuals to analyse all manner of datasets and their accessibility and their ability to be integrated into the wider python ecosystem has made data science more accessible to all.

These tools come with a limitation, however, and that limitation is scale. These tools are not typically designed to work with more than the amount of RAM that a computer has. When dealing with large datasets, this places an effective upper limit on what can be processed and users typically have to resort to one or more measures to work around the problem:
 * Cut down the dataset somehow so that the subset is manageable
 * Buy a machine with more memory
 * Install a relational database such as `postgres` (or other type of datastore) and learn its API and how to maintain it

The problem of scale is surmountable. All of the operations required to analyse relational tables, such as sorts, joins, aggregations, and so forth have scalable implementations that can make use of a hard drive, and, if these operations are provided, it is possible to analyse data approaching terabyte scales on a typical laptop computer.
