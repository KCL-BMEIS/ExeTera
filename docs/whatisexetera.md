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

## Journaling

It is typically the case that data is provided in multiple snapshots over time. It is possible to compare snapshots and perform longitudinal analyses of the dataset, but this is even more memory-intensive than analysing a single snapshot. For the most part, the snapshots are likely to contain mainly additions to the dataset and only a small number of modifications (new users vs. user address changes, for example), and by starting with the first snapshot and only recording the differences between the snapshots, the dataset can be stored using much less memory than storing both full snapshots.

## Curation

Being able to load and analyse the data is a precondition for making analysis possible, but reliable, reproducible analytics requires more than this. Full reproducibility requires that both data and algorithms are treated as immutable. For algorithms, this requires that, once a version of an algorithm is released, any changes to the algorithm result in a new version of that algorithm. Furthermore, maintaining of current and historical versions of algorithms allows a comparison between them to be done when evaluating which version to use. This includes algorithms that turn out to be erroneous for some reason; being able to run both the erroneous version of an algorithm and its fixed version allows the user to check how the error has affected existing analyses.
