import csv

from exetera.core.dataset import Dataset

pfilename = "/home/ben/covid/patients_export_geocodes_20200830040058.csv"
afilename = "/home/ben/covid/assessments_export_20200830040058.csv"
tfilename = "/home/ben/covid/covid_test_export_20200830040058.csv"
dfilename = "/home/ben/covid/diet_study_export_20200830040058.csv"

print("columns")
for fn in (pfilename, afilename, tfilename, dfilename):
    with open(fn) as f:
        csvr = iter(csv.reader(f))
        row = next(csvr)
        print(" ", pfilename, len(row))
        print(row)


print("rows")
for fn in (pfilename, afilename, tfilename, dfilename):
    with open(fn) as f:
        print(" ", pfilename)
        ds = Dataset(f, keys=('id',), show_progress_every=5000000)
        print(" ", ds.row_count())
