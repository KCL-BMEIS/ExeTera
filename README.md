The KCL covid19 joinzoe data preparation pipeline.

## Pipeline help
```
python pipeline.py --help
```

## Running the pipeline
```
python pipeline.py -t <territory> -p <input patient csv> -a <input assessor csv> -po <output patient csv -ao <output assessor csv>
```


## Including in a script

Use the function `pipeline()` from `pipeline.py`.

Proper documentation and packaging to follow


## Changes

### v0.1 -> v0.1.1
* Change: Converted `'NA'` to `''` for csv export

### v0.1.1 -> v0.1.2
* Fix: `day` no longer overwriting `tested_covid_positive` on assessment export
* Fix: `tested_covid_positive` output as a label instead of a number

### v0.1.2 -> v0.1.3
* Fix: `-po` and `-ao` options now properly export patient and assessment csvs respectively
