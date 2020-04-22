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

### v0.1.3 -> v0.1.4
* Fix: added missing value `rarely_left_the_house_but_visit_lots` to `level_of_isolation`
* Fix: added missing fields `weight_clean`, `height_clean` and `bmi_clean`

### v0.1.4 -> v0.1.5
* Fix: `health_status` was not being accumulated during the assessment compression phase of cleanup
