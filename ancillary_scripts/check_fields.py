import dataset
import pipeline

filename = 'test_patients.py'
ds = dataset.Dataset(filename)

for ir in range(ds.row_count()):
    pipeline.print_diagnostic_row(f'{ir}', ds, ir,
                                  ('patient_id', 'weight_kg', 'height_cm', 'bmi', 'weight_clean', 'height_clean', 'bmi_clean'))
