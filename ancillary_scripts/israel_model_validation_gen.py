from collections import defaultdict
import dataset

"""
* (year_of_birth) Age
* (gender) Gender
* Any of the prior medical conditions:
  * (has_diabetes) Diabetes mellitus
  * (takes_blood_pressure_medications_pril, takes_any_blood_pressure_medications,
    takes_blood_pressure_medications_sartan) Hypertension
  * (has_heart_disease) Cardiovascular disease
  * (has_lung_disease) Chronic lung disease
  * (has_kidney_disease) Chronic kidney disease,
  * (has_cancer or takes_immunosuppressants) Malignancy (cancer) or Immunodeficiency

* General feeling
* Sore throat
* Cough
* Shortness of breath
* Smell or taste loss
* Fever (over 38 degrees celicius)
"""

pfilename = '/home/ben/covid/patients_export_geocodes_20200601030001.csv'
pkeys = ('id', 'year_of_birth', 'gender', 'has_cancer', 'has_diabetes',
         'takes_blood_pressure_medications_pril', 'takes_any_blood_pressure_medications',
         'takes_blood_pressure_medications_sartan', 'has_heart_disease', 'has_lung_disease',
         'has_kidney_disease', 'has_cancer', 'takes_immunosuppressants', 'past_symptom_fever')
with open(pfilename) as f:
    pds = dataset.Dataset(f, keys=pkeys, show_progress_every=1000000,
                          # stop_after=999999)
                          )
print(pds.names_)
patients = set()
pids = pds.field_by_name('id')
ppsfs = pds.field_by_name('past_symptom_fever')
fever_count = 0
for i_r in range(pds.row_count()):
    patients.add(pids[i_r])
    if ppsfs[i_r]:
        fever_count += 1
print(fever_count)
del pds
del pids
del ppsfs

afilename = '/home/ben/covid/assessments_export_20200601030001.csv'
akeys = ('id', 'patient_id', 'fever')
with open(afilename) as f:
    # ads = dataset.Dataset(f, keys=akeys, show_progress_every=1000000)
    ads = dataset.Dataset(f, keys=akeys, show_progress_every=1000000)
print(ads.names_)



class PatientHadFever:
    def __init__(self):
        self.had_fever = False
    def add(self, had_fever):
        self.had_fever = self.had_fever or had_fever

apatientshadfever = defaultdict(PatientHadFever)

a_pids = ads.field_by_name('patient_id')
a_hfs = ads.field_by_name('fever')
for i_r in range(ads.row_count()):
    if a_pids[i_r] in patients:
        apatientshadfever[a_pids[i_r]].add(a_hfs[i_r])
print(len(apatientshadfever))

had_fever_count = 0
for k, v in apatientshadfever.items():
    if v.had_fever:
        had_fever_count += 1
print(had_fever_count)
