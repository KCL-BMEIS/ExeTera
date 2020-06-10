from collections import defaultdict

import dataset
import utils
import analytics


filename = '/home/ben/covid/assessments_export_20200527030002.csv'


with open(filename) as f:
    a_ds = dataset.Dataset(f, keys=('id', 'patient_id', 'created_at', 'updated_at', 'health_status'),
                           stop_after=5000000,
                           show_progress_every=5000000)

print('sorting')
a_ds.sort(('patient_id', 'created_at'))
# a_ds.sort(('created_at',))
a_pids = a_ds.field_by_name('patient_id')
a_cats = a_ds.field_by_name('created_at')
a_hlts = a_ds.field_by_name('health_status')

print('analysing')
print(utils.build_histogram(a_hlts))
patients = defaultdict(analytics.TestIndices)

for i_r in range(a_ds.row_count()):
    patients[a_pids[i_r]].add(i_r)

class FirstHealthStatus:
    def __init__(self):
        self.never_unhealthy = 0
        self.initially_unhealthy = 0
        self.eventually_unhealthy = 0

    def add(self, value):
        if value == 0:
            self.never_unhealthy += 1
        elif value == 1:
            self.eventually_unhealthy += 1
        else:
            self.initially_unhealthy += 1

    def __repr__(self):
        return f"{self.never_unhealthy}, {self.eventually_unhealthy}, {self.initially_unhealthy}"

by_day = defaultdict(FirstHealthStatus)

# first_entry_not_healthy = 0
all_statuses = [0, 0, 0]

for pk, pv in patients.items():
    first_unhealthy = False
    ever_unhealthy = False
    first_day = utils.timestamp_to_day(a_cats[pv.indices[0]])
    if a_hlts[pv.indices[0]] == 'not_healthy':
        first_unhealthy = True
        ever_unhealthy = True
    for i in range(1, len(pv.indices)):
        if a_hlts[pv.indices[i]] == 'not_healthy':
            ever_unhealthy = True
            break
    by_day[first_day].add(first_unhealthy + ever_unhealthy)
    all_statuses[first_unhealthy + ever_unhealthy] += 1
#     first_entry_not_healthy += 1

print(all_statuses)
by_day_entries = sorted(list(by_day.items()))
verified = [0, 0, 0]
for e in by_day_entries:
    print(e)
    verified[0] += e[1].never_unhealthy
    verified[1] += e[1].eventually_unhealthy
    verified[2] += e[1].initially_unhealthy
print(verified)
# print(first_entry_not_healthy)