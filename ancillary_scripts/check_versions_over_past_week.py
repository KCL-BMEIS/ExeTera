from collections import defaultdict
from datetime import datetime, timezone

import dataset
import utils

afilename = '/home/ben/covid/assessments_export_20200604030001.csv'

akeys = ('patient_id', 'created_at', 'version')

date = datetime.now(timezone.utc)
with open(afilename) as f:
    ads = dataset.Dataset(f, keys=akeys, show_progress_every=1000000,
                          # stop_after=999999)
                          )
apids = ads.field_by_name('patient_id')
pcats = ads.field_by_name('created_at')
avsns = ads.field_by_name('version')
version_counts = defaultdict(int)
for i_r in range(ads.row_count()):
    if i_r % 1000000 == 0:
        print(i_r)
    if (date - utils.timestamp_to_datetime(pcats[i_r])).days <= 7:
        version_counts[avsns[i_r]] += 1

print(version_counts)