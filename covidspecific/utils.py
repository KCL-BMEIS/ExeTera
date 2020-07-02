def iterate_over_patient_assessments(fields, filter_status, visitor):
    patient_ids = fields[1]
    cur_id = patient_ids[0]
    cur_start = 0
    cur_end = 0
    i = 1
    while i < len(filter_status):
        while patient_ids[i] == cur_id:
            cur_end = i
            i += 1
            if i >= len(filter_status):
                break

        visitor(fields, filter_status, cur_start, cur_end)

        if i < len(filter_status):
            cur_start = i
            cur_end = cur_start
            cur_id = patient_ids[i]


def iterate_over_patient_assessments2(patient_ids, filter_status, visitor):
    cur_id = patient_ids[0]
    cur_start = 0
    cur_end = 0
    i = 1
    while i < len(patient_ids):
        while patient_ids[i] == cur_id:
            cur_end = i
            i += 1
            if i >= len(patient_ids):
                break

        visitor(cur_id, filter_status, cur_start, cur_end)

        if i < len(patient_ids):
            cur_start = i
            cur_end = cur_start
            cur_id = patient_ids[i]
