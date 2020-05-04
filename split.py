import csv

import dataset
import utils

# read patients in batches of n
# read assessments for those pages and output them to n

def patient_splitter(input_filename, output_filenames, output_ranges):
    ch_del = ','
    ch_quote = '"'
    rows_parsed = 0
    with open(input_filename) as f_i:
        csvdr = csv.DictReader(f_i, delimiter=ch_del, quotechar=ch_quote)
        keys = csvdr.fieldnames
        csvr = csv.reader(f_i, delimiter=ch_del, quotechar=ch_quote)

        for subset in range(len(output_filenames)):
            output_filename = output_filenames[subset]
            limit = output_ranges[subset]
            with open(output_filename, 'w') as f_o:
                csvw = csv.writer(f_o, delimiter=ch_del, quotechar=ch_quote)
                csvw.writerow(keys)
                while rows_parsed < limit:
                    r = next(csvr)
                    if rows_parsed > 0 and rows_parsed % 100000 == 0:
                        print(f"{rows_parsed}")
                    csvw.writerow(r)
                    rows_parsed += 1
                start = 0 if subset == 0 else output_ranges[subset - 1]
                print(f"complete: {rows_parsed - start}")

def patient_splitter_2(input_filename, output_filenames, sorted_indices, bucket_size):
    ch_del = ','
    ch_quote = '"'
    rows_parsed = 0
    with open(input_filename) as f_i:
        csvr = csv.reader(f_i, delimiter=ch_del, quotechar=ch_quote)

        keys = next(csvr)

        input_rows = list()
        for r in csvr:
            input_rows.append(r)

    remaining_rows = len(input_rows)

    accumulated = 0
    for ofn in output_filenames:
        with open(ofn, 'w') as f_o:
            print("writing", ofn)
            csvw = csv.writer(f_o, delimiter=ch_del, quotechar=ch_quote)

            csvw.writerow(keys)

            for i_r in range(min(bucket_size, remaining_rows)):
                ind = sorted_indices[accumulated + i_r]
                csvw.writerow(input_rows[ind])
            accumulated += bucket_size
            remaining_rows -= bucket_size




def assessment_splitter(input_filename, output_filename, assessment_buckets, bucket):
    ch_del = ','
    ch_quote = '"'
    rows_parsed = 0
    rows_written = 0
    with open(input_filename) as f_i:
        with open(output_filename, 'w') as f_o:
            csvdr = csv.DictReader(f_i, delimiter=ch_del, quotechar=ch_quote)
            keys = csvdr.fieldnames
            csvr = csv.reader(f_i, delimiter=ch_del, quotechar=ch_quote)

            csvw = csv.writer(f_o, delimiter=ch_del, quotechar=ch_quote)
            csvw.writerow(keys)

            for r in csvr:
                if rows_parsed > 0 and rows_parsed % 100000 == 0:
                    print(f"{rows_parsed} ({rows_written})")
                if assessment_buckets[rows_parsed] == bucket:
                    csvw.writerow(r)
                    rows_written += 1
                rows_parsed += 1

    print(f"complete: {rows_parsed} ({rows_written})")


def split_data(patient_data, assessment_data, bucket_size=500000):

    with open(patient_data) as f:
        p_ds = dataset.Dataset(f, keys=('id', 'created_at'),
                               progress=True)
                               # progress=True, stop_after=500000)
        p_ds.sort(('created_at', 'id'))
        p_ids = p_ds.field_by_name('id')
        p_dts = p_ds.field_by_name('created_at')

    # put ids into buckets
    #bucket_size = 1 << 20
    buckets = dict()
    bucket_index = 0
    bucket_count = 0
    for i_r in range(p_ds.row_count()):
        if bucket_index == bucket_size:
            bucket_index = 0
            bucket_count += 1
        buckets[p_ids[i_r]] = bucket_count
        bucket_index += 1

    ranges = list()
    acc_row_count = bucket_size
    while acc_row_count <= p_ds.row_count():
        ranges.append(acc_row_count)
        acc_row_count += bucket_size
    if ranges[-1] < p_ds.row_count():
        ranges.append(p_ds.row_count())
    print(ranges)

    filenames = list()
    for b in range(bucket_count+1):
        destination_filename = patient_data[:-4] + f"_{b:04d}" + ".csv"
        filenames.append(destination_filename)
    print(filenames)
    sorted_indices = p_ds.index_
    del p_ds

    patient_splitter_2(patient_data, filenames, sorted_indices, bucket_size)

    print('buckets:', bucket_index)
    with open(assessment_data) as f:
        a_ds = dataset.Dataset(f, keys=('patient_id', 'other_symptoms'), progress=True)

    print(utils.build_histogram(buckets.values()))

    print('associating assessments with patients')
    orphaned_assessments = 0
    a_buckets = list()
    a_pids = a_ds.field_by_name('patient_id')
    a_os = a_ds.field_by_name('other_symptoms')
    for i_r in range(a_ds.row_count()):
        if a_pids[i_r] in buckets:
            a_buckets.append(buckets[a_pids[i_r]])
        else:
            orphaned_assessments += 1
            a_buckets.append(-1)

    del a_ds
    print('orphaned_assessments:', orphaned_assessments)

    print(f'{bucket_count + 1} buckets')
    for i in range(bucket_count + 1):
        print('bucket', i)
        destination_filename = assessment_data[:-4] + f"_{i:04d}" + ".csv"
        print(destination_filename)
        # with open(assessment_data) as f:
        #     a_ds = dataset.Dataset(f, filter_fn=lambda j: a_buckets[j] == i, progress=True)
        #
        # del a_ds
        assessment_splitter(assessment_data, destination_filename, a_buckets, i)

    print('done!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patient_data',
                        help='the location and name of the patient data csv file')
    parser.add_argument('-a', '--assessment_data',
                        help='the location and name of the assessment data csv file')
    parser.add_argument('-b', '--bucket_size', type=int, default=500000,
                        help='the number of patients to include in a bucket')

    args = parser.parse_args()
    if args.bucket_size < 10000:
        print('--bucket_size cannot be less than 10000')
        exit(-1)

    try:
        split_data(args.patient_data, args.assessment_data, args.bucket_size)
    except Exception as e:
        print(e)
        exit(-1)
