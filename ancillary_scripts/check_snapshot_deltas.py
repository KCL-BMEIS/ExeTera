import dataset


def check_deltas(filename1, filename2, fields_to_compare, show_progress_every=None):

    with open(filename1) as f1:
        ds1 = dataset.Dataset(f1, stop_after=1)
    names1 = set(ds1.names_)
    with open(filename2) as f2:
        ds2 = dataset.Dataset(f2, stop_after=1)
    names2 = set(ds2.names_)
    print('only_in_f1:', names1.difference(names2))
    print('only_in_f2:', names2.difference(names1))
    print(f'loading {filename1}')
    with open(filename1) as f1:
        ds1 = dataset.Dataset(f1, keys=fields_to_compare,
                              show_progress_every=show_progress_every)
    ds1.sort(('created_at', 'id'))
    ids1 = ds1.field_by_name('id')
    cats1 = ds1.field_by_name('created_at')
    uats1 = ds1.field_by_name('updated_at')

    print(f'loading {filename2}')
    with open(filename2) as f2:
        ds2 = dataset.Dataset(f2, keys=fields_to_compare,
                              show_progress_every=show_progress_every)
    ds2.sort(('created_at', 'id'))
    ids2 = ds1.field_by_name('id')
    cats2 = ds1.field_by_name('created_at')
    uats2 = ds1.field_by_name('updated_at')

    mismatches = 0
    updated = 0
    i = 0
    j = 0
    while i < ds1.row_count() and j < ds2.row_count():
        if show_progress_every is not None:
            if i % show_progress_every == 0 or j % show_progress_every == 0:
                print(i, j, mismatches, updated)
        inci = False
        incj = False
        if cats1[i] < cats2[j]:
            inci = True
        elif cats1[i] > cats2[j]:
            incj = True
        else:
            if ids1[i] < ids2[j]:
                inci = True
            elif ids1[i] > ids2[j]:
                incj = True
            else:
                inci = True
                incj = True

        if not inci or not incj:
            mismatches += 1
        else:
            if uats1[i] != uats2[j]:
                updated += 1

        if inci:
            i += 1
        if incj:
            j += 1
    if i < ds1.row_count():
        mismatches += ds1.row_count() - i
    if j < ds2.row_count():
        mismatches += ds2.row_count() - j
    print(mismatches, updated)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-f1', required=True)
    parser.add_argument('-f2', required=True)
    args = parser.parse_args()

    check_deltas(args.f1, args.f2, ('id', 'created_at', 'updated_at'), 1000000)
