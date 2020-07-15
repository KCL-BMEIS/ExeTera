import csv
import pandas

from core import utils, persistence


def export_to_csv(destination, datastore, fields):

    expected_length = None
    for f in fields:
        if expected_length is None:
            expected_length = len(datastore.get_reader(f))
        else:
            length = len(datastore.get_reader(f))
            if length != expected_length:
                raise ValueError(f"field '{f}' is not the same length as the first field:"
                                 f"expected {expected_length} but got {length}")
    print('expected_length:', expected_length)
    readers = [datastore.get_reader(f) for f in fields]

    with open(destination, 'w') as d:
        csvw = csv.writer(d)
        #header
        header = [None] * len(fields)
        for i_f, f in enumerate(fields):
            header[i_f] = f.name.split('/')[-1]
        csvw.writerow(header)

        row = [None] * len(fields)
        for chunk in datastore.chunks(expected_length):
            s = slice(chunk[0], chunk[1])
            print(s)
            chunks = [r[s] for r in readers]
            for i_r in range(chunk[1] - chunk[0]):
                for i_c, c in enumerate(chunks):
                    if isinstance(readers[i_c], persistence.FixedStringReader):
                        row[i_c] = c[i_r].decode()
                    else:
                        row[i_c] = c[i_r]
                csvw.writerow(row)
